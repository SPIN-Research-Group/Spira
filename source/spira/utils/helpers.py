__all__ = [
    'as_tuple', 'make_divisible', 'tune',
]

import contextlib
import json
import os.path
from typing import Iterable, Optional, List, Tuple, Dict, Union
from tqdm import tqdm
import numpy as np
from torch.nn import Module
import functools
import torch
import time
import itertools
from collections import defaultdict
from spira.utils.typing import ScalarOrTuple, ScalarOrIterable


def set_kernel_map_cache(module: Module, cache: 'spira.nn.KernelMapCache'):
  if hasattr(module, "set_kernel_map_cache"):
    module.set_kernel_map_cache(cache)

  for submodule in module.children():
    set_kernel_map_cache(submodule, cache=cache)

  return module

def as_tuple(value: ScalarOrTuple,
             *,
             size: int = 3,
             name: Optional[str] = None):

  name = name or "Value"
  if not isinstance(value, Iterable):
    value = tuple(value for _ in range(size))
    return value

  value = tuple(value)
  if len(value) != size:
    raise ValueError(f"{name} must be a scalar or a iterable of size {size} "
                     f"but found {len(value)} elements")
  return value

# ============================================================================
# STABLE TIME ACCUMULATOR
# ============================================================================

class StableTimeAccumulator:
    """Accumulates timing measurements while filtering outliers."""
    def __init__(self):
        self.trial_count = 0
        self.average_time = 0.0

    def add_measurement(self, measured_time: float):
        """Add a timing measurement, filtering outliers (>5x current average)."""
        if measured_time <= 0:
            return

        if self.trial_count == 0:
            self.average_time = measured_time
            self.trial_count = 1
        else:
            if measured_time <= 5 * self.average_time:
                self.average_time = (
                    (self.trial_count * self.average_time) + measured_time
                ) / (self.trial_count + 1)
                self.trial_count += 1

    def get_time(self):
        return self.average_time

    def get_trial_count(self):
        return self.trial_count


# ============================================================================
# DATAFLOW CONFIGURATION
# ============================================================================

class DataflowConfig:
    """Configuration for SparseConv dataflow."""
    def __init__(self, map_dataflow: int, force_os: bool = False):
        self.map_dataflow = map_dataflow
        self.force_os = force_os

    def apply_to_layer(self, layer):
        """Apply this configuration to a SparseConv layer."""
        layer._tunable_config['map_dataflow'] = self.map_dataflow
        layer._tunable_config['force_os'] = self.force_os

    def __repr__(self):
        return f"DataflowConfig(map_dataflow={self.map_dataflow}, force_os={self.force_os})"


def set_config_for_group(model: Module,
                        group_layers: List[str],
                        config: DataflowConfig):

    # Calculate how many layers get force_os=True (first half)
    num_layers = len(group_layers)
    num_force_os = num_layers // 2  # Integer division - first half

    for idx, name in enumerate(group_layers):
        for module_name, module in model.named_modules():
            if module_name == name:
                # Determine if this layer should get force_os=True
                if config.force_os and idx < num_force_os:
                    # First half: apply force_os=True
                    module._tunable_config['map_dataflow'] = config.map_dataflow
                    module._tunable_config['force_os'] = True
                else:
                    # force_os=False for entire group
                    module._tunable_config['map_dataflow'] = config.map_dataflow
                    module._tunable_config['force_os'] = False
                break

# ============================================================================
# TIMING MEASUREMENT
# ============================================================================

def measure_model_time(model: Module,
                      input_data,
                      kernel_map_cache=None) -> float:

    # Reset cache to force recomputation
    if kernel_map_cache is not None:
        kernel_map_cache.reset()

    assert len(kernel_map_cache._kernel_maps_cache) == 0, \
    "Kernel map cache was not properly reset"


    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        _ = model(input_data)

    torch.cuda.synchronize()
    end_time = time.time()

    return end_time - start_time


# ============================================================================
# TUNING LOGIC
# ============================================================================


def tune_dataflows(model: Module,
                   data_loader: Iterable,
                  group_to_name: Dict[Tuple, List[str]],
                  config_options: List[DataflowConfig],
                  kernel_map_cache,
                  n_samples: int = 10,
                  verbose: bool = True) -> Dict[Tuple, DataflowConfig]:

    from spira.nn import SparseConv

    # ✅ SAVE tune_gs results BEFORE testing dataflows
    tuned_gs_params = {}
    for name, module in model.named_modules():
        if isinstance(module, SparseConv):
            tuned_gs_params[name] = {
                'gather_tile_size': module._tunable_config['gather_tile_size'],
                'scatter_tile_size': module._tunable_config['scatter_tile_size'],
                'threshold': module._tunable_config['threshold']
            }

    best_configs = {}

    for group_id, layer_names in tqdm(group_to_name.items(),
                                      desc="Tuning groups",
                                      disable=not verbose):

        kernel_size = group_id[1]
        layer_stride = group_id[2]

        if all(s == 1 for s in layer_stride) and all(k == 1 for k in kernel_size):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Skipping trivial group: {group_id}")
                print(f"  Layers ({len(layer_names)}): {layer_names}")
                print(f"  Reason: All strides=1 and all kernels=1 (trivial convolution)")
                print(f"{'='*70}\n")

            continue

        if any(s > 1 for s in layer_stride):
            # This is a downsampling layer - no hybrid dataflows
            valid_configs = [
                cfg for cfg in config_options
                if cfg.map_dataflow in {0, 1, 2}
            ]
            if verbose and len(valid_configs) < len(config_options):
                print(f"\nNote: Layer stride {layer_stride} > 1 (downsampling)")
                print(f"      Restricting to dataflows 0-2 only")
        else:
            # Regular layer (stride = 1), allow all dataflows
            valid_configs = config_options

        config_times = defaultdict(StableTimeAccumulator)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Tuning group: {group_id}")
            print(f"  Layers ({len(layer_names)}): {layer_names}")
            print(f"  Testing {len(valid_configs)} configurations")
            print(f"{'='*70}")

        for config in valid_configs:
            if verbose:
                print(f"  Testing: {config}")

            # ✅ RESTORE tune_gs params before setting dataflow
            for name in layer_names:
                for module_name, module in model.named_modules():
                    if module_name == name and isinstance(module, SparseConv):
                        if name in tuned_gs_params:
                            module._tunable_config['gather_tile_size'] = tuned_gs_params[name]['gather_tile_size']
                            module._tunable_config['scatter_tile_size'] = tuned_gs_params[name]['scatter_tile_size']
                            module._tunable_config['threshold'] = tuned_gs_params[name]['threshold']
                        break

            set_config_for_group(model, layer_names, config)

            sample_count = 0
            for sample_idx, input_data in enumerate(data_loader):
                # Measure time on this individual sample (with cache reset)
                elapsed_time = measure_model_time(model, input_data, kernel_map_cache)
                if sample_idx > 2:
                    config_times[str(config)].add_measurement(elapsed_time)

                sample_count += 1
                if sample_count >= n_samples:
                    break

            avg_time = config_times[str(config)].get_time()
            trial_count = config_times[str(config)].get_trial_count()
            if verbose:
                print(f"    Average time: {avg_time*1000:.3f} ms (over {trial_count} samples)")

        # Select the fastest configuration for this group
        best_config_str = min(config_times.keys(),
                             key=lambda cfg: config_times[cfg].get_time())

        best_config = None
        best_time = float('inf')
        for config in valid_configs:
            if str(config) == best_config_str:
                best_config = config
                best_time = config_times[best_config_str].get_time()
                break

        best_configs[group_id] = best_config

        if verbose:
            print(f"\n  ✓ Best configuration: {best_config}")
            print(f"    Time: {best_time*1000:.3f} ms")
            print()

    # ✅ RESTORE tune_gs params one final time after all testing
    for name, module in model.named_modules():
        if isinstance(module, SparseConv) and name in tuned_gs_params:
            module._tunable_config['gather_tile_size'] = tuned_gs_params[name]['gather_tile_size']
            module._tunable_config['scatter_tile_size'] = tuned_gs_params[name]['scatter_tile_size']
            module._tunable_config['threshold'] = tuned_gs_params[name]['threshold']

    return best_configs


def tune_gs(model: Module,
            kernel_map_cache,
            data_loader,
            n_samples: int = 10):

  from spira.nn import SparseConv

  with contextlib.ExitStack() as stack:

    def _tune_module(module: Module):
      if isinstance(module, SparseConv):
        stack.enter_context(module.tune_gs())

      for submodule in module.children():
        _tune_module(submodule)

    _tune_module(model)

    sample_count = 0
    for input_data in data_loader:

        kernel_map_cache.reset()
        _ = model(input_data)

        sample_count += 1
        if sample_count >= n_samples:
            break


# ============================================================================
# APPLY TUNED CONFIGURATIONS
# ============================================================================

def apply_tuned_configs(model: Module,
                       name_to_group: Dict[str, Tuple],
                       best_configs: Dict[Tuple, DataflowConfig],
                       layer_configs: Dict[str, Dict] = None):  # ✅ Add layer_configs parameter

    from spira.nn import SparseConv

    for name, module in model.named_modules():
        if not isinstance(module, SparseConv):
            continue

        # ✅ Priority 1: Use per-layer config if available (includes tune_gs results)
        if layer_configs and name in layer_configs:
            config = layer_configs[name]
            #module._tunable_config['map_dataflow'] = config['map_dataflow']
            #module._tunable_config['force_os'] = config['force_os']
            module._tunable_config['gather_tile_size'] = config['gather_tile_size']
            module._tunable_config['scatter_tile_size'] = config['scatter_tile_size']
            module._tunable_config['threshold'] = config['threshold']

        # ✅ Priority 2: Fall back to group-level config (backwards compatible)
        group_to_layers = defaultdict(list)
        for name, gid in name_to_group.items():
            group_to_layers[gid].append(name)

        for gid, layers in group_to_layers.items():
            if gid in best_configs:
                set_config_for_group(model, layers, best_configs[gid])


# ============================================================================
# SAVE AND LOAD CONFIGURATIONS
# ============================================================================

def save_tuning_results(name_to_group: Dict,
                       best_configs: Dict[Tuple, DataflowConfig],
                       save_path: str,
                       model: Module):
    """Save tuning results to disk."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # ✅ Save FULL config for each individual layer
    from spira.nn import SparseConv
    layer_configs = {}
    for name, module in model.named_modules():
        if isinstance(module, SparseConv):
            layer_configs[name]= module._tunable_config.copy()


    # Keep group configs for backwards compatibility / summary
    serializable_group_configs = {
        k: {'map_dataflow': v.map_dataflow, 'force_os': v.force_os}
        for k, v in best_configs.items()
    }

    torch.save({
        'name_to_group': name_to_group,
        'best_configs': serializable_group_configs,  # Group-level (for summary)
        'layer_configs': layer_configs                # ✅ Layer-level (actual configs)
    }, save_path)



def load_tuning_results(load_path: str) -> Tuple[Dict, Dict[Tuple, DataflowConfig], Dict]:
    """Load previously saved tuning results."""
    data = torch.load(load_path)

    best_configs = {
        k: DataflowConfig(v['map_dataflow'], v['force_os'])
        for k, v in data['best_configs'].items()
    }

    # Load per-layer configs
    layer_configs = data.get('layer_configs', {})

    return data['name_to_group'], best_configs, layer_configs


# ============================================================================
# MAIN TUNE FUNCTION
# ============================================================================

def tune(model: Module,
         data_loader: Iterable,
         kernel_map_cache=None,
         map_dataflow_options: List[int] = None,
         test_force_os: bool = True,
         n_samples: int = 10,
         n_samples_for_gs: int = 1,
         save_dir: str = ".tune_configs",
         tune_tag: str = "default",
         force_retune: bool = False,
         allow_gs_tuning:bool = True,          #ResNets do not need this if we want speed
         verbose: bool = False,
         return_dataflows: bool = False) -> Union[Module, Tuple[Module, Dict]]:

    from spira.nn import SparseConv

    if map_dataflow_options is None:
        map_dataflow_options = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    force_os_allowed = {1, 2, 3, 4}

    data_loader_fn = list(itertools.islice(data_loader, n_samples+1))


    # Generate all configuration combinations to test
    config_options = []
    for map_dataflow in map_dataflow_options:
        # Always test force_os = False
        config_options.append(
            DataflowConfig(map_dataflow=map_dataflow, force_os=False)
        )

        # Test force_os = True only for allowed dataflows
        if test_force_os and map_dataflow in force_os_allowed:
            config_options.append(
                DataflowConfig(map_dataflow=map_dataflow, force_os=True)
            )

    # ========================================================================
    # CHECK FOR CACHED RESULTS
    # ========================================================================

    save_path = os.path.join(save_dir, f"{tune_tag}.pt")

    if os.path.exists(save_path) and not force_retune:
        if verbose:
            print(f"Loading cached tuning results from: {save_path}")

        try:
            name_to_group, best_configs, layer_configs = load_tuning_results(save_path)
            apply_tuned_configs(model, name_to_group, best_configs, layer_configs)  # ✅ Pass layer_configs

            if verbose:
                print("✓ Tuning configurations applied from cache")
                print("\nCached configuration summary:")
                for group_id, config in best_configs.items():
                    print(f"  Group {group_id}: {config}")


            group_to_map_dataflow = {
                group_id: config.map_dataflow
                for group_id, config in best_configs.items()
            }
            if return_dataflows:
                return model, group_to_map_dataflow
            return model

        except Exception as e:
            if verbose:
                print(f"Warning: Could not load cache ({e}). Re-tuning...")

    # ========================================================================
    # NO CACHE - START TUNING
    # ========================================================================


    if verbose:
        print("\n" + "="*80)
        print("SPARSECONV AUTO-TUNER (with Shared KernelMapCache)")
        print("="*80)
        print(f"Model: {type(model).__name__}")
        print(f"Testing map_dataflow: {map_dataflow_options}")
        print(f"Testing force_os: {test_force_os}")
        print(f"Total configurations: {len(config_options)}")
        print(f"Samples per test: {n_samples}")
        if kernel_map_cache is not None:
            print(f"KernelMapCache: {type(kernel_map_cache).__name__} (shared)")
        else:
            print("WARNING: No KernelMapCache provided! Cache clearing disabled.")
        print("="*80)

    if allow_gs_tuning:
        tune_gs(
            model=model,
            data_loader=data_loader_fn,
            kernel_map_cache=kernel_map_cache,
            n_samples=n_samples_for_gs
        )

    if verbose:
        print("Tuned GS metrics")

    name_to_group = {}
    group_to_name = defaultdict(list)
    handler_collection = []

    def grouping_hook(module, inputs, outputs, name):

        input_sparse_tensor = inputs[0]

        # Get input stride (spatial resolution indicator)
        input_stride = input_sparse_tensor._stride

        # Adjust for transposed convolutions
        if module.transposed:
            effective_stride = tuple(
                input_stride[k] // module.stride[k] for k in range(len(input_stride))
            )
        else:
            effective_stride = input_stride

        # CREATE GROUP ID (exactly like TorchSparse)
        group_id = (
            effective_stride,
            module.kernel_size,
            module.stride,
        )

        # ASSIGN LAYER TO GROUP
        name_to_group[name] = group_id
        group_to_name[group_id].append(name)

    if verbose:
        print("\nStage 0: Dumping model structure (grouping layers)...")

    # Register hooks on all SparseConv layers
    for name, module in model.named_modules():
        if isinstance(module, SparseConv):
            handler = module.register_forward_hook(
                functools.partial(grouping_hook, name=name)
            )
            handler_collection.append(handler)

    # Run ONE forward pass to trigger hooks and populate groups
    for i, sample_input in enumerate(
        tqdm(data_loader_fn, desc="Dump the model structure", leave=False, total=1)
    ):
        with torch.no_grad():
            # Generate dumps - this triggers all hooks!
            name_to_group = {}
            group_to_name = defaultdict(list)
            _ = model(sample_input)

            # Detach the hooks
            for handler in handler_collection:
                handler.remove()

        break  # Only need one pass for grouping

    if verbose:
        print(f"✓ Found {len(group_to_name)} groups from {len(name_to_group)} layers")
        if len(name_to_group) > 0:

            # Print each group and its layers
            print("\n  Group details:")
            for group_idx, (group_id, layer_names) in enumerate(group_to_name.items(), 1):
                print(f"    Group {group_idx}: {group_id}")
                print(f"      Layers ({len(layer_names)}): {layer_names}")
        print()

    # ========================================================================
    # STAGE 1 & 2: TEST CONFIGURATIONS FOR EACH GROUP
    # ========================================================================
    if verbose:
        print("\nStage 1 & 2: Testing configurations for each group...")
        print("Note: KernelMapCache is reset before each timing measurement")

    best_configs = tune_dataflows(
        model=model,
        data_loader=data_loader_fn,
        group_to_name=group_to_name,
        config_options=config_options,
        kernel_map_cache=kernel_map_cache,
        n_samples=n_samples,
        verbose=verbose
    )

    # ========================================================================
    # STAGE 3: APPLY BEST CONFIGURATIONS
    # ========================================================================

    if verbose:
        print("\nStage 3: Applying optimized configurations...")

    apply_tuned_configs(model, name_to_group, best_configs, None)  # ✅ Pass None for layer_configs

    # ========================================================================
    # STAGE 4: SAVE RESULTS
    # ========================================================================

    save_tuning_results(name_to_group, best_configs, save_path, model)  # ✅ This will save per-layer

    if verbose:
        print(f"✓ Tuning results saved to: {save_path}")
        print("\n" + "="*80)
        print("TUNING COMPLETE!")
        print("="*80)
        print("\nConfiguration Summary:")

        config_summary = defaultdict(list)
        for group_id, config in best_configs.items():
            config_key = f"map_dataflow={config.map_dataflow}, force_os={config.force_os}"
            config_summary[config_key].append((group_id, group_to_name[group_id]))

        for config_str, groups in config_summary.items():
            total_layers = sum(len(layers) for _, layers in groups)
            print(f"\n  {config_str}")
            print(f"    Applied to {len(groups)} group(s), {total_layers} layer(s)")
            for group_id, layer_names in groups:#[:3]:
                print(f"      Group {group_id}: {layer_names}")
            #if len(groups) > 3:
            #    print(f"      ... and {len(groups)-3} more group(s)")

    group_to_map_dataflow = {
        group_id: config.map_dataflow
        for group_id, config in best_configs.items()
    }
    if return_dataflows:
        return model, group_to_map_dataflow

    return model