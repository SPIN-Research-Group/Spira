import argparse
import torch
import numpy as np
import random
import spira
from spira.nn import KernelMapCache as Spira_KMC
from spira.nn.functional import unflatten
from models.convert_weights import layout_spira_weights, layout_torchsparse_weights
from torchsparse.nn.functional.conv.conv_config import get_default_conv_config, set_global_conv_config

conv_config = get_default_conv_config()

conv_config.downsample_mode = "minkowski" # We follow the same downsampling definition as MinkowskiEngine
conv_config.kmap_mode = "hashmap"         # hashmap-on-the-fly mode cannot support minkowski downsample mode 

set_global_conv_config(conv_config)

LAYER_CONFIGS = [
    dict(cin=64, cout=64, kernel_size=3, stride=1),     #Submanifold
    dict(cin=64, cout=64, kernel_size=3, stride=2),     #Downsampling
    dict(cin=64, cout=64, kernel_size=5, stride=1),     #Submanifold
    dict(cin=64, cout=64, kernel_size=5, stride=2),     #Downsampling
]

SPIRA_DATAFLOWS = [                                   #These are the dataflows that Spira's tuner selects from
    0,    #Output-Stationary(t=L1NormMax+1)
    1,    #Weight-Stationary(t=0) (Single-Kernel)
    2,    #Weight-Stationary(t=0) (Gather-Scatter)
    3,    #Hybrid-Stationary(t=1) (Single-Kernel)
    4,    #Hybrid-Stationary(t=1) (Gather-Scatter)
    5,    #Hybrid-Stationary(t=2) (Single-Kernel)  
    6,    #Hybrid-Stationary(t=2) (Gather-Scatter)
    7,    #Hybrid-Stationary(t=3) (Single-Kernel)
    8     #Hybrid-Stationary(t=3) (Gather-Scatter)
]



def prepare_tensor(coords: torch.Tensor, feats: torch.Tensor,  lib: str):
    if lib == "spira":
        from spira import SparseTensor

        return SparseTensor(coordinates  = coords, features = feats).order()
    elif lib == "minuet":
        from minuet import SparseTensor
        from minuet.nn import functional as F
        index = F.build_sorted_index(coords)
        return SparseTensor(coordinates = coords[index].contiguous(), features = feats[index].contiguous())
    elif lib == "torchsparse":
        from torchsparse import SparseTensor

        coords = coords[:, [0, 2, 1]]  #TorchSparse operates with xyz -> xzy
        batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device="cuda")
        coords = torch.cat([batch, coords], dim=1) # add batch index
        coords = coords.contiguous()
        return SparseTensor(coords = coords, feats = feats)
    else:
        raise NotImplementedError(lib)


def prepare_layer(lib: str, cin: int, cout: int, kernel_size: int, stride: int):
    if lib == "spira":
        from spira.nn import SparseConv
        layer = SparseConv(in_channels=cin, out_channels=cout, kernel_size=kernel_size, stride=stride)
        src_cache = Spira_KMC()
        spira.set_kernel_map_cache(layer, src_cache)
        src_cache.reset()
        return layer
    elif lib == "minuet":
        from minuet.nn import SparseConv
        import minuet
        from minuet.nn import KernelMapCache as Minuet_KMC
        layer = SparseConv(ndim=3, in_channels=cin, out_channels=cout, kernel_size=kernel_size, stride=stride)
        std_cache = Minuet_KMC(ndim=3, dtype=torch.int32, device='cuda')
        std_cache.reset()
        minuet.set_kernel_map_cache(layer, std_cache)
        return layer
    elif lib == "torchsparse":
        from torchsparse.nn import Conv3d
        layer = Conv3d(in_channels=cin, out_channels=cout, kernel_size=kernel_size, stride=stride)
        return layer
    else:
        raise NotImplementedError(lib)

def main(
    seed: int,
    baseline: str,
    scene: str,
    fp32: bool,
):
    if fp32:
        dtype = torch.float32
    else:
        dtype = torch.float16

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tensor_data_np = np.load(scene)
    size = tensor_data_np.shape[0]
    np.random.shuffle(tensor_data_np)
    coords = torch.tensor(tensor_data_np, dtype=torch.int32, device='cuda')
    min_vals = coords.min(0).values
    coords = coords - min_vals

    feats = torch.rand((size, 64), device='cuda', dtype=dtype)

    src_tensor =  prepare_tensor(coords, feats, lib="spira")

    std_tensor = prepare_tensor(coords, feats, lib=baseline)

    for config in LAYER_CONFIGS:
        print(f"\n=== Testing Config: cin={config['cin']}, cout={config['cout']}, "
            f"kernel_size={config['kernel_size']}, stride={config['stride']} ===\n")

        src_layer = prepare_layer(lib="spira", **config).cuda().eval().to(dtype)

        base_layer = prepare_layer(lib=baseline, **config).cuda().eval().to(dtype) 

        if baseline == "minuet":
            std_layer = base_layer

        if baseline == "torchsparse":
            std_layer = layout_torchsparse_weights(base_layer)

        with torch.no_grad():        
            std_output = std_layer(std_tensor)
            
            std_feats = std_output.F

            std_coords = None

            if baseline == "minuet":
                std_coords = std_output.C
            else:   
                std_coords = std_output.C[:, 1:]
                std_coords = std_coords[:, [0, 2, 1]]
                # Sort from least significant to most significant column
                indices = torch.argsort(std_coords[:, 2], stable=True)  # sort by z
                std_coords = std_coords[indices]
                std_feats = std_feats[indices]
                indices = torch.argsort(std_coords[:, 1], stable=True)  # then by y
                std_coords = std_coords[indices]
                std_feats = std_feats[indices]
                indices = torch.argsort(std_coords[:, 0], stable=True)  # then by x
                std_coords = std_coords[indices]
                std_feats = std_feats[indices]
                stride = torch.tensor(std_output.stride, device='cuda')
                std_coords = (std_coords * stride).contiguous()
            
            for dataflow in SPIRA_DATAFLOWS:

                if(config['stride'] > 1 and dataflow > 2):
                    continue

                print(f"Testing Spira Dataflow: {dataflow}")

                for key, value in base_layer.named_parameters():
                    dict(src_layer.named_parameters())[key].data[:] = value.data[:]

                src_layer._kernel_map_cache.reset()

                src_layer._tunable_config['map_dataflow'] = dataflow

                layout_spira_weights(src_layer)

                src_output = src_layer(src_tensor)

                src_coords, src_feats = unflatten(src_output._coordinates), src_output._features

                coords_match = (src_coords == std_coords).all().item()

                if coords_match:
                    status = "\033[92m✓\033[0m"   # green
                else:
                    status = "\033[91m✗\033[0m"   # red

                print(f"Coordinates match? {status}")

                diff = torch.abs(src_feats - std_feats)

                threshold = 0.001 * torch.max(torch.abs(std_feats))

                # Boolean mask: which features are “close enough”
                mask = diff <= threshold

                # Percent match
                percent_match = mask.float().mean().item() * 100

                if percent_match >= 96.0:
                    status = "\033[92m✓\033[0m"  # green check
                else:
                    status = "\033[91m✗\033[0m"  # red X

                print(f"Percentage of Features with less than 0.1% difference: {percent_match:.2f}% {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility (default: 2026)")
    parser.add_argument("--baseline", type=str, default='minuet', choices=['minuet', 'torchsparse'])
    parser.add_argument("--scene", type=str, required=True, help="Path to a single scene file (.npy)")
    parser.add_argument("--fp32", action="store_true", help="FP32 precision for increased accuracy")

    args = parser.parse_args()
    main(**vars(args))