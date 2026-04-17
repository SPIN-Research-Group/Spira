__all__ = ['SparseConv', 'KernelMapCache']

import operator
import contextlib
from typing import Optional, Union, Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import Module

from spira.nn import functional as F
from spira.utils.helpers import as_tuple
from spira.utils.typing import ScalarOrTuple

from spira import SparseTensor

class KernelMapCache(object):

  def __init__(self):
    self._kernel_maps_cache = dict()
    self._coordinates_cache = dict()

  def reset(self):
    self._coordinates_cache.clear()
    self._kernel_maps_cache.clear()

  def get_target_coordinates(self,
                             source_coordinates: torch.Tensor,
                             source_stride: ScalarOrTuple[int],
                             target_stride: ScalarOrTuple[int],
                             transposed: bool = False):

    source_stride = as_tuple(source_stride, size=3, name="source_stride")

    if source_stride not in self._coordinates_cache:
      self._coordinates_cache[source_stride] = source_coordinates

    if target_stride in self._coordinates_cache:
      return self._coordinates_cache[target_stride]

    if transposed:
      raise RuntimeError(
          "Layer must transpose=True must have its corresponding "
          "non-transposed layer before its execution")

    target_coordinates = F.downsample(source_coordinates, target_stride)
    self._coordinates_cache[target_stride] = target_coordinates

    return target_coordinates

  def get_kernel_map(self,
                     source_coordinates: torch.Tensor,
                     source_stride: ScalarOrTuple[int],
                     target_coordinates: torch.Tensor,
                     target_stride: ScalarOrTuple[int],
                     kernel_size: ScalarOrTuple[int],
                     map_dataflow: int,
                     force_os: bool = False,
                     transposed: bool = False):
    

    source_stride = as_tuple(source_stride, size=3, name="source_stride")
    target_stride = as_tuple(target_stride, size=3, name="target_stride")

    kernel_size = as_tuple(kernel_size, size=3, name="kernel_size")
    
    kernel_map_key = list()
    if transposed:
      kernel_map_key.append(target_stride)
      kernel_map_key.append(source_stride)
    else:
      kernel_map_key.append(source_stride)
      kernel_map_key.append(target_stride)
    kernel_map_key.append(kernel_size)
    kernel_map_key = tuple(kernel_map_key)

    if kernel_map_key in self._kernel_maps_cache:

        cache_entry = self._kernel_maps_cache[kernel_map_key]  # [flipped_flag, kernel_map]
        if force_os:
          if not cache_entry[0]:  # is unflipped
                cache_entry[3] = cache_entry[3].T.contiguous()
                cache_entry[0] = True

        if (map_dataflow == 0 or force_os) and transposed:
          out_in_map_t = F.convert_transposed_os(cache_entry[3], cache_entry[1])
          new_entry = list(cache_entry[1:]) 
          new_entry[2] = out_in_map_t
          return new_entry
        
        return cache_entry[1:]  
    
    if transposed:
      raise RuntimeError(
          "Layer must transpose=True must have its corresponding "
          "non-transposed layer before its execution")

    kernel_map = F.build_kmap(
                source_coordinates,
                target_coordinates,
                kernel_size,
                source_stride,
                map_dataflow
            )

    cache_entry = [False, *kernel_map]

    if force_os:
        cache_entry[3] = cache_entry[3].T.contiguous()
        cache_entry[0] = True

    self._kernel_maps_cache[kernel_map_key] = cache_entry

    return cache_entry[1:]  

  def add_all_kernel_maps(
      self,
      source_coordinates: torch.Tensor,
      kernel_sizes: List[ScalarOrTuple[int]],
      source_strides: List[ScalarOrTuple[int]], 
      target_strides: List[ScalarOrTuple[int]], 
      init_stride: Union[int, Tuple[int, ...]],
      dataflow: List[int]):

      output = F.streamed_output_generate(source_coordinates, target_strides)
      
      voxels: List[torch.Tensor] = [source_coordinates] + output  

      s_stride = as_tuple(init_stride, size=3)
      self._coordinates_cache[s_stride] = voxels[0]

      for i, t_stride in enumerate(target_strides):
        t_stride = as_tuple(t_stride, size=3)
        self._coordinates_cache[t_stride] = voxels[i + 1]

      kmaps = F.build_kmap_streamed(voxels, kernel_sizes, source_strides[:-1], dataflow)
      
      s_stride = as_tuple(init_stride, size=3)

      for i in range(len(kernel_sizes)):
          key = list()           
          t_stride = as_tuple(source_strides[i+1], size=3)
          ksize = kernel_sizes[i]
          key.append(kernel_sizes[i])
          key = (s_stride, t_stride, as_tuple(ksize, size=3))
          key = tuple(key)
          kernel_map = kmaps[i]
          self._kernel_maps_cache[key] = [False, *kernel_map]
          s_stride = t_stride
      
      return self

class SparseConv(Module):

  __AVAILABLE_THRESHOLD__ = [-1, *list(np.linspace(0, 2, num=50)), None]
  __NUM_TUNING_ROUNDS__ = 5

  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               bias: bool = False,
               transposed: bool = False):
    
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = as_tuple(kernel_size, size=3, name="kernel_size")
    self.stride = as_tuple(stride, size=3, name="stride")
    self.transposed = transposed

    kernel_volume = np.prod(self.kernel_size).item()
    self.kernel = torch.nn.Parameter(torch.empty(kernel_volume, in_channels, out_channels))

    if bias:
      self.bias = torch.nn.Parameter(torch.empty(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

    self._kernel_map_cache: Optional[KernelMapCache] = None
    
    self._tunable_config = {
        'threshold': -1,
        'gather_tile_size': min(in_channels, 16),
        'scatter_tile_size': min(out_channels, 16),
        'map_dataflow': 2,
        'force_os': False
    }

    self._tuning_data = None

    

  @property
  def tunable_config(self) -> Dict[str, Any]:
    return self._tunable_config

  def reset_parameters(self):
    std = (self.out_channels if self.transposed else self.in_channels)
    std = 1. / np.sqrt(std * len(self.kernel))
    self.kernel.data.uniform_(-std, std)
    if self.bias is not None:
      self.bias.data.uniform_(-std, std)

  def set_kernel_map_cache(self, cache: KernelMapCache):
    self._kernel_map_cache = cache

  @property
  def is_trivial(self):
    return all(x == 1 for x in self.stride) and \
      all(x == 1 for x in self.kernel_size)



  @contextlib.contextmanager
  def tune_gs(self, free_buffers: bool = True):

    try:
      self._tuning_data = list()
      if free_buffers:
        F.cuda_free_buffers()

      yield

      if free_buffers:
        F.cuda_free_buffers()

      if self.is_trivial:
        return

      optimal_time = None
      for threshold in self.__AVAILABLE_THRESHOLD__:
        try:
          timings = []
          for _ in range(self.__NUM_TUNING_ROUNDS__):
            for source_masks, target_masks, kernel_map_sizes in self._tuning_data:
              timings.append(
                  F.cuda_time_gemm( 
                      threshold=threshold,
                      weights=self.kernel,
                      source_masks=source_masks,
                      target_masks=target_masks,
                      kernel_map_sizes=kernel_map_sizes))
          timings = np.average(timings)
          if optimal_time is None or optimal_time > timings:
            self._tunable_config['threshold'] = threshold
            optimal_time = timings
        except RuntimeError:
          F.cuda_reset_error()
          print(f"Threshold {threshold} overflows memory exiting tuning")
          print(f"optimal={self._tunable_config['threshold']}")
          break

      if free_buffers:
        F.cuda_free_buffers()

      optimal_time = None
      for tile_size in range(1, self.in_channels + 1):
        if self.in_channels % tile_size != 0 or (tile_size % 4 != 0):
          continue

        timings = []
        for _ in range(self.__NUM_TUNING_ROUNDS__):
          for source_masks, target_masks, kernel_map_sizes in self._tuning_data:
            timings.append(
                F.cuda_time_gather(    
                    threshold=self._tunable_config['threshold'],
                    tile_size=tile_size,
                    weights=self.kernel,
                    source_masks=source_masks,
                    target_masks=target_masks,
                    kernel_map_sizes=kernel_map_sizes))
        timings = np.average(timings)
        if optimal_time is None or optimal_time > timings:
          self._tunable_config['gather_tile_size'] = tile_size
          optimal_time = timings

      if free_buffers:
        F.cuda_free_buffers()

      optimal_time = None
      for tile_size in range(1, self.out_channels + 1):
        if self.out_channels % tile_size != 0 or (tile_size % 4 != 0):
          continue

        timings = []
        for _ in range(self.__NUM_TUNING_ROUNDS__):
          for source_masks, target_masks, kernel_map_sizes in self._tuning_data:
            timings.append(
                F.cuda_time_scatter(    
                    threshold=self._tunable_config['threshold'],
                    tile_size=tile_size,
                    weights=self.kernel,
                    source_masks=source_masks,
                    target_masks=target_masks,
                    kernel_map_sizes=kernel_map_sizes))
        timings = np.average(timings)
        if optimal_time is None or optimal_time > timings:
          self._tunable_config['scatter_tile_size'] = tile_size
          optimal_time = timings

      if free_buffers:
        F.cuda_free_buffers()
    finally:
      self._tuning_data = None


  def forward(self, inputs: SparseTensor):
    source_features = inputs._features 
    if self.is_trivial:
      kernel = self.kernel.view(self.in_channels, self.out_channels)
      target_features = torch.mm(source_features, kernel)
      target_coordinates = inputs._coordinates
      target_stride = inputs._stride
    else:
      if self._kernel_map_cache is None:
        raise ValueError("Kernel map cache must be specified")

      op = operator.floordiv if self.transposed else operator.mul
      target_stride = tuple(op(a, b) for a, b in zip(inputs._stride, self.stride))

      target_coordinates = self._kernel_map_cache.get_target_coordinates(inputs._coordinates, inputs._stride, target_stride, self.transposed)

      map_dataflow = self._tunable_config['map_dataflow']
      force_os = self._tunable_config['force_os']

      map = self._kernel_map_cache.get_kernel_map(inputs._coordinates, inputs._stride, target_coordinates, target_stride, 
                                                         self.kernel_size, 
                                                         map_dataflow, force_os, self.transposed)

      if map_dataflow == 0 or force_os:
        if self.transposed:
          target_features = F.sparse_convolution_forward_os(map[1], map[0], map[2], source_features, self.kernel)
        else:
          target_features = F.sparse_convolution_forward_os(map[0], map[1], map[2], source_features, self.kernel)
      
      
      
      
      elif map_dataflow == 1:
        if self.transposed:
          target_features = F.sparse_convolution_forward_ws_merged(map[1], map[0], source_features, self.kernel, 
                                                            map[5], map[6], map[3], map[7], map[8], self.transposed)
        else:
          target_features = F.sparse_convolution_forward_ws_merged(map[0], map[1], source_features, self.kernel, 
                                                map[5], map[6], map[3], map[7], map[8], self.transposed)
      
            
      elif map_dataflow == 2:

        if self.transposed:
            if self._tuning_data is not None:
              self._tuning_data.append((map[4], map[3], map[5]))

            target_features = F.sparse_convolution_forward_ws(map[1], map[0], source_features, self.kernel, 
                                                          map[4], map[3], map[5], 
                                                          self._tunable_config['gather_tile_size'], self._tunable_config['scatter_tile_size'],
                                                          None, self._tunable_config['threshold'])
        else:
            if self._tuning_data is not None:
              self._tuning_data.append((map[3], map[4], map[5]))

            target_features = F.sparse_convolution_forward_ws(map[0], map[1], source_features, self.kernel, 
                                                          map[3], map[4], map[5], 
                                                          self._tunable_config['gather_tile_size'], self._tunable_config['scatter_tile_size'],
                                                          None, self._tunable_config['threshold'])
      
      
      
      
      elif (map_dataflow % 2 == 1):
        if self.transposed:
          if map_dataflow == 3:  
              target_features = F.sparse_convolution_forward_hs_merged(map[1], map[0], source_features, self.kernel, 
                                                            map[5], map[6], map[3], None, map[7], map[8], self.transposed)
          else:
              target_features = F.sparse_convolution_forward_hs_merged(map[1], map[0], source_features, self.kernel, 
                                                            map[5], map[6], map[3], map[2], map[7], map[8], self.transposed)
        else:
          if map_dataflow == 3:  
              target_features = F.sparse_convolution_forward_hs_merged(map[0], map[1], source_features, self.kernel, 
                                                            map[5], map[6], map[3], None, map[7], map[8], self.transposed)
          else:
              target_features = F.sparse_convolution_forward_hs_merged(map[0], map[1], source_features, self.kernel, 
                                                            map[5], map[6], map[3], map[2], map[7], map[8], self.transposed)
      
      
      
      else:
        if self.transposed:
            if map_dataflow == 4:
                target_features = F.sparse_convolution_forward_hs(map[1], map[0], source_features, self.kernel, 
                                                          map[4], map[3], map[5], None,
                                                          self._tunable_config['gather_tile_size'], self._tunable_config['scatter_tile_size'],
                                                          None, self._tunable_config['threshold'])
            else:
                target_features = F.sparse_convolution_forward_hs(map[1], map[0], source_features, self.kernel, 
                                                          map[4], map[3], map[5], map[2],
                                                          self._tunable_config['gather_tile_size'], self._tunable_config['scatter_tile_size'],
                                                          None, self._tunable_config['threshold'])
        else:
            if map_dataflow == 4:
                target_features = F.sparse_convolution_forward_hs(map[0], map[1], source_features, self.kernel, 
                                                          map[3], map[4], map[5], None,
                                                          self._tunable_config['gather_tile_size'], self._tunable_config['scatter_tile_size'],
                                                          None, self._tunable_config['threshold'])
            else:
                target_features = F.sparse_convolution_forward_hs(map[0], map[1], source_features, self.kernel, 
                                                          map[3], map[4], map[5], map[2],
                                                          self._tunable_config['gather_tile_size'], self._tunable_config['scatter_tile_size'],
                                                          None, self._tunable_config['threshold'])

    if self.bias is not None:
      target_features += self.bias
    return SparseTensor(features=target_features,
                        coordinates=target_coordinates,
                        stride=target_stride)


