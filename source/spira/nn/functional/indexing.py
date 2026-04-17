__all__ = [
    'flat_sort',
    'unflatten',
    'downsample',  
    'build_kmap',
    'convert_transposed_os',
    'build_kmap_streamed',
    'streamed_output_generate',
    'build_kmap_no_small_z'  
]

from typing import Optional, Tuple, List

from spira.utils.typing import ScalarOrTuple

from spira.utils.helpers import as_tuple

import torch

from spira.nn.functional import _C

def flat_sort(coordinates: torch.Tensor, double: bool = False):
  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    flat_coords, index = _C.cuda_flatten_sort(coordinates, double)
    return flat_coords, index.to(torch.int64)
  raise NotImplementedError

def unflatten(coordinates: torch.Tensor):
  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    return _C.cuda_unflatten(coordinates)
  raise NotImplementedError

def streamed_output_generate(coordinates: torch.Tensor, target_strides: List[Tuple[int, int, int]]):
  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    return _C.cuda_streamed_output_generate(coordinates, target_strides)
  raise NotImplementedError

def build_kmap_streamed(
    voxels: List[torch.Tensor],
    kernel_sizes: List[ScalarOrTuple[int]],
    source_strides: List[ScalarOrTuple[int]],
    dataflow: List[int]):
  
  voxels = [v.contiguous() for v in voxels]
  kernel_sizes = [as_tuple(kernel_size, size=3, name="kernel_size") for kernel_size in kernel_sizes]
  source_strides = [as_tuple(source_stride, size=3, name="kernel_stride") for source_stride in source_strides]

  if not (len(kernel_sizes) == len(source_strides) == len(dataflow)):
    raise ValueError(
        f"All argument lists must have the same length in build_kmap_streamed"
    )

  return _C.cuda_binary_search_streamed(voxels, kernel_sizes, source_strides, dataflow)

def downsample(coordinates: torch.Tensor, target_stride: Tuple[int, int, int]):
  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    return _C.cuda_downsample(coordinates, *target_stride)
  raise NotImplementedError

def build_kmap(
    sources: torch.Tensor,
    targets: torch.Tensor,
    kernel_size: ScalarOrTuple[int],
    source_stride: ScalarOrTuple[int],
    dataflow: int = 0):

  sources = sources.contiguous()
  targets = targets.contiguous()

  kernel_size = as_tuple(kernel_size, size=3, name="kernel_size")
  source_stride = as_tuple(source_stride, size=3, name="kernel_stride")

  return _C.cuda_binary_search(sources, targets, kernel_size, source_stride, dataflow)

def convert_transposed_os(
    out_in_map: torch.Tensor,
    input_size: int
  ):

  out_in_map = out_in_map.contiguous()

  return _C.cuda_convert_transposed_os(out_in_map, input_size)

#############################################################################################################################


###################### no z delta binary search ######################

def build_kmap_no_small_z(
    sources: torch.Tensor,
    targets: torch.Tensor,
    kernel_size: ScalarOrTuple[int],
    source_stride: ScalarOrTuple[int]):

  sources = sources.contiguous()
  targets = targets.contiguous()

  kernel_size = as_tuple(kernel_size, size=3, name="kernel_size")
  source_stride = as_tuple(source_stride, size=3, name="kernel_stride")

  return _C.cuda_binary_search_no_small_z(sources, targets, kernel_size, source_stride)
