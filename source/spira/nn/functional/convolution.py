__all__ = [
    'cuda_time_gather', 'cuda_time_scatter', 'sparse_convolution_forward_ws', 'sparse_convolution_forward_ws_merged', 'sparse_convolution_forward_os',
    'sparse_convolution_forward_hs', 'sparse_convolution_forward_hs_merged', 'cuda_time_gemm', 'set_gemm_parallel_level'
]

from typing import Optional

import torch

from spira.nn.functional import _C

GEMM_PARALLEL_LEVEL = 4

def set_gemm_parallel_level(level: int):
  
  global GEMM_PARALLEL_LEVEL
  GEMM_PARALLEL_LEVEL = level

def cuda_time_gather(weights: torch.Tensor,
                     source_masks: torch.Tensor,
                     target_masks: torch.Tensor,
                     kernel_map_sizes: torch.Tensor,
                     tile_size: int,
                     is_hybrid: bool = False,
                     threshold: float = 0) -> float:
  
  return _C.cuda_time_gather(tile_size, threshold, is_hybrid,
                             weights, source_masks, target_masks,
                             kernel_map_sizes)


def cuda_time_scatter(weights: torch.Tensor,
                      source_masks: torch.Tensor,
                      target_masks: torch.Tensor,
                      kernel_map_sizes: torch.Tensor,
                      tile_size: int,
                      is_hybrid: bool = False,
                      threshold: float = 0):

  return _C.cuda_time_scatter(tile_size, threshold, is_hybrid,
                              weights, source_masks, target_masks,
                              kernel_map_sizes)


def cuda_time_gemm(weights: torch.Tensor,
                   source_masks: torch.Tensor,
                   target_masks: torch.Tensor,
                   kernel_map_sizes: torch.Tensor,
                   parallel: Optional[int] = None,
                   is_hybrid: bool = False,
                   threshold: float = 0):
  
  if parallel is None:
    parallel = GEMM_PARALLEL_LEVEL
  return _C.cuda_time_gemm(parallel, threshold, is_hybrid, weights,
                           source_masks, target_masks, kernel_map_sizes)


def sparse_convolution_forward_ws(num_sources: int,
                                  num_targets:int, 
                                  sources: torch.Tensor,
                                  weights: torch.Tensor,
                                  source_masks: torch.Tensor,
                                  target_masks: torch.Tensor,
                                  kernel_map_sizes: torch.Tensor,
                                  gather_tile_size: int,
                                  scatter_tile_size: int,
                                  parallel: Optional[int] = None,
                                  threshold: Optional[float] = 0) -> torch.Tensor:
  if sources.is_cuda:
    if parallel is None:
      parallel = GEMM_PARALLEL_LEVEL
    sources = sources.contiguous()
    weights = weights.contiguous()
    source_masks = source_masks.contiguous()
    target_masks = target_masks.contiguous()
    kernel_map_sizes = kernel_map_sizes.contiguous()
    return _C.cuda_sparse_convolution_forward_ws(
        num_sources, num_targets, gather_tile_size, scatter_tile_size, parallel,
        threshold, sources, weights, source_masks, target_masks,
        kernel_map_sizes)
  raise NotImplementedError


def sparse_convolution_forward_ws_merged(num_sources: int,
                                         num_targets: int,
                                         sources: torch.Tensor,
                                         weights: torch.Tensor,
                                         naddrs: torch.Tensor,  
                                         qnaddrs: torch.Tensor,   
                                         nbmaps: torch.Tensor,
                                         mapsize: int,
                                         qmapsize: int,
                                         transpose: Optional[bool] = False) -> torch.Tensor:
  if sources.is_cuda:
    sources = sources.contiguous()
    weights = weights.contiguous()
    naddrs = naddrs.contiguous()
    qnaddrs = qnaddrs.contiguous()
    nbmaps = nbmaps.contiguous()
    return _C.cuda_sparse_convolution_forward_ws_merged(
        num_sources, num_targets, sources, weights, naddrs, qnaddrs,
        nbmaps, mapsize, qmapsize, transpose)
  raise NotImplementedError


    


def sparse_convolution_forward_hs(num_sources: int,
                                  num_targets:int, 
                                  sources: torch.Tensor,
                                  weights: torch.Tensor,
                                  source_masks: torch.Tensor,
                                  target_masks: torch.Tensor,
                                  kernel_map_sizes: torch.Tensor,
                                  out_in_map: torch.Tensor,
                                  gather_tile_size: int,
                                  scatter_tile_size: int,
                                  parallel: Optional[int] = None,
                                  threshold: Optional[float] = 0) -> torch.Tensor:
  if sources.is_cuda:
    if parallel is None:
      parallel = GEMM_PARALLEL_LEVEL
    sources = sources.contiguous()
    weights = weights.contiguous()
    source_masks = source_masks.contiguous()
    target_masks = target_masks.contiguous()
    kernel_map_sizes = kernel_map_sizes.contiguous()
    if out_in_map is not None:
        out_in_map = out_in_map.contiguous()
    return _C.cuda_sparse_convolution_forward_hs(
        num_sources, num_targets, gather_tile_size, scatter_tile_size, parallel,
        threshold, sources, weights, source_masks, target_masks,
        kernel_map_sizes, out_in_map)
  raise NotImplementedError


def sparse_convolution_forward_hs_merged(num_sources: int,
                                         num_targets: int,
                                         sources: torch.Tensor,
                                         weights: torch.Tensor,
                                         naddrs: torch.Tensor,  
                                         qnaddrs: torch.Tensor,   
                                         nbmaps: torch.Tensor,
                                         out_in_map: torch.Tensor,
                                         mapsize: int,
                                         qmapsize: int,
                                         transpose: Optional[bool] = False) -> torch.Tensor:
  if sources.is_cuda:
    sources = sources.contiguous()
    weights = weights.contiguous()
    naddrs = naddrs.contiguous()
    qnaddrs = qnaddrs.contiguous()
    nbmaps = nbmaps.contiguous()
    if out_in_map is not None:
        out_in_map = out_in_map.contiguous()
    return _C.cuda_sparse_convolution_forward_hs_merged(
        num_sources, num_targets, sources, weights, naddrs, qnaddrs,
        nbmaps, out_in_map, mapsize, qmapsize, transpose)
  raise NotImplementedError

def sparse_convolution_forward_os(num_sources: int,
                                  num_targets:int, 
                                  out_in_map: torch.Tensor,
                                  sources: torch.Tensor,
                                  weights: torch.Tensor) -> torch.Tensor:
  if sources.is_cuda:
    sources = sources.contiguous()
    weights = weights.contiguous()
    out_in_map = out_in_map.contiguous()
    return _C.cuda_sparse_convolution_forward_os(
        num_sources, num_targets, out_in_map, sources, weights)
  raise NotImplementedError



