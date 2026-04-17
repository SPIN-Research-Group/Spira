#pragma once

#include "spira/cuda/context.cuh"

namespace spira::cuda::kernels {

__global__ void Convert_out_in_map_kernel(const int32_t *out_in_map, int32_t *out_in_map_t, std::size_t aligned_num_targets, std::size_t num_offsets){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= aligned_num_targets * num_offsets) return;
  int input_idx = out_in_map[idx];
  if(input_idx < 0) return;
  out_in_map_t[idx % num_offsets + input_idx * num_offsets] = idx / num_offsets;
}

}  // namespace spira::cuda::kernels




