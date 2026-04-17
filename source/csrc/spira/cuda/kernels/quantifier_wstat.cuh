#pragma once

#include "spira/cuda/context.cuh"

namespace spira::cuda::kernels {


__global__ void exclusive_scan_for_kernel_quantified(
                const std::int32_t kv, 
                const std::int32_t *input, 
                const int q, 
                std::int32_t *output,
                std::int32_t *qoutput,
                std::int32_t *d_sizes
){

  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= kv){return;}

  if (id == 0) {
        output[0] = 0;
        qoutput[0] = 0;
        return;
  }

  int acc = 0;
  int qacc = 0;
#pragma unroll 
  for (int i = 0; i < id; i++){ 
    acc += input[i];
    qacc += (input[i] + q - 1) / q * q;
  }

  output[id] = acc;
  qoutput[id] = qacc;

  if (id == kv - 1) {
    d_sizes[0] = acc;
    d_sizes[1] = qacc;
  }
}



__global__ void exclusive_scan_for_kernel(
                const std::int32_t kv, 
                const std::int32_t *input, 
                std::int32_t *output
){

  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= kv){return;}

  if (id == 0) {
      output[0] = 0;
      return;
  }

  int acc = 0;
#pragma unroll 
  for (int i = 0; i < id; i++){ 
    acc += input[i];
  }

  output[id] = acc;
}



}  // namespace spira::cuda::kernels