#pragma once

#include "spira/cuda/context.cuh"

namespace spira::cuda::kernels {

template <typename T>
__global__ void FillRange(std::size_t n, T *targets) {
  auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  auto gsz = SPIRA_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < n; i += gsz) {
    targets[i] = i;
  }
}

}  // namespace spira::cuda::kernels
