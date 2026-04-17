#pragma once

#include "spira/cuda/context.cuh"
#include "spira/compress_layout.h"

namespace spira::cuda::kernels {

template <typename CT>
__global__ void Compress(std::size_t n,                                  
                                   const int32_t *__restrict__ sources,  
                                   CT *__restrict__ targets) {

  auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  auto gsz = SPIRA_N_GLOBAL_THREADS(x);

  for (UIter i = gid; i < n; i += gsz) {
    auto source = sources + i * 3;
    CT ux = static_cast<CT>(source[0]);  
    CT uy = static_cast<CT>(source[1]);  
    CT uz = static_cast<CT>(source[2]);   

    targets[i] = (ux << Layout<CT>::X_SHIFT) | (uy << Layout<CT>::Y_SHIFT) | uz;
  }
}


template <typename CT>
__global__ void UnCompress(std::size_t n,                                  
                                   const CT *__restrict__ sources,  
                                   int32_t *__restrict__ targets) {

    auto gid = SPIRA_GLOBAL_THREAD_ID(x);
    auto gsz = SPIRA_N_GLOBAL_THREADS(x);

    for (UIter i = gid; i < n; i += gsz) {
        CT key = sources[i];
        CT ux = (key >> Layout<CT>::X_SHIFT) & Layout<CT>::X_ONES; 
        CT uy = (key >> Layout<CT>::Y_SHIFT) & Layout<CT>::Y_ONES;   
        CT uz = key & Layout<CT>::Z_ONES;           

        auto target = targets + i * 3;
        target[0] = static_cast<int32_t>(ux);
        target[1] = static_cast<int32_t>(uy);
        target[2] = static_cast<int32_t>(uz);
    }
}


}  // namespace spira::cuda::kernels
