#pragma once

#include <cstddef>

#include "spira/cuda/device/data_movement.cuh"

namespace spira::cuda::kernels {

template <typename IT>
__global__ void GenerateMasksFromKernelMap(
    std::size_t num_entries,                                      //
    std::size_t num_sources,                                      //
    std::size_t num_targets,
    std::size_t aligned_num_targets,                                 
    const IT *__restrict__ kernel_map_sizes,                      //
    const std::int64_t *__restrict__ kernel_map_nonzero_indices,  //
    IT *__restrict__ source_masks,                                //
    IT *__restrict__ target_masks) {
  auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  auto gsz = SPIRA_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_entries; i += gsz) {
    auto entry = kernel_map_nonzero_indices[i];  // (value, index)
    auto s = entry % num_sources;
    auto tmp = entry / num_sources;
    auto t = tmp % aligned_num_targets;
    auto o = tmp / aligned_num_targets;
    auto index = i - kernel_map_sizes[o];
    source_masks[o * num_sources + s] = index;
    target_masks[o * num_targets + t] = index;
  }
}


template <typename IT>
__global__ void GenerateMasksFromKernelMap_Half(
    std::size_t num_entries,                                      
    std::size_t num_sources,                                      
    std::size_t num_targets,
    std::size_t num_offsets,
    std::size_t aligned_num_targets,                                 
    const IT *__restrict__ kernel_map_sizes,                      
    const std::int64_t *__restrict__ kernel_map_nonzero_indices,  
    IT *__restrict__ source_masks,                                
    IT *__restrict__ target_masks) {
  
  auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  auto gsz = SPIRA_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_entries; i += gsz) {
    auto entry = kernel_map_nonzero_indices[i];  // (value, index)
    auto s = entry % num_sources;
    auto tmp = entry / num_sources;
    auto t = tmp % aligned_num_targets;
    auto o = tmp / aligned_num_targets;
    auto index = i - kernel_map_sizes[o];
    source_masks[o * num_sources + s] = index;
    target_masks[o * num_targets + t] = index;
    if(2 * o + 1 != num_offsets){
        source_masks[(num_offsets - 1 - o) * num_sources + t] = index;
        target_masks[(num_offsets - 1 - o) * num_targets + s] = index;
    }
  }
}

}  // namespace spira::cuda::kernels