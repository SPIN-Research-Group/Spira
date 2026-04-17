#pragma once

#include "spira/cuda/context.cuh"
#include "spira/cuda/device/binary_search.cuh"

namespace spira::cuda::kernels {

template <typename CT, uint32_t KZ>      
__global__ void BinarySearch_kz_wstationary(
    std::size_t num_sources,                      
    std::size_t num_targets,
    std::size_t aligned_num_targets,                                                      
    std::size_t num_offsets,  
    uint32_t sz,                
    const CT *__restrict__ sources,               
    const CT *__restrict__ targets,
    const CT *__restrict__ offsets,                             
    std::int32_t *__restrict__ kernel_map) {
  
  const auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  const auto gsz = SPIRA_N_GLOBAL_THREADS(x);

  for (UIter t = gid; t < aligned_num_targets * num_offsets; t += gsz) {
      auto num_target = t % aligned_num_targets;
      auto o = t / aligned_num_targets;

      if(num_target >= num_targets){
        #pragma unroll
        for(uint32_t iz = 0; iz < KZ; ++iz) {
            kernel_map[(KZ * o + iz) * aligned_num_targets + num_target] = -1;
        }
        return;
      }

      CT cached = targets[num_target] + offsets[o];

      auto index = device::BinarySearchMinimize<std::int32_t>(
                      0, num_sources,
                      [&](auto k) { return sources[k]; },
                      cached);

      kernel_map[(KZ * o) * aligned_num_targets + num_target] = (sources[index] == cached) ? index : -1;

      //linear search for kz

      #pragma unroll
      for(uint32_t iz = 1; iz < KZ; ++iz) {
          auto cond = (sources[index] <= cached);
          index += cond;
          cached += sz;

          kernel_map[(KZ * o + iz) * aligned_num_targets + num_target] = 
              (index < num_sources && sources[index] == cached) ? index : -1;
      }
    }
}

template <typename CT, uint32_t KZ>      
__global__ void BinarySearch_kz_hybrid(    
    std::size_t num_sources,                      
    std::size_t num_targets,
    std::size_t aligned_num_targets,
    std::size_t num_offsets,                                            
    uint32_t sz,          
    const CT *__restrict__ sources,               
    const CT *__restrict__ targets,
    const CT *__restrict__ offsets,                
    const std::uint32_t *__restrict__ indexes,              
    std::int32_t *__restrict__ kernel_map) {

  const auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  const auto gsz = SPIRA_N_GLOBAL_THREADS(x);

  for (UIter t = gid; t < aligned_num_targets * num_offsets; t += gsz) {
      auto num_target = t % aligned_num_targets;
      auto o = t / aligned_num_targets;

      // Load positions from indexes array
      uint32_t positions[KZ];
      #pragma unroll
      for(uint32_t iz = 0; iz < KZ; ++iz) {
          positions[iz] = indexes[KZ * o + iz];
      }

      if(num_target >= num_targets){
          #pragma unroll
          for(uint32_t iz = 0; iz < KZ; ++iz) {
              kernel_map[positions[iz] * aligned_num_targets + num_target] = -1;
          }
          return;
      }

      CT cached = targets[num_target] + offsets[o];

      auto index = device::BinarySearchMinimize<std::int32_t>(
                      0, num_sources,
                      [&](auto k) { return sources[k]; },
                      cached);

      kernel_map[positions[0] * aligned_num_targets + num_target] = (sources[index] == cached) ? index : -1;

      //linear search for kz

      #pragma unroll
      for(uint32_t iz = 1; iz < KZ; ++iz) {
          auto cond = (sources[index] <= cached);
          index += cond;
          cached += sz;
          kernel_map[positions[iz] * aligned_num_targets + num_target] = 
              (index < num_sources && sources[index] == cached) ? index : -1;
      }
    }
}


template <typename CT, uint32_t KZ>      
__global__ void BinarySearch_kz_ostationary(
    std::size_t num_sources,                      
    std::size_t num_targets,
    std::size_t aligned_num_targets,                                                      
    std::size_t num_offsets,  
    uint32_t sz,                 
    const CT *__restrict__ sources,               
    const CT *__restrict__ targets,
    const CT *__restrict__ offsets,                           
    std::int32_t *__restrict__ kernel_map) {
  
  const auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  const auto gsz = SPIRA_N_GLOBAL_THREADS(x);

  for (UIter t = gid; t < aligned_num_targets * num_offsets; t += gsz) {
      auto o = t % num_offsets;
      auto num_target = t / num_offsets;

      if(num_target >= num_targets){
          #pragma unroll
          for(uint32_t iz = 0; iz < KZ; ++iz) {
              kernel_map[KZ * (num_target * num_offsets) + o + iz * num_offsets] = -1;
          }
          return;
      }

      CT cached = targets[num_target] + offsets[o];

      auto index = device::BinarySearchMinimize<std::int32_t>(
                      0, num_sources,
                      [&](auto k) { return sources[k]; },
                      cached);

      kernel_map[KZ * (num_target * num_offsets) + o] = (sources[index] == cached) ? index : -1;

      //linear search for kz

      #pragma unroll
      for(uint32_t iz = 1; iz < KZ; ++iz) {
          auto cond = (sources[index] <= cached);
          index += cond;
          cached += sz;
          kernel_map[KZ * (num_target * num_offsets) + o + iz * num_offsets] = 
              (index < num_sources && sources[index] == cached) ? index : -1;
      }
  }
}

template <typename CT>
__global__ void BinarySearch_no_z_ostationary(
    std::size_t num_sources,                      
    std::size_t num_targets,
    std::size_t aligned_num_targets,                                                  
    std::size_t num_offsets,
    uint32_t kz,                   
    const CT* __restrict__ sources,               
    const CT* __restrict__ targets,
    const CT *__restrict__ offsets,              
    std::int32_t *__restrict__ kernel_map) {
  
  const auto gid = SPIRA_GLOBAL_THREAD_ID(x);
  const auto gsz = SPIRA_N_GLOBAL_THREADS(x);

  for (UIter t = gid; t < aligned_num_targets * num_offsets; t += gsz) {

      auto o = t % num_offsets;

      auto num_target = t /  num_offsets;

      if(num_target >= num_targets){
        kernel_map[num_target * num_offsets + o] = -1;
        return;
      }

      CT cached = targets[num_target] + offsets[o];

      auto index = device::BinarySearchMinimize<std::int32_t>(
                      0, num_sources,
                      [&](auto k) {
                             return sources[k];
                      }, cached);

      kernel_map[num_target * num_offsets + o] = (sources[index] == cached) ? index : -1;
    }
}

}