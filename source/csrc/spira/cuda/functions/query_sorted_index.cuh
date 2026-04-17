#pragma once

#include <cstdint>

#include "spira/cuda/context.cuh"
#include "spira/cuda/stream.cuh"

namespace spira::cuda {

class BinarySearch_Streamed_Os {
 public:
 template <typename CT>
  void operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,  
                  std::size_t num_offsets,  
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,
                  uint32_t kz,
                  uint32_t sz,      
                  std::int32_t *d_kernel_map,            
                  cudaStream_t stream) const;
};


class BinarySearch_Streamed_Ws {
 public:
 template <typename CT>
  void operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,    
                  std::size_t num_offsets,  
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,
                  uint32_t kz,
                  uint32_t sz,       
                  std::int32_t *d_kernel_map,
                  std::int64_t *nbmaps,
                  std::int32_t *nbsizes, 
                  std::int32_t *naddrs, 
                  std::int32_t *qnaddrs,
                  std::int32_t &map_size_val,
                  std::int32_t &qmap_size_val,
                  void* d_temp,
                  std::size_t d_temp_size,
                  std::int32_t *d_sizes,            
                  cudaStream_t stream) const;
};


class BinarySearch_Streamed_Ws_Half {
 public:
 template <typename CT>
  void operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,  
                  std::size_t num_offsets,  
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,
                  uint32_t kz,
                  uint32_t sz,        
                  std::int32_t *d_kernel_map,
                  std::int64_t *nbmaps,
                  std::int32_t *nbsizes, 
                  std::int32_t *naddrs, 
                  std::int32_t *qnaddrs,
                  std::int32_t &map_size_val,
                  std::int32_t &qmap_size_val,
                  void* d_temp,
                  std::size_t d_temp_size,
                  std::int32_t *d_sizes,            
                  cudaStream_t stream) const;
};


class BinarySearch_Streamed_GS {
public:
  template <typename CT>
  void operator()(std::size_t num_sources,
        std::size_t num_targets,
        std::size_t aligned_num_targets,
        std::size_t num_offsets,
        const CT *d_sources,      
        const CT *d_targets,
        const CT *d_offsets,
        uint32_t kz,
        uint32_t sz, 
        std::int32_t *d_kernel_map,
        std::int32_t *d_source_masks,            
        std::int32_t *d_target_masks,
        std::int32_t *nbsizes,
        std::int64_t *d_kernel_map_nonzero_indices, 
        std::int32_t *nb_sizes_cumsum,   
        void* d_temp,
        std::size_t d_temp_size,
        cudaStream_t stream) const;
};


class BinarySearch_Streamed_GS_Half {
public:
  template <typename CT>
  void operator()(std::size_t num_sources,
        std::size_t num_targets,
        std::size_t aligned_num_targets,
        std::size_t num_offsets,
        const CT *d_sources,      
        const CT *d_targets,
        const CT *d_offsets,
        uint32_t kz,
        uint32_t sz,    
        std::int32_t *d_kernel_map,
        std::int32_t *d_source_masks,            
        std::int32_t *d_target_masks,
        std::int32_t *nbsizes,
        std::int64_t *d_kernel_map_nonzero_indices, 
        std::int32_t *nb_sizes_cumsum,   
        void* d_temp,
        std::size_t d_temp_size,
        cudaStream_t stream) const;
};

//Hybrid


class BinarySearch_Streamed_Hybrid_GS {
public:
  template <typename CT>
  void operator()(std::size_t num_sources,  
                std::size_t num_targets,
                std::size_t aligned_num_targets,
                std::size_t num_offsets,
                std::int32_t dense_offsets,   
                const CT *d_sources,      
                const CT *d_targets,
                const CT *d_offsets,      
                uint32_t kz,
                uint32_t sz,     
                const std::uint32_t *d_indexes,
                std::int32_t *d_kernel_map,
                std::int32_t *d_source_masks,            
                std::int32_t *d_target_masks,
                std::int32_t *nbsizes,
                std::int64_t *d_kernel_map_nonzero_indices, 
                std::int32_t *nb_sizes_cumsum,   
                void* d_temp,
                std::size_t d_temp_size,
                cudaStream_t stream) const;
};

class BinarySearch_Streamed_Hybrid_WS {
 public:
 template <typename CT>
  void operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,  
                  std::size_t num_offsets,  
                  std::int32_t dense_offsets,   
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,      
                  uint32_t kz,
                  uint32_t sz,        
                  const std::uint32_t *d_indexes,   
                  std::int32_t *d_kernel_map,
                  std::int64_t *nbmaps,
                  std::int32_t *nbsizes, 
                  std::int32_t *naddrs, 
                  std::int32_t *qnaddrs,
                  std::int32_t &map_size_val,
                  std::int32_t &qmap_size_val,
                  void* d_temp,
                  std::size_t d_temp_size,
                  std::int32_t *d_sizes,            
                  cudaStream_t stream) const;
};

// small z

class BinarySearch_No_Small_Z {
 public:
 template <typename CT>
  void operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,  
                  std::size_t num_offsets,  
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,
                  uint32_t kz,    
                  std::int32_t *d_kernel_map,            
                  cudaStream_t stream) const;
};


//convert out_in_map transposed

class Convert_Transposed_Os {
 public:
  void operator()(const std::int32_t *d_out_in_map,  
                  std::int32_t *d_out_in_map_t,
                  std::size_t aligned_num_targets,
                  std::size_t num_offsets) const;
};

}  // namespace spira::cuda
