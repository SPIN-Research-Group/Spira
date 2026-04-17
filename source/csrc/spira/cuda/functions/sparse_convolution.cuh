#pragma once

#include <optional>

#include "spira/cuda/context.cuh"

namespace spira::cuda {

struct SparseConvolutionForward_HS {
  template <typename FT>
  void operator()(std::size_t num_sources,                 
        std::size_t num_targets,                 
        std::size_t num_offsets_ws,
        std::size_t num_offsets_os,        
        const std::optional<double> &threshold,  
        std::size_t parallel,                    
        std::size_t num_source_features,  
        std::size_t num_target_features,  
        const int32_t *d_source_masks,         
        const int32_t *d_target_masks,       
        const int32_t *d_kernel_map_sizes,
        const int32_t *d_out_in_map,      
        const FT *d_sources,             
        const FT *d_weights_ws,
        const FT *d_weights_os,            
        FT *d_targets_ws, 
        FT *d_targets_os,            
        std::size_t gather_tile_size,     
        std::size_t scatter_tile_size,   
        const Context &context) const;
};


struct SparseConvolutionForward_HS_Merged {
  template <typename FT>
  void operator()(std::size_t num_sources,                 
                  std::size_t num_targets, 
                  std::size_t num_offsets_ws,
                  std::size_t num_offsets_os,                
                  std::size_t num_source_features,        
                  std::size_t num_target_features,
                  int32_t qsum_nnz,         
                  const int32_t *in_map_ptr,
                  const int32_t *out_map_ptr,
                  const int32_t *kpos_ptr,
                  const int32_t *qkpos_ptr, 
                  const int32_t *d_out_in_map,      
                  const FT *d_sources,                     
                  const FT *d_weights_ws,
                  const FT *d_weights_os,                  
                  FT *d_targets,              
                  const Context &context) const;
};

struct SparseConvolutionForward_WS {
  template <typename FT>
  void operator()(std::size_t num_sources,                 
                  std::size_t num_targets,                 
                  std::size_t num_offsets,                 
                  const std::optional<double> &threshold,  
                  std::size_t parallel,                    
                  std::size_t num_source_features,       
                  std::size_t num_target_features,         
                  const int32_t *d_source_masks,                
                  const int32_t *d_target_masks,             
                  const int32_t *d_kernel_map_sizes,           
                  const FT *d_sources,                     
                  const FT *d_weights,                     
                  FT *d_targets,                           
                  std::size_t gather_tile_size,            
                  std::size_t scatter_tile_size,           
                  const Context &context) const;
};

struct SparseConvolutionForward_WS_Merged {
  template <typename FT>
  void operator()(std::size_t num_sources,                 
                  std::size_t num_targets, 
                  std::size_t num_offsets,                
                  std::size_t num_source_features,        
                  std::size_t num_target_features,
                  int32_t qsum_nnz,
                  bool is_hybrid,         
                  const int32_t *in_map_ptr,
                  const int32_t *out_map_ptr,
                  const int32_t *kpos_ptr,
                  const int32_t *qkpos_ptr, 
                  const FT *d_sources,                     
                  const FT *d_weights,                  
                  FT *d_targets,                           
                  cudaStream_t stream) const;
};


struct SparseConvolutionForward_OS {
  template <typename FT>
  void operator()(std::size_t num_sources,              
    std::size_t num_targets,                 
    std::size_t num_offsets,                
    std::size_t num_source_features,       
    std::size_t num_target_features,        
    const int32_t *d_out_in_map,                 
    const FT *d_sources,                     
    const FT *d_weights,                     
    FT *d_targets,
    const Context &context,                           
    cudaStream_t stream) const;
  };

struct TimeGEMM {
  template <typename FT>
  float operator()(
      std::size_t num_sources,                 // S
      std::size_t num_targets,                 // T
      std::size_t num_offsets,                 // O
      bool is_hybrid,
      const std::optional<double> &threshold,  //
      std::size_t parallel,                    //
      std::size_t num_source_features,  // C_in
      std::size_t num_target_features,  // C_out
      const int32_t *d_kernel_map_sizes,     // [O]
      const FT *d_weights,              // [O, C_in, C_out]
      const Context &context) const;
};

struct TimeGather {
  template <typename FT>
  float operator()(
      std::size_t num_sources,                 // S
      std::size_t num_targets,                 // T
      std::size_t num_offsets,                 // O
      bool is_hybrid,
      const std::optional<double> &threshold,  //
      std::size_t num_source_features,  // C_in
      std::size_t num_target_features,  // C_out
      const int32_t *d_source_masks,         // [S, O]
      const int32_t *d_kernel_map_sizes,     // [O]
      std::size_t gather_tile_size,     //
      const Context &context) const;
};

struct TimeScatter {
  template <typename FT>
  float operator()(
      std::size_t num_sources,                 // S
      std::size_t num_targets,                 // T
      std::size_t num_offsets,                 // O
      bool is_hybrid,
      const std::optional<double> &threshold,  //
      std::size_t num_source_features,  // C_in
      std::size_t num_target_features,  // C_out
      const int32_t *d_target_masks,         // [T, O]
      const int32_t *d_kernel_map_sizes,     // [O]
      std::size_t tile_size,            //
      const Context &context) const;
};

}  // namespace spira::cuda