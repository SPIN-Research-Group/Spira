#pragma once

#include <cstdint>

#include "spira/cuda/context.cuh"
#include "spira/cuda/stream.cuh"

namespace spira::cuda {

    

class FlattenSort {
  public:
  template <typename CT>
  void operator()(std::size_t n,            
                  const std::int32_t *d_coordinates,  
                  std::int32_t *d_indices,            
                  CT *sorted_compressed_coords,  
                  const Context &context) const;
};

class UnFlatten {
  public:
  template <typename CT>
  void operator()(std::size_t n,            
                  const CT *coordinates,  
                  std::int32_t *uncompressed_coords,
                  const Context &context) const;
};



class SortUnique {
  public:
  template <typename CT>
  void operator()(int n,            
                  const CT *input_coordinates,  
                  CT *output_coordinates,
                  std::uint32_t *d_num_uniques_ptr,     
                  const Context &context) const;
};

class SortStreamed {
  public:
  template <typename CT>
  void operator()(std::size_t n,            
                  const CT *input_coordinates,  
                  CT *output_coordinates,
                  cudaStream_t stream,     
                  void* d_temp,
                  std::size_t d_temp_size_sort,
                  std::size_t d_temp_size_unique,
                  std::uint32_t *d_num_uniques_ptr  ) const;
};

}  // namespace spira::cuda
