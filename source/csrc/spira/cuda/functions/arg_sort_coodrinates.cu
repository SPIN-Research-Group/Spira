#include "spira/cuda/context.cuh"
#include "spira/cuda/functions/arg_sort_coodrinates.cuh"
#include "spira/cuda/helpers.cuh"
#include "spira/cuda/kernels/fill_range.cuh"
#include "spira/cuda/memory.cuh"
#include "spira/enabled_arguments.h"
#include "spira/cuda/kernels/compress.cuh"


namespace spira::cuda {


template <typename CT>    
void Compress(std::size_t n,                           
        const std::int32_t *d_sources,           
        CT *d_targets,      
        const Context &context) {

    context.Launch(n, 128, 0,                                    
            kernels::Compress<CT>,                
            n, d_sources, d_targets);
}

template <typename CT>    
void UnCompress(std::size_t n,                           
        const CT *d_sources,     
        std::int32_t *d_targets, 
        const Context &context) {

    context.Launch(n, 128, 0,                                    
            kernels::UnCompress<CT>,                
            n, d_sources, d_targets);
}

template <typename CT>    
void FlattenSort::operator()(std::size_t n,                   
                                    const int32_t *d_coordinates,  
                                    int32_t *d_indices,            
                                    CT *sorted_compressed_coords,  
                                    const Context &context) const {

  auto d_compressed_buffer = context.NewBuffer<CT>(n); 
  
  CT *compressed_coordinates = d_compressed_buffer.device_data(); 

  Compress<CT>(n, d_coordinates, compressed_coordinates, context); 

  std::size_t d_temp_size;

  SPIRA_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
      nullptr,                                // d_temp_storage
      d_temp_size,                            // temp_storage_bytes
      static_cast<const CT*>(nullptr),        // d_keys_in
      static_cast<CT*>(nullptr),              // d_keys_out
      static_cast<const int32_t *>(nullptr),  // d_values_in
      static_cast<int32_t *>(nullptr),        // d_values_out,
      static_cast<int>(n),                    // num_items
      0,                                      // begin_bit
      sizeof(CT) * CHAR_BIT,                  // end_bit
      context.stream()                        // stream
      ));

  auto d_temp = context.NewBuffer(d_temp_size);
  auto d_indices_buffer = context.NewBuffer<int32_t>(n);

  int32_t *d_indices_original = d_indices_buffer.device_data(); 

  context.Launch(n,                     
                 128,                   
                 0,                       
                 kernels::FillRange<int32_t>,  
                 n, d_indices_original);
  
    const CT *d_keys_in = compressed_coordinates;

    SPIRA_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp.device_data(),         
        d_temp_size,                  
        d_keys_in,                   
        sorted_compressed_coords,     
        d_indices_original,           
        d_indices,                   
        static_cast<int>(n),          
        0,                            
        sizeof(CT) * CHAR_BIT,               
        context.stream()             
        ));
}

template <typename CT>    
void UnFlatten::operator()(std::size_t n,                   
                                    const CT *coordinates,  
                                    int32_t *uncompressed_coords,             
                                    const Context &context) const {


  UnCompress<CT>(n, coordinates, uncompressed_coords, context); 
}




template <typename CT>
void SortStreamed::operator()(std::size_t n,            
                  const CT *input_coordinates,  
                  CT *output_coordinates,
                  cudaStream_t stream,
                  void* d_temp,
                  std::size_t d_temp_size_sort,
                  std::size_t d_temp_size_unique,
                  std::uint32_t *d_num_uniques_ptr  
                  ) const{

  SPIRA_CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
      d_temp, 
      d_temp_size_sort,
      input_coordinates,  
      output_coordinates,
      static_cast<int>(n),
      0, 
      sizeof(CT) * CHAR_BIT, 
      stream
      ));

  SPIRA_CHECK_CUDA(cub::DeviceSelect::Unique(
      d_temp,
      d_temp_size_unique,
      output_coordinates,  
      output_coordinates,
      d_num_uniques_ptr,
      static_cast<int>(n),
      stream
      ));
}



template <typename CT>
void SortUnique::operator()(int n,            
                  const CT *input_coordinates,  
                  CT *output_coordinates,
                  std::uint32_t *d_num_uniques_ptr,   
                  const Context &context) const{

  std::size_t d_temp_size_sort;
  SPIRA_CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
      nullptr, 
      d_temp_size_sort,
      static_cast<CT*>(nullptr),  
      static_cast<CT*>(nullptr),
      n,
      0, 
      sizeof(CT) * CHAR_BIT, 
      context.stream()
      ));

  auto d_temp_sort = context.NewBuffer(d_temp_size_sort);

  SPIRA_CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
      d_temp_sort.device_data(), 
      d_temp_size_sort,
      input_coordinates,  
      output_coordinates,
      n,
      0, 
      sizeof(CT) * CHAR_BIT, 
      context.stream()
      ));

  std::size_t d_temp_size_unique;
  SPIRA_CHECK_CUDA(cub::DeviceSelect::Unique(
      nullptr, 
      d_temp_size_unique,
      output_coordinates,
      output_coordinates,
      d_num_uniques_ptr,
      n,
      context.stream()
      ));

  auto d_temp_unique = context.NewBuffer(d_temp_size_unique);

  SPIRA_CHECK_CUDA(cub::DeviceSelect::Unique(
      d_temp_unique.device_data(), 
      d_temp_size_unique,
      output_coordinates,  
      output_coordinates,
      d_num_uniques_ptr,
      static_cast<int>(n),
      context.stream()
      ));
}

#define SPIRA_EXPLICIT_INSTANTIATION(_, CT)                              \
  template void FlattenSort::operator()<CT>(std::size_t n,               \
                  const std::int32_t *d_coordinates,                     \
                  std::int32_t *d_indices,                               \
                  CT *sorted_compressed_coords,                          \
                  const Context &context) const;                         \
    template void UnFlatten::operator()<CT>(std::size_t n,               \
        const CT *coordinates,                                           \
        std::int32_t *uncompressed_coords,                               \
        const Context &context) const;                                   \
  template void SortStreamed::operator()<CT>(                            \
        std::size_t n,                                                   \
        const CT *input_coordinates,                                     \
        CT *output_coordinates,                                          \
        cudaStream_t stream,                                             \
        void* d_temp,                                                    \
        std::size_t d_temp_size_sort,                                    \
        std::size_t d_temp_size_unique,                                  \
        std::uint32_t *d_num_uniques_ptr) const;                         \
  template void SortUnique::operator()<CT>(                              \
        int n,                                                           \
        const CT *input_coordinates,                                     \
        CT *output_coordinates,                                          \
        std::uint32_t *d_num_uniques_ptr,                                \
        const Context &context) const                                   

SPIRA_FOR_ALL_C_TYPES(SPIRA_EXPLICIT_INSTANTIATION);

}  // namespace spira::cuda
