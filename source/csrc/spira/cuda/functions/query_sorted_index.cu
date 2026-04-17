#include "spira/cpu/memory.h"
#include "spira/cuda/helpers.cuh"
#include "spira/cuda/functions/query_sorted_index.cuh"
#include "spira/cuda/kernels/query_sorted.cuh"
#include "spira/cuda/kernels/quantifier_wstat.cuh"
#include "spira/cuda/kernels/convert_transposed_os.cuh"
#include "spira/enabled_arguments.h"
#include "spira/cuda/kernels/generate_masks_from_kernel_map.cuh"

namespace spira::cuda {

template <typename CT>    
void BinarySearch_Streamed_Os::operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,  
                  std::size_t num_offsets,  
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,
                  uint32_t kz,
                  uint32_t sz,       
                  std::int32_t *d_kernel_map,            
                  cudaStream_t stream) const {
       
    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    if(kz == 2){
        kernels::BinarySearch_kz_ostationary<CT, 2> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 3){
        kernels::BinarySearch_kz_ostationary<CT, 3> <<<  DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_ostationary<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, 3 * THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }
}            

template <typename CT>
void BinarySearch_Streamed_Ws::operator()(std::size_t num_sources,  
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
                  cudaStream_t stream) const {
       
    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    int32_t host_sizes[2];

    if(kz == 2){
        kernels::BinarySearch_kz_wstationary<CT, 2> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 3){
        kernels::BinarySearch_kz_wstationary<CT, 3> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_wstationary<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }
////////////////////////////////////////////////////////////////////////


    auto NonNegative = [kernel_map = d_kernel_map] SPIRA_DEVICE(UIter x) {
       return kernel_map[x] != -1;
    };

    auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    cub::DiscardOutputIterator d_unique_out;
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NonNegative, d_values_in);
    cub::DiscardOutputIterator d_num_out;
    SPIRA_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,       
                                        d_temp_size,                
                                        d_keys_in,                  
                                        d_unique_out,               
                                        d_values_in,                
                                        nbsizes,         
                                        d_num_out,             
                                        cub::Sum(),                 
                                        num_offsets * aligned_num_targets * kz,  
                                        stream));


    auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
        return entry != -1;
    };
    auto GetEntryPair = [aligned_num_targets,
                        kernel_map = d_kernel_map] SPIRA_DEVICE(auto index) {
        auto value = kernel_map[index];
        return (value != -1) ? (static_cast<std::int64_t>(index % aligned_num_targets) << 32) | static_cast<std::int32_t>(value) : -1;
    };
    
    CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, GetEntryPair, d_in);

    SPIRA_CHECK_CUDA(cub::DeviceSelect::If(d_temp,                        
                            d_temp_size,                                 
                            d_in,                                        
                            nbmaps,  
                            d_num_out,                          
                            kz * num_offsets * aligned_num_targets,                   
                            IsValidEntry,                                
                            stream));


    kernels::exclusive_scan_for_kernel_quantified<<<DivCeil(num_offsets * kz + 1, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            kz * num_offsets + 1,
            nbsizes,
            128,
            naddrs,
            qnaddrs, 
            d_sizes
    );

    SPIRA_CHECK_CUDA(
        cudaMemcpyAsync(
            host_sizes,
            d_sizes,
            2 * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream));

    map_size_val = host_sizes[0];
    qmap_size_val = host_sizes[1];
}

template <typename CT>
void BinarySearch_Streamed_Ws_Half::operator()(std::size_t num_sources,  
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
                  cudaStream_t stream) const {
       
    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    int32_t host_sizes[2];

    if(kz == 2){
        kernels::BinarySearch_kz_wstationary<CT, 2> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 3){
        kernels::BinarySearch_kz_wstationary<CT, 3> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_wstationary<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    auto NonNegative = [kernel_map = d_kernel_map] SPIRA_DEVICE(UIter x) {
       return kernel_map[x] != -1;
    };

    auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    cub::DiscardOutputIterator d_unique_out;
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NonNegative, d_values_in);
    cub::DiscardOutputIterator d_num_out;
    SPIRA_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,       
                                        d_temp_size,                
                                        d_keys_in,                  
                                        d_unique_out,               
                                        d_values_in,                
                                        nbsizes,         
                                        d_num_out,             
                                        cub::Sum(),                 
                                        ((num_offsets  * kz + 1) / 2) * aligned_num_targets,  
                                        stream));


    auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
        return entry != -1;
    };
    auto GetEntryPair = [aligned_num_targets,
                        kernel_map = d_kernel_map] SPIRA_DEVICE(auto index) {
        auto value = kernel_map[index];
        return (value != -1) ? (static_cast<std::int64_t>(index % aligned_num_targets) << 32) | static_cast<std::int32_t>(value) : -1;
    };
    
    CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, GetEntryPair, d_in);

    SPIRA_CHECK_CUDA(cub::DeviceSelect::If(d_temp,                        
                            d_temp_size,                                 
                            d_in,                                        
                            nbmaps,  
                            d_num_out,                          
                            ((num_offsets  * kz + 1) / 2) * aligned_num_targets,                   
                            IsValidEntry,                                
                            stream));

    kernels::exclusive_scan_for_kernel_quantified<<<DivCeil(((num_offsets * kz + 1) / 2) + 1, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
        ((num_offsets * kz + 1) / 2) + 1,
        nbsizes,
        128,
        naddrs,
        qnaddrs,
        d_sizes
    );

    SPIRA_CHECK_CUDA(
        cudaMemcpyAsync(
            host_sizes,
            d_sizes,
            2 * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream));

    map_size_val = host_sizes[0];
    qmap_size_val = host_sizes[1];
    
}



template <typename CT>
void BinarySearch_Streamed_GS::operator()(std::size_t num_sources,
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
        cudaStream_t stream) const {

    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    if(kz == 2){
        kernels::BinarySearch_kz_wstationary<CT, 2> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 3){
        kernels::BinarySearch_kz_wstationary<CT, 3> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_wstationary<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }

////////////////////////////////////////////////////////////////////////


    auto NonNegative = [kernel_map = d_kernel_map] SPIRA_DEVICE(UIter x) {
       return kernel_map[x] != -1;
    };

    auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    cub::DiscardOutputIterator d_unique_out;
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NonNegative, d_values_in);
    cub::DiscardOutputIterator d_num_out;
    SPIRA_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,       
                                        d_temp_size,                
                                        d_keys_in,                  
                                        d_unique_out,               
                                        d_values_in,                
                                        nbsizes,         
                                        d_num_out,             
                                        cub::Sum(),                 
                                        num_offsets * aligned_num_targets * kz,  
                                        stream));
    
    auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
      return entry != -1;
    };

    auto GetEntryPair = [num_sources, kernel_map = d_kernel_map] SPIRA_DEVICE(auto index) {
      auto value = kernel_map[index];
      return (value != -1) ? static_cast<std::int64_t>(index) * num_sources + value : -1;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, GetEntryPair, d_in);
    cub::DiscardOutputIterator d_num_selected_out;
    SPIRA_CHECK_CUDA(
        cub::DeviceSelect::If(d_temp,                        
                              d_temp_size,                                 
                              d_in,                                        
                              d_kernel_map_nonzero_indices,  
                              d_num_selected_out,                          
                              num_offsets * aligned_num_targets * kz,                    
                              IsValidEntry,                                
                              stream));


    kernels::exclusive_scan_for_kernel <<<DivCeil(kz * num_offsets + 1, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            kz * num_offsets + 1,
            nbsizes,
            nb_sizes_cumsum
    );

    SPIRA_CHECK_CUDA(cudaMemsetAsync(d_source_masks, -1, 4 * kz * num_offsets * num_sources, stream));

    SPIRA_CHECK_CUDA(cudaMemsetAsync(d_target_masks, -1, 4 * kz * num_offsets * num_targets, stream));

    int32_t actual_map_size;
    
    SPIRA_CHECK_CUDA(cudaMemcpyAsync(&actual_map_size, nb_sizes_cumsum + (kz * num_offsets), sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream));

    if(actual_map_size){
        kernels::GenerateMasksFromKernelMap<std::int32_t> <<<DivCeil<UIter>(actual_map_size, 512), 128, 0, stream>>>(
        actual_map_size,                             
        num_sources,                                 
        num_targets,
        aligned_num_targets,
        nb_sizes_cumsum,         
        d_kernel_map_nonzero_indices,  
        d_source_masks,                              
        d_target_masks);    
    }   
}

template <typename CT>
void BinarySearch_Streamed_GS_Half::operator()(std::size_t num_sources,
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
        cudaStream_t stream) const {

    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    if(kz == 2){
        kernels::BinarySearch_kz_wstationary<CT, 2> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 3){
        kernels::BinarySearch_kz_wstationary<CT, 3> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_wstationary<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }

////////////////////////////////////////////////////////////////////////

    auto NonNegative = [kernel_map = d_kernel_map] SPIRA_DEVICE(UIter x) {
       return kernel_map[x] != -1;
    };

    auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    cub::DiscardOutputIterator d_unique_out;
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NonNegative, d_values_in);
    cub::DiscardOutputIterator d_num_out;
    SPIRA_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,       
                                        d_temp_size,                
                                        d_keys_in,                  
                                        d_unique_out,               
                                        d_values_in,                
                                        nbsizes,         
                                        d_num_out,             
                                        cub::Sum(),                 
                                        ((num_offsets * kz + 1)/2) * aligned_num_targets,  
                                        stream));
    
    auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
      return entry != -1;
    };

    auto GetEntryPair = [num_sources, kernel_map = d_kernel_map] SPIRA_DEVICE(auto index) {
      auto value = kernel_map[index];
      return (value != -1) ? static_cast<std::int64_t>(index) * num_sources + value : -1;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, GetEntryPair, d_in);
    cub::DiscardOutputIterator d_num_selected_out;
    SPIRA_CHECK_CUDA(
        cub::DeviceSelect::If(d_temp,                        
                              d_temp_size,                                 
                              d_in,                                        
                              d_kernel_map_nonzero_indices,  
                              d_num_selected_out,                          
                              ((num_offsets * kz + 1)/2) * aligned_num_targets,                 
                              IsValidEntry,                                
                              stream));


    kernels::exclusive_scan_for_kernel <<<DivCeil(((num_offsets * kz + 1)/2) + 1, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            ((num_offsets * kz + 1)/2) + 1,
            nbsizes,
            nb_sizes_cumsum
    );

    SPIRA_CHECK_CUDA(cudaMemsetAsync(d_source_masks, -1, 4 * kz * num_offsets * num_sources, stream));

    SPIRA_CHECK_CUDA(cudaMemsetAsync(d_target_masks, -1, 4 * kz * num_offsets * num_targets, stream));

    int32_t actual_map_size;
    
    SPIRA_CHECK_CUDA(cudaMemcpyAsync(&actual_map_size, nb_sizes_cumsum + ((num_offsets * kz + 1)/2), sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream));

    if(actual_map_size){
        kernels::GenerateMasksFromKernelMap_Half<std::int32_t> <<<DivCeil<UIter>(actual_map_size, 512), 128, 0, stream>>>(
        actual_map_size,                             
        num_sources,                                 
        num_targets,
        kz * num_offsets,
        aligned_num_targets,
        nb_sizes_cumsum,         
        d_kernel_map_nonzero_indices,  
        d_source_masks,                              
        d_target_masks);    
    }   
}



/////////////////////////////////////////////////////////////////// HYBRID ////////////////////////////////////////////////////////

template <typename CT>
void BinarySearch_Streamed_Hybrid_GS::operator()(std::size_t num_sources,  
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
                cudaStream_t stream) const {

    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    if(kz == 3){
        kernels::BinarySearch_kz_hybrid<CT, 3> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_indexes,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_hybrid<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_indexes,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }

    auto NonNegative = [kernel_map = d_kernel_map, aligned_num_targets, dense_offsets] SPIRA_DEVICE(UIter x) {
       return kernel_map[x + dense_offsets * aligned_num_targets] != -1;
    };

    auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    cub::DiscardOutputIterator d_unique_out;
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NonNegative, d_values_in);
    cub::DiscardOutputIterator d_num_out;
    SPIRA_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,       
                                        d_temp_size,                
                                        d_keys_in,                  
                                        d_unique_out,               
                                        d_values_in,                
                                        nbsizes,         
                                        d_num_out,             
                                        cub::Sum(),                 
                                        ((kz * num_offsets - dense_offsets)/2) * aligned_num_targets,  
                                        stream));
    
    auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
      return entry != -1;
    };

    auto GetEntryPair = [num_sources, kernel_map = d_kernel_map, aligned_num_targets, dense_offsets] SPIRA_DEVICE(auto index) {
      auto value = kernel_map[index + dense_offsets * aligned_num_targets];
      return (value != -1) ? static_cast<std::int64_t>(index) * num_sources + value : -1;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, GetEntryPair, d_in);
    cub::DiscardOutputIterator d_num_selected_out;
    SPIRA_CHECK_CUDA(
        cub::DeviceSelect::If(d_temp,                        
                              d_temp_size,                                 
                              d_in,                                        
                              d_kernel_map_nonzero_indices,  
                              d_num_selected_out,                          
                              ((kz * num_offsets - dense_offsets)/2) * aligned_num_targets,                 
                              IsValidEntry,                                
                              stream));


    kernels::exclusive_scan_for_kernel <<<DivCeil(((kz * num_offsets - dense_offsets)/2) + 1, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            ((kz * num_offsets - dense_offsets)/2) + 1,
            nbsizes,
            nb_sizes_cumsum
    );

    SPIRA_CHECK_CUDA(cudaMemsetAsync(d_source_masks, -1, 4 * (kz * num_offsets - dense_offsets) * num_sources, stream));

    SPIRA_CHECK_CUDA(cudaMemsetAsync(d_target_masks, -1, 4 * (kz * num_offsets - dense_offsets) * num_targets, stream));

    int32_t actual_map_size;
    
    SPIRA_CHECK_CUDA(cudaMemcpyAsync(&actual_map_size, nb_sizes_cumsum + ((kz * num_offsets - dense_offsets)/2), sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream));

    if(actual_map_size){
        kernels::GenerateMasksFromKernelMap_Half<std::int32_t> <<<DivCeil<UIter>(actual_map_size, 512), 128, 0, stream>>>(
        actual_map_size,                             
        num_sources,                                 
        num_targets,
        kz * num_offsets - dense_offsets,
        aligned_num_targets,
        nb_sizes_cumsum,         
        d_kernel_map_nonzero_indices,  
        d_source_masks,                              
        d_target_masks);    
    }   
}


template <typename CT>
void BinarySearch_Streamed_Hybrid_WS::operator()(std::size_t num_sources,  
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
                  cudaStream_t stream) const {
       
    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    int32_t host_sizes[2];

    if(kz == 3){
        kernels::BinarySearch_kz_hybrid<CT, 3> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_indexes,
            d_kernel_map
        );
    }

    else if(kz == 5){
        kernels::BinarySearch_kz_hybrid<CT, 5> <<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            sz,        
            d_sources,
            d_targets,
            d_offsets,
            d_indexes,
            d_kernel_map
        );
    }
    else{
        SPIRA_ERROR("Currently Kernel Size in Z axis can only be 2, 3, 5");
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    auto NonNegative = [kernel_map = d_kernel_map, aligned_num_targets, dense_offsets] SPIRA_DEVICE(UIter x) {
       return kernel_map[x + dense_offsets * aligned_num_targets] != -1;
    };

    auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
    };

    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);
    cub::DiscardOutputIterator d_unique_out;
    CUB_TRANSFORMED_INPUT_ITERATOR(UIter, NonNegative, d_values_in);
    cub::DiscardOutputIterator d_num_out;
    SPIRA_CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,       
                                        d_temp_size,                
                                        d_keys_in,                  
                                        d_unique_out,               
                                        d_values_in,                
                                        nbsizes,         
                                        d_num_out,             
                                        cub::Sum(),                 
                                        ((num_offsets  * kz - dense_offsets) / 2) * aligned_num_targets,  
                                        stream));


    auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
      return entry != -1;
    };

    auto GetEntryPair = [kernel_map = d_kernel_map, aligned_num_targets, dense_offsets] SPIRA_DEVICE(auto index) {
        
        auto value = kernel_map[index + dense_offsets * aligned_num_targets];
        return (value != -1) ? (static_cast<std::int64_t>(index % aligned_num_targets) << 32) | static_cast<std::int32_t>(value) : -1;
    };
    
    CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, GetEntryPair, d_in);

    SPIRA_CHECK_CUDA(cub::DeviceSelect::If(d_temp,                        
                            d_temp_size,                                 
                            d_in,                                        
                            nbmaps,  
                            d_num_out,                          
                            ((num_offsets  * kz - dense_offsets) / 2) * aligned_num_targets,                   
                            IsValidEntry,                                
                            stream));


    kernels::exclusive_scan_for_kernel_quantified<<<DivCeil(((num_offsets * kz - dense_offsets) / 2) + 1, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
            ((num_offsets * kz - dense_offsets) / 2) + 1,
            nbsizes,
            128,
            naddrs,
            qnaddrs, 
            d_sizes
    );

    SPIRA_CHECK_CUDA(
        cudaMemcpyAsync(
            host_sizes,
            d_sizes,
            2 * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream));

    map_size_val = host_sizes[0];
    qmap_size_val = host_sizes[1];
}

///////////////////////////////////////////////////// NO Z ////////////////////////////////////////////////////   

template <typename CT>
void BinarySearch_No_Small_Z::operator()(std::size_t num_sources,  
                  std::size_t num_targets,
                  std::size_t aligned_num_targets,  
                  std::size_t num_offsets,  
                  const CT *d_sources,      
                  const CT *d_targets,
                  const CT *d_offsets,
                  uint32_t kz,    
                  std::int32_t *d_kernel_map,
                  cudaStream_t stream) const {
       
    constexpr const std::size_t THREAD_BLOCK_SIZE = 128;

    kernels::BinarySearch_no_z_ostationary<CT> <<<  DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE, 0, stream>>>(
        num_sources,
        num_targets,
        aligned_num_targets,
        num_offsets,
        kz,     
        d_sources,
        d_targets,
        d_offsets,
        d_kernel_map
    );
}            

//////////////////////////// helper Convert Tranposed - only needed for OS in UNET //////////////////////

void Convert_Transposed_Os::operator()(
    const std::int32_t *d_out_in_map,  
    std::int32_t *d_out_in_map_t,
    std::size_t aligned_num_targets,
    std::size_t num_offsets) const {

    constexpr const std::size_t THREAD_BLOCK_SIZE = 256;

    kernels::Convert_out_in_map_kernel<<<DivCeil(aligned_num_targets * num_offsets, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE>>>(
        d_out_in_map,
        d_out_in_map_t,
        aligned_num_targets,
        num_offsets
    );
}


#define SPIRA_EXPLICIT_INSTANTIATION(_, CT)                                   \
/* Streamed Os */                                                             \
template void spira::cuda::BinarySearch_Streamed_Os::operator()<CT>(          \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    const CT*, const CT*, const CT*,                                          \
    uint32_t, uint32_t,                                                       \
    std::int32_t*, cudaStream_t) const;                                       \
                                                                               \
/* Streamed Ws */                                                             \
template void spira::cuda::BinarySearch_Streamed_Ws::operator()<CT>(          \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    const CT*, const CT*, const CT*,                                          \
    uint32_t, uint32_t,                                                       \
    std::int32_t*, std::int64_t*, std::int32_t*, std::int32_t*, std::int32_t*, \
    std::int32_t&, std::int32_t&, void*, std::size_t, std::int32_t*, cudaStream_t) const; \
                                                                               \
/* Streamed Ws Half */                                                        \
template void spira::cuda::BinarySearch_Streamed_Ws_Half::operator()<CT>(     \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    const CT*, const CT*, const CT*,                                          \
    uint32_t, uint32_t,                                                        \
    std::int32_t*, std::int64_t*, std::int32_t*, std::int32_t*, std::int32_t*, \
    std::int32_t&, std::int32_t&, void*, std::size_t, std::int32_t*, cudaStream_t) const; \
                                                                               \
/* Streamed GS */                                                             \
template void spira::cuda::BinarySearch_Streamed_GS::operator()<CT>(          \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    const CT*, const CT*, const CT*,                                          \
    uint32_t, uint32_t,                                                         \
    std::int32_t*, std::int32_t*, std::int32_t*, std::int32_t*,               \
    std::int64_t*, std::int32_t*, void*, std::size_t, cudaStream_t) const;    \
                                                                               \
/* Streamed GS Half */                                                        \
template void spira::cuda::BinarySearch_Streamed_GS_Half::operator()<CT>(     \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    const CT*, const CT*, const CT*,                                           \
    uint32_t, uint32_t,                                                       \
    std::int32_t*, std::int32_t*, std::int32_t*, std::int32_t*,               \
    std::int64_t*, std::int32_t*, void*, std::size_t, cudaStream_t) const;    \
                                                                               \
/* Hybrid GS */                                                               \
template void spira::cuda::BinarySearch_Streamed_Hybrid_GS::operator()<CT>(   \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    std::int32_t, const CT*, const CT*, const CT*,                            \
    uint32_t, uint32_t,                                                       \
    const std::uint32_t*, std::int32_t*, std::int32_t*, std::int32_t*, std::int32_t*,        \
    std::int64_t*, std::int32_t*, void*, std::size_t, cudaStream_t) const;    \
                                                                               \
/* Hybrid WS */                                                               \
template void spira::cuda::BinarySearch_Streamed_Hybrid_WS::operator()<CT>(   \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    std::int32_t, const CT*, const CT*, const CT*,                             \
    uint32_t, uint32_t,                                                       \
    const std::uint32_t*, std::int32_t*, std::int64_t*, std::int32_t*,        \
    std::int32_t*, std::int32_t*, std::int32_t&, std::int32_t&, void*,        \
    std::size_t, std::int32_t*, cudaStream_t) const;                          \
                                                                               \
/* No Small Z */                                                              \
template void spira::cuda::BinarySearch_No_Small_Z::operator()<CT>(           \
    std::size_t, std::size_t, std::size_t, std::size_t,                        \
    const CT*, const CT*, const CT*, uint32_t,                                 \
    std::int32_t*, cudaStream_t) const;

SPIRA_FOR_ALL_C_TYPES(SPIRA_EXPLICIT_INSTANTIATION);

}  // namespace spira::cuda
