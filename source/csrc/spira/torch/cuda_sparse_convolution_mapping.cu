#include "spira/common/exception.h"
#include "spira/cuda/functions/query_sorted_index.cuh"
#include "spira/enabled_arguments.h"
#include "spira/torch/cuda_common.cuh"
#include "spira/cuda/helpers.cuh"
#include "spira/compress_layout.h"

namespace spira {    

template <typename CT>   
torch::Tensor generate_offsets(
    const std::tuple<int,int,int> &kernel_size,
    const std::tuple<int,int,int> &source_stride)
{
    auto kx = std::get<0>(kernel_size);
    auto ky = std::get<1>(kernel_size);
    auto kz = std::get<2>(kernel_size);

    auto sx = std::get<0>(source_stride);
    auto sy = std::get<1>(source_stride);
    auto sz = std::get<2>(source_stride);

    auto kx_min = (kx % 2 == 1) ? -(kx / 2) : -(kx / 2) + 1;
    auto kx_max = (kx / 2);

    auto ky_min = (ky % 2 == 1) ? -(ky / 2) : -(ky / 2) + 1;
    auto ky_max = (ky / 2);

    auto kz_min = (kz % 2 == 1) ? -(kz / 2) : -(kz / 2) + 1;


    auto nx = kx_max - kx_min + 1;
    auto ny = ky_max - ky_min + 1;
    auto n = nx * ny;

    auto compressed_cpu = torch::empty({n}, TypeConversion<CT>::TORCH_DTYPE);
    auto acc = TypeConversion<CT>::GetCppPointer(compressed_cpu);    

    int idx = 0;
    for (auto ix = kx_min; ix <= kx_max; ++ix) {
        for (auto iy = ky_min; iy <= ky_max; ++iy) {
            auto xx = ix * sx;
            auto yy = iy * sy;
            auto zz = kz_min * sz;
            acc[idx++] = xx * (static_cast<CT>(1) << Layout<CT>::X_SHIFT) + yy * (static_cast<CT>(1) << Layout<CT>::Y_SHIFT) + zz;
        }
    }

    return compressed_cpu;
}

template <typename CT>
torch::Tensor generate_offsets_all(
    const std::vector<std::tuple<int,int,int>>& kernel_sizes,
    const std::vector<std::tuple<int,int,int>>& source_strides)
{

    int64_t total_size = 0;

    for (size_t i = 0; i < kernel_sizes.size(); ++i) {
        auto kx = std::get<0>(kernel_sizes[i]);
        auto ky = std::get<1>(kernel_sizes[i]);

        auto kx_min = (kx % 2 == 1) ? -(kx / 2) : -(kx / 2) + 1;
        auto kx_max = (kx / 2);

        auto ky_min = (ky % 2 == 1) ? -(ky / 2) : -(ky / 2) + 1;
        auto ky_max = (ky / 2);

        auto nx = kx_max - kx_min + 1;
        auto ny = ky_max - ky_min + 1;

        total_size += nx * ny;
    }

    auto compressed_cpu_all = torch::empty({total_size}, TypeConversion<CT>::TORCH_DTYPE);
    auto acc = TypeConversion<CT>::GetCppPointer(compressed_cpu_all);

    int64_t idx = 0;

    for (size_t i = 0; i < kernel_sizes.size(); ++i) {

        auto kx = std::get<0>(kernel_sizes[i]);
        auto ky = std::get<1>(kernel_sizes[i]);
        auto kz = std::get<2>(kernel_sizes[i]);

        auto sx = std::get<0>(source_strides[i]);
        auto sy = std::get<1>(source_strides[i]);
        auto sz = std::get<2>(source_strides[i]);

        auto kx_min = (kx % 2 == 1) ? -(kx / 2) : -(kx / 2) + 1;
        auto kx_max = (kx / 2);

        auto ky_min = (ky % 2 == 1) ? -(ky / 2) : -(ky / 2) + 1;
        auto ky_max = (ky / 2);

        auto kz_min = (kz % 2 == 1) ? -(kz / 2) : -(kz / 2) + 1;

        for (int ix = kx_min; ix <= kx_max; ++ix) {
            for (int iy = ky_min; iy <= ky_max; ++iy) {

                auto xx = ix * sx;
                auto yy = iy * sy;
                auto zz = kz_min * sz;

                acc[idx++] = xx * (static_cast<CT>(1) << Layout<CT>::X_SHIFT) + yy * (static_cast<CT>(1) << Layout<CT>::Y_SHIFT) + zz;
            }
        }
    }

    return compressed_cpu_all;
}

template <typename CT>   
torch::Tensor generate_offsets_full(
    const std::tuple<int,int,int> &kernel_size,
    const std::tuple<int,int,int> &source_stride)
{
    auto kx = std::get<0>(kernel_size);
    auto ky = std::get<1>(kernel_size);
    auto kz = std::get<2>(kernel_size);

    auto sx = std::get<0>(source_stride);
    auto sy = std::get<1>(source_stride);
    auto sz = std::get<2>(source_stride);

    auto kx_min = (kx % 2 == 1) ? -(kx / 2) : -(kx / 2) + 1;
    auto kx_max = (kx / 2);

    auto ky_min = (ky % 2 == 1) ? -(ky / 2) : -(ky / 2) + 1;
    auto ky_max = (ky / 2);

    auto kz_min = (kz % 2 == 1) ? -(kz / 2) : -(kz / 2) + 1;
    auto kz_max = (kz / 2);

    auto nx = kx_max - kx_min + 1;
    auto ny = ky_max - ky_min + 1;
    auto nz = kz_max - kz_min + 1;
    auto n = nx * ny * nz;

    auto compressed_cpu_full = torch::empty({n}, TypeConversion<CT>::TORCH_DTYPE);
    auto acc = TypeConversion<CT>::GetCppPointer(compressed_cpu_full);    

    int idx = 0;
    for (auto ix = kx_min; ix <= kx_max; ++ix) {
        for (auto iy = ky_min; iy <= ky_max; ++iy) {
            for (auto iz = kz_min; iz <= kz_max; ++iz) {
                auto xx = ix * sx;
                auto yy = iy * sy;
                auto zz = iz * sz;
                acc[idx++] = xx * (static_cast<CT>(1) << Layout<CT>::X_SHIFT) + yy * (static_cast<CT>(1) << Layout<CT>::Y_SHIFT) + zz;
            }
        }
    }

    return compressed_cpu_full;
}

std::vector<int32_t> generate_indexes_L1(
    const uint32_t kx,
    const uint32_t ky,
    const uint32_t kz,
    int dataflow)
{
    const int threshold = (dataflow - 1) / 2;

    // Compute min/max using your scheme
    const int kx_min = (kx % 2 == 1) ? -(kx / 2) : -(kx / 2) + 1;
    const int ky_min = (ky % 2 == 1) ? -(ky / 2) : -(ky / 2) + 1;
    const int kz_min = (kz % 2 == 1) ? -(kz / 2) : -(kz / 2) + 1;

    const int kx_max = kx_min + kx - 1;
    const int ky_max = ky_min + ky - 1;
    const int kz_max = kz_min + kz - 1;

    const int N = kx * ky * kz;
    std::vector<int32_t> data(N);

    // Count front elements
    int front_count = 0;
    for (int x = kx_min; x <= kx_max; ++x)
        for (int y = ky_min; y <= ky_max; ++y)
            for (int z = kz_min; z <= kz_max; ++z)
                if (std::abs(x) + std::abs(y) + std::abs(z) < threshold)
                    ++front_count;

    int front_pos = 0;
    int back_pos  = front_count;

    int original_idx = 0;

    // Build inverse permutation
    for (int x = kx_min; x <= kx_max; ++x) {
        for (int y = ky_min; y <= ky_max; ++y) {
            for (int z = kz_min; z <= kz_max; ++z) {
                const int l1 = std::abs(x) + std::abs(y) + std::abs(z);

                if (l1 < threshold)
                    data[original_idx] = front_pos++;
                else
                    data[original_idx] = back_pos++;

                ++original_idx;
            }
        }
    }

    return data;
}

template <typename CT>    
std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>> BinarySearch(
        torch::Tensor &sources,
        torch::Tensor &targets,
        const std::tuple<int,int,int> &kernel_size,
        const std::tuple<int,int,int> &source_stride,
        int dataflow){

    const auto num_sources = sources.size(0);
    const auto num_targets = targets.size(0); 

    auto aligned_num_targets = (num_targets + 127)/128 * 128;

    auto device = GetTorchDeviceFromTensors({sources});
  
    auto context = GetCUDAContextFromTorchCUDADevice(device);

    context.Synchronize();

    uint32_t kz = std::get<2>(kernel_size);

    uint32_t sz = std::get<2>(source_stride);

    auto offsets = generate_offsets<CT>(kernel_size, source_stride).to(device);

    int64_t num_offsets = offsets.size(0);

    if(dataflow == 0){

       auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({aligned_num_targets, kz * num_offsets}, torch::TensorOptions(device).dtype(torch::kInt32)), c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

       cuda::BinarySearch_Streamed_Os().operator()<CT>(
          num_sources,
          num_targets,
          aligned_num_targets,
          num_offsets,
          TypeConversion<CT>::GetCppPointer(sources),                                                             
          TypeConversion<CT>::GetCppPointer(targets),
          TypeConversion<CT>::GetCppPointer(offsets),                                                                                                      
          kz,                                                                                                                                                                                                
          sz,      
          std::get<2>(kernel_map).data_ptr<int32_t>(),
          context.stream());

          return kernel_map;
    }

    else if (dataflow == 1){
      auto d_sizes = context.NewBuffer(sizeof(int32_t) * 2);
      auto d_sizes_ptr = static_cast<int32_t*>(d_sizes.device_data());

      std::int32_t mapsize_val = 0;
      std::int32_t qmapsize_val = 0;

      std::size_t d_temp_size_reduce = 0;
      std::size_t d_temp_size_select = 0;

      if((num_targets == num_sources ) && (kz * num_offsets) % 2 == 1){      //submanifold odd case - half map

        auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),            //weight-out-map
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets + 1) / 2) * num_targets, 2}, torch::TensorOptions(device).dtype(torch::kInt32))),          //nbmaps
                                c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1) / 2 }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                                c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1) / 2 + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //naddrs
                                c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1) / 2 + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //qnaddrs
                                c10::optional<std::int32_t>(mapsize_val),                                                                                                                //mapsize
                                c10::optional<std::int32_t>(qmapsize_val));                                                                                                              //qmapsize


        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
          return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<4>(kernel_map).value().data_ptr<int32_t>(),
                d_num_out,
                cub::Sum(),
                ((kz * num_offsets + 1)/2) * aligned_num_targets 
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
            return entry != -1;
        };


        auto dummy2 = [] SPIRA_DEVICE(auto index) {
            return static_cast<std::int64_t>(0);
        };
    
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                reinterpret_cast<int64_t*>(std::get<3>(kernel_map).value().data_ptr<int32_t>()),                             
                d_num_out,                
                ((kz * num_offsets + 1)/2) * aligned_num_targets, 
                IsValidEntry                                           
            )
        );
        
        auto d_temp_size = std::max(d_temp_size_reduce, d_temp_size_select);
        auto d_temp_storage = context.NewBuffer(d_temp_size);

        cuda::BinarySearch_Streamed_Ws_Half().operator()<CT>(
          num_sources,
          num_targets,
          aligned_num_targets,
          num_offsets,
          TypeConversion<CT>::GetCppPointer(sources),                                                             
          TypeConversion<CT>::GetCppPointer(targets),
          TypeConversion<CT>::GetCppPointer(offsets),                                                                                                      
          kz,                                                                                                                                                                                                       
          sz,    
          std::get<2>(kernel_map).data_ptr<int32_t>(),
          reinterpret_cast<int64_t*>(std::get<3>(kernel_map).value().data_ptr<int32_t>()),
          std::get<4>(kernel_map).value().data_ptr<int32_t>(),
          std::get<5>(kernel_map).value().data_ptr<int32_t>(),
          std::get<6>(kernel_map).value().data_ptr<int32_t>(),
          std::get<7>(kernel_map).value(),
          std::get<8>(kernel_map).value(),
          d_temp_storage.device_data(),
          d_temp_size,
          d_sizes_ptr,
          context.stream());


        auto& nbmaps = std::get<3>(kernel_map).value();
        auto mapsize_val = std::get<7>(kernel_map).value();

        nbmaps = nbmaps.narrow(0, 0, static_cast<int64_t>(mapsize_val)).t().contiguous();

        return kernel_map;
      }        


      else{              
        auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),                                     //weight-out-map
                                 c10::optional<torch::Tensor>(torch::empty({kz * num_offsets * num_targets, 2}, torch::TensorOptions(device).dtype(torch::kInt32))),            //nbmaps
                                 c10::optional<torch::Tensor>(torch::empty({kz * num_offsets }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                                 c10::optional<torch::Tensor>(torch::empty({kz * num_offsets + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //naddrs
                                 c10::optional<torch::Tensor>(torch::empty({kz * num_offsets + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //qnaddrs
                                 c10::optional<std::int32_t>(mapsize_val),                                                                                                      //mapsize
                                 c10::optional<std::int32_t>(qmapsize_val));                                                                                                    //qmapsize



        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
          return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<4>(kernel_map).value().data_ptr<int32_t>(),
                d_num_out,
                cub::Sum(),
                num_offsets * aligned_num_targets * kz
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
            return entry != -1;
        };


        auto dummy2 = [] SPIRA_DEVICE(auto index) {
            return static_cast<std::int64_t>(0);
        };
    
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                reinterpret_cast<int64_t*>(std::get<3>(kernel_map).value().data_ptr<int32_t>()),                             
                d_num_out,                
                num_offsets * aligned_num_targets * kz, 
                IsValidEntry                                           
            )
        );

        auto d_temp_size = std::max(d_temp_size_reduce, d_temp_size_select);
        auto d_temp_storage = context.NewBuffer(d_temp_size);

        cuda::BinarySearch_Streamed_Ws().operator()<CT>(
          num_sources,
          num_targets,
          aligned_num_targets,
          num_offsets,
          TypeConversion<CT>::GetCppPointer(sources),                                                             
          TypeConversion<CT>::GetCppPointer(targets),
          TypeConversion<CT>::GetCppPointer(offsets),    
          kz,
          sz,
          std::get<2>(kernel_map).data_ptr<int32_t>(),
          reinterpret_cast<int64_t*>(std::get<3>(kernel_map).value().data_ptr<int32_t>()),
          std::get<4>(kernel_map).value().data_ptr<int32_t>(),
          std::get<5>(kernel_map).value().data_ptr<int32_t>(),
          std::get<6>(kernel_map).value().data_ptr<int32_t>(),
          std::get<7>(kernel_map).value(),
          std::get<8>(kernel_map).value(),
          d_temp_storage.device_data(),
          d_temp_size,
          d_sizes_ptr,
          context.stream());
                
        auto& nbmaps = std::get<3>(kernel_map).value();
        auto mapsize_val = std::get<7>(kernel_map).value();

        nbmaps = nbmaps.narrow(0, 0, static_cast<int64_t>(mapsize_val)).t().contiguous();

        return kernel_map;
      }
    }

    else if(dataflow == 2){

        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        if((num_targets == num_sources) && (kz * num_offsets) % 2 == 1){

            auto d_kernel_map_nonzero_indices = context.NewBuffer<std::int64_t>((kz * num_offsets + 1)/2 * num_targets);
 
            auto nb_sizes_cumsum = context.NewBuffer<std::int32_t>((kz * num_offsets + 1)/2 + 1);
             
            auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),   //weight-out-map
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_sources }, torch::TensorOptions(device).dtype(torch::kInt32))),                //source_masks
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_targets }, torch::TensorOptions(device).dtype(torch::kInt32))),                //target_masks
                                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1)/2}, torch::TensorOptions(device).dtype(torch::kInt32))),                      //nbsizes
                                    c10::nullopt, c10::nullopt, c10::nullopt);                                                    
            
            auto dummy1 = [] SPIRA_DEVICE(UIter) {
                return false;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

            auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
            return x / aligned_num_targets;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

            cub::DiscardOutputIterator d_unique_out;
            //cub::DiscardOutputIterator dummy_out;
            cub::DiscardOutputIterator d_num_out;

            // Compute temporary storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceReduce::ReduceByKey(
                    nullptr,
                    d_temp_size_reduce,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    std::get<5>(kernel_map).value().data_ptr<int32_t>(),       //nbsizes
                    d_num_out,
                    cub::Sum(),
                    aligned_num_targets * ((kz * num_offsets + 1)/2)
                )
            );

            auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                    return entry != -1;
            };

            auto dummy2 = [] SPIRA_DEVICE(auto index) {
                    return static_cast<std::int64_t>(0);
            };
            
            CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

            // Compute temp storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceSelect::If(
                    nullptr,                          
                    d_temp_size_select,                
                    d_in,                              
                    std::get<2>(kernel_map).data_ptr<int32_t>(),                             
                    d_num_out,                
                    aligned_num_targets * ((kz * num_offsets + 1)/2), 
                    IsValidEntry                                           
                )
            );

            auto d_temp_size = std::max(d_temp_size_reduce, d_temp_size_select);
            auto d_temp_storage = context.NewBuffer(d_temp_size);


            cuda::BinarySearch_Streamed_GS_Half().operator()<CT>(
                num_sources,
                num_targets,
                aligned_num_targets,
                num_offsets,
                TypeConversion<CT>::GetCppPointer(sources),                                                             
                TypeConversion<CT>::GetCppPointer(targets),
                TypeConversion<CT>::GetCppPointer(offsets),    
                kz,
                sz,
                std::get<2>(kernel_map).data_ptr<int32_t>(),
                std::get<3>(kernel_map).value().data_ptr<int32_t>(),
                std::get<4>(kernel_map).value().data_ptr<int32_t>(),
                std::get<5>(kernel_map).value().data_ptr<int32_t>(),
                d_kernel_map_nonzero_indices.device_data(),
                nb_sizes_cumsum.device_data(),
                d_temp_storage.device_data(),
                d_temp_size,
                context.stream());

            return kernel_map;                        
        }

        else{
            auto d_kernel_map_nonzero_indices = context.NewBuffer<std::int64_t>(kz * num_offsets * num_targets);
            
            auto nb_sizes_cumsum = context.NewBuffer<std::int32_t>(kz * num_offsets + 1);

            auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),   //weight-out-map
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_sources }, torch::TensorOptions(device).dtype(torch::kInt32))),               //source_masks
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_targets }, torch::TensorOptions(device).dtype(torch::kInt32))),               //target_masks
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                                    c10::nullopt, c10::nullopt, c10::nullopt);                                                    
                                    
            auto dummy1 = [] SPIRA_DEVICE(UIter) {
                return false;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

            auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
                return x / aligned_num_targets;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

            cub::DiscardOutputIterator d_unique_out;
            //cub::DiscardOutputIterator dummy_out;
            cub::DiscardOutputIterator d_num_out;

            // Compute temporary storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceReduce::ReduceByKey(
                    nullptr,
                    d_temp_size_reduce,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    std::get<5>(kernel_map).value().data_ptr<int32_t>(),       //nbsizes
                    d_num_out,
                    cub::Sum(),
                    num_offsets * aligned_num_targets * kz
                )
            );

            auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                    return entry != -1;
            };

            auto dummy2 = [] SPIRA_DEVICE(auto index) {
                    return static_cast<std::int64_t>(0);
            };
            
            CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

            // Compute temp storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceSelect::If(
                    nullptr,                          
                    d_temp_size_select,                
                    d_in,                              
                    std::get<2>(kernel_map).data_ptr<int32_t>(),                             
                    d_num_out,                
                    num_offsets * aligned_num_targets * kz, 
                    IsValidEntry                                           
                )
            );

            auto d_temp_size = std::max(d_temp_size_reduce, d_temp_size_select);
            auto d_temp_storage = context.NewBuffer(d_temp_size);

            cuda::BinarySearch_Streamed_GS().operator()<CT>(
                num_sources,
                num_targets,
                aligned_num_targets,
                num_offsets,
                TypeConversion<CT>::GetCppPointer(sources),                                                             
                TypeConversion<CT>::GetCppPointer(targets),
                TypeConversion<CT>::GetCppPointer(offsets),    
                kz,
                sz,
                std::get<2>(kernel_map).data_ptr<int32_t>(),
                std::get<3>(kernel_map).value().data_ptr<int32_t>(),
                std::get<4>(kernel_map).value().data_ptr<int32_t>(),
                std::get<5>(kernel_map).value().data_ptr<int32_t>(),
                d_kernel_map_nonzero_indices.device_data(),
                nb_sizes_cumsum.device_data(),
                d_temp_storage.device_data(),
                d_temp_size,
                context.stream());

            return kernel_map;
    }
}

    else if(dataflow % 2 == 0){
        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        uint32_t kx = std::get<0>(kernel_size);
        uint32_t ky = std::get<1>(kernel_size);

        auto data = generate_indexes_L1(kx, ky, kz, dataflow);

        auto dense_offsets = data[0];

        auto indexes = torch::tensor(data, torch::dtype(torch::kInt32)).to(device);

        auto d_kernel_map_nonzero_indices = context.NewBuffer<std::int64_t>(((kz * num_offsets - dense_offsets)/2) * num_targets);

        auto nb_sizes_cumsum = context.NewBuffer<std::int32_t>(((kz * num_offsets - dense_offsets)/2) + 1);
            
        auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),   //weight-out-map
                                c10::optional<torch::Tensor>(torch::empty({kz * num_offsets - dense_offsets, num_sources }, torch::TensorOptions(device).dtype(torch::kInt32))),                //source_masks
                                c10::optional<torch::Tensor>(torch::empty({kz * num_offsets - dense_offsets, num_targets }, torch::TensorOptions(device).dtype(torch::kInt32))),                //target_masks
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets)/2)}, torch::TensorOptions(device).dtype(torch::kInt32))),                              //nbsizes
                                c10::nullopt, c10::nullopt, c10::nullopt);                                                    
        
        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<5>(kernel_map).value().data_ptr<int32_t>(),       //nbsizes
                d_num_out,
                cub::Sum(),
                aligned_num_targets * ((kz * num_offsets - dense_offsets)/2)
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                return entry != -1;
        };

        auto dummy2 = [] SPIRA_DEVICE(auto index) {
                return static_cast<std::int64_t>(0);
        };
        
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                std::get<2>(kernel_map).data_ptr<int32_t>(),                             
                d_num_out,                
                aligned_num_targets * ((kz * num_offsets - dense_offsets)/2), 
                IsValidEntry                                           
            )
        );

        auto d_temp_size = std::max(d_temp_size_reduce, d_temp_size_select);
        auto d_temp_storage = context.NewBuffer(d_temp_size);

        cuda::BinarySearch_Streamed_Hybrid_GS().operator()<CT>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            dense_offsets,
            TypeConversion<CT>::GetCppPointer(sources),                                                             
            TypeConversion<CT>::GetCppPointer(targets),
            TypeConversion<CT>::GetCppPointer(offsets),    
            kz,
            sz,
            reinterpret_cast<const uint32_t*>(indexes.data_ptr<int32_t>()),
            std::get<2>(kernel_map).data_ptr<int32_t>(),
            std::get<3>(kernel_map).value().data_ptr<int32_t>(),
            std::get<4>(kernel_map).value().data_ptr<int32_t>(),
            std::get<5>(kernel_map).value().data_ptr<int32_t>(),
            d_kernel_map_nonzero_indices.device_data(),
            nb_sizes_cumsum.device_data(),
            d_temp_storage.device_data(),
            d_temp_size,
            context.stream());

        if(dataflow > 4){    
            auto& out_in_map = std::get<2>(kernel_map);
            out_in_map = out_in_map.narrow(0, 0, static_cast<int64_t>(dense_offsets)).t().contiguous();
        }
        return kernel_map;   
    }
    
    else{
        auto d_sizes = context.NewBuffer(sizeof(int32_t) * 2);
        auto d_sizes_ptr = static_cast<int32_t*>(d_sizes.device_data());

        std::int32_t mapsize_val = 0;
        std::int32_t qmapsize_val = 0;

        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        uint32_t kx = std::get<0>(kernel_size);
        uint32_t ky = std::get<1>(kernel_size);

        auto data = generate_indexes_L1(kx, ky, kz, dataflow);

        auto dense_offsets = data[0];

        auto indexes = torch::tensor(data, torch::dtype(torch::kInt32)).to(device);

        auto kernel_map = std::make_tuple(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),            //weight-out-map
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets) / 2) * num_targets, 2}, torch::TensorOptions(device).dtype(torch::kInt32))),          //nbmaps
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets)/2) }, torch::TensorOptions(device).dtype(torch::kInt32))),                //nbsizes
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets)/2) + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),             //naddrs
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets)/2) + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),             //qnaddrs
                                c10::optional<std::int32_t>(mapsize_val),                                                                                                                //mapsize
                                c10::optional<std::int32_t>(qmapsize_val)); 
                                               
        
        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<4>(kernel_map).value().data_ptr<int32_t>(),       //nbsizes
                d_num_out,
                cub::Sum(),
                aligned_num_targets * ((kz * num_offsets - dense_offsets)/2)
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
            return entry != -1;
        };


        auto dummy2 = [] SPIRA_DEVICE(auto index) {
            return static_cast<std::int64_t>(0);
        };
    
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                reinterpret_cast<int64_t*>(std::get<3>(kernel_map).value().data_ptr<int32_t>()),                             
                d_num_out,                
                ((kz * num_offsets - dense_offsets)/2) * aligned_num_targets, 
                IsValidEntry                                           
            )
        );

        auto d_temp_size = std::max(d_temp_size_reduce, d_temp_size_select);
        auto d_temp_storage = context.NewBuffer(d_temp_size);

        cuda::BinarySearch_Streamed_Hybrid_WS().operator()<CT>(
          num_sources,
          num_targets,
          aligned_num_targets,
          num_offsets,
          dense_offsets,
          TypeConversion<CT>::GetCppPointer(sources),                                                             
          TypeConversion<CT>::GetCppPointer(targets),
          TypeConversion<CT>::GetCppPointer(offsets),    
          kz,
          sz,
          reinterpret_cast<const uint32_t*>(indexes.data_ptr<int32_t>()),
          std::get<2>(kernel_map).data_ptr<int32_t>(),
          reinterpret_cast<int64_t*>(std::get<3>(kernel_map).value().data_ptr<int32_t>()),
          std::get<4>(kernel_map).value().data_ptr<int32_t>(),
          std::get<5>(kernel_map).value().data_ptr<int32_t>(),
          std::get<6>(kernel_map).value().data_ptr<int32_t>(),
          std::get<7>(kernel_map).value(),
          std::get<8>(kernel_map).value(),
          d_temp_storage.device_data(),
          d_temp_size,
          d_sizes_ptr,
          context.stream()
        );

        auto& nbmaps = std::get<3>(kernel_map).value();
        mapsize_val = std::get<7>(kernel_map).value();

        nbmaps = nbmaps.narrow(0, 0, static_cast<int64_t>(mapsize_val)).t().contiguous();

        if(dataflow > 4){    
            auto& out_in_map = std::get<2>(kernel_map);
            out_in_map = out_in_map.narrow(0, 0, static_cast<int64_t>(dense_offsets)).t().contiguous();
        }

        return kernel_map;   
    }
}

template <typename CT>    
std::vector<std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>>> BinarySearchStreamed(
    const std::vector<torch::Tensor> &voxels,
    std::vector<std::tuple<int,int,int>> &kernel_sizes,
    std::vector<std::tuple<int,int,int>> &source_strides,
    std::vector<int> &dataflow) {

  const size_t num_maps = kernel_sizes.size();

  auto device = GetTorchDeviceFromTensors({voxels});
  
  auto context = GetCUDAContextFromTorchCUDADevice(device);

  context.Synchronize();

  std::vector<std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>>> kernel_maps;
  kernel_maps.reserve(num_maps);

  std::vector<c10::optional<std::pair<torch::Tensor, torch::Tensor>>> metadata_buffers;
  metadata_buffers.reserve(num_maps);

  std::vector<c10::optional<torch::Tensor>> sizes;
  sizes.reserve(num_maps);

  std::vector<c10::optional<torch::Tensor>> indexes;
  indexes.reserve(num_maps);

  std::vector<std::size_t> temp_offsets;
  temp_offsets.reserve(num_maps + 1);
  
  std::vector<int32_t> dense_offsets;
  dense_offsets.reserve(num_maps);

  std::size_t total_temp_size = 0;

  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(num_maps);

  temp_offsets.emplace_back(0);

  auto offsets_all = generate_offsets_all<CT>(kernel_sizes, source_strides).to(device);

  for (size_t i = 0; i < num_maps; ++i) {

    const uint32_t kx = std::get<0>(kernel_sizes[i]);
    const uint32_t ky = std::get<1>(kernel_sizes[i]);
    const uint32_t kz = std::get<2>(kernel_sizes[i]);

    int64_t num_offsets = kx * ky;

    const auto num_sources = voxels[i/2].size(0);
    const auto num_targets = voxels[(i+1)/2].size(0);

    auto aligned_num_targets = (num_targets + 127)/128 * 128;

    streams.emplace_back(c10::cuda::getStreamFromPool(false, device.index()));

    if(dataflow[i] == 0){
       dense_offsets.emplace_back(-1);

       sizes.emplace_back(c10::nullopt);

       indexes.emplace_back(c10::nullopt);

       metadata_buffers.emplace_back(c10::nullopt);

       temp_offsets.emplace_back(total_temp_size);
      
       kernel_maps.emplace_back(num_sources, num_targets, torch::empty({aligned_num_targets, kz * num_offsets}, torch::TensorOptions(device).dtype(torch::kInt32)), c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
    }

    else if(dataflow[i] == 1){
        dense_offsets.emplace_back(-1);

        indexes.emplace_back(c10::nullopt);

        metadata_buffers.emplace_back(c10::nullopt);

        sizes.emplace_back(torch::empty({2}, torch::TensorOptions(device).dtype(torch::kInt32)));

        std::int32_t mapsize_val = 0;
        std::int32_t qmapsize_val = 0;

        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        if((num_targets == num_sources ) && (kz * num_offsets) % 2 == 1){      //submanifold odd case - half map

            kernel_maps.emplace_back(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),                                    //weight-out-map
                    c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets + 1) / 2) * num_targets, 2}, torch::TensorOptions(device).dtype(torch::kInt32))),          //nbmaps
                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1) / 2 }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1) / 2 + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //naddrs
                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets + 1) / 2 + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //qnaddrs
                    c10::optional<std::int32_t>(mapsize_val),                                                                                                                //mapsize
                    c10::optional<std::int32_t>(qmapsize_val));                                                                                                              //qmapsize


        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
          return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
                d_num_out,
                cub::Sum(),
                ((kz * num_offsets + 1)/2) * aligned_num_targets 
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
            return entry != -1;
        };


        auto dummy2 = [] SPIRA_DEVICE(auto index) {
            return static_cast<std::int64_t>(0);
        };
    
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                reinterpret_cast<int64_t*>(std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>()),                             
                d_num_out,                
                ((kz * num_offsets + 1)/2) * aligned_num_targets, 
                IsValidEntry                                           
            )
        );

        }
        else{
            kernel_maps.emplace_back(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),                                     //weight-out-map
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets * num_targets, 2}, torch::TensorOptions(device).dtype(torch::kInt32))),            //nbmaps
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //naddrs
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //qnaddrs
                                    c10::optional<std::int32_t>(mapsize_val),                                                                                                      //mapsize
                                    c10::optional<std::int32_t>(qmapsize_val));                                                                                                    //qmapsize



            auto dummy1 = [] SPIRA_DEVICE(UIter) {
                return false;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

            auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
            return x / aligned_num_targets;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

            cub::DiscardOutputIterator d_unique_out;
            //cub::DiscardOutputIterator dummy_out;
            cub::DiscardOutputIterator d_num_out;

            // Compute temporary storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceReduce::ReduceByKey(
                    nullptr,
                    d_temp_size_reduce,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    d_num_out,
                    cub::Sum(),
                    num_offsets * aligned_num_targets * kz
                )
            );

            auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                return entry != -1;
            };


            auto dummy2 = [] SPIRA_DEVICE(auto index) {
                return static_cast<std::int64_t>(0);
            };
        
            CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

            // Compute temp storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceSelect::If(
                    nullptr,                          
                    d_temp_size_select,                
                    d_in,                              
                    reinterpret_cast<int64_t*>(std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>()),                             
                    d_num_out,                
                    num_offsets * aligned_num_targets * kz, 
                    IsValidEntry                                           
                )
            );

        }
        total_temp_size += std::max(d_temp_size_reduce, d_temp_size_select);
        temp_offsets.emplace_back(total_temp_size);
    }
    
    else if(dataflow[i] == 2){
        dense_offsets.emplace_back(-1);

        indexes.emplace_back(c10::nullopt);

        sizes.emplace_back(c10::nullopt);

        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        if(i % 2 == 0 && (kz * num_offsets) % 2 == 1){
            metadata_buffers.emplace_back(std::pair{
                torch::empty({((kz * num_offsets + 1)/2) * num_targets}, torch::TensorOptions(device).dtype(torch::kInt64)),
                torch::empty({((kz * num_offsets + 1)/2) + 1}, torch::TensorOptions(device).dtype(torch::kInt32)), 
            });

            kernel_maps.emplace_back(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),   //weight-out-map
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_sources }, torch::TensorOptions(device).dtype(torch::kInt32))),               //source_masks
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_targets }, torch::TensorOptions(device).dtype(torch::kInt32))),               //target_masks
                                    c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets + 1)/2)}, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                                    c10::nullopt, c10::nullopt, c10::nullopt);                                                                    
                                    
            auto dummy1 = [] SPIRA_DEVICE(UIter) {
                return false;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

            auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
            return x / aligned_num_targets;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

            cub::DiscardOutputIterator d_unique_out;
            //cub::DiscardOutputIterator dummy_out;
            cub::DiscardOutputIterator d_num_out;

            // Compute temporary storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceReduce::ReduceByKey(
                    nullptr,
                    d_temp_size_reduce,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),       //nbsizes
                    d_num_out,
                    cub::Sum(),
                    ((kz * num_offsets + 1)/2) * aligned_num_targets
                )
            );

            auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                    return entry != -1;
            };

            auto dummy2 = [] SPIRA_DEVICE(auto index) {
                    return static_cast<std::int64_t>(0);
            };
            
            CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

            // Compute temp storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceSelect::If(
                    nullptr,                          
                    d_temp_size_select,                
                    d_in,                              
                    std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),                             
                    d_num_out,                
                    ((kz * num_offsets + 1)/2) * aligned_num_targets, 
                    IsValidEntry                                           
                )
            );
        }


        else{
            metadata_buffers.emplace_back(std::pair{
                torch::empty({kz * num_offsets * num_targets}, torch::TensorOptions(device).dtype(torch::kInt64)),
                torch::empty({kz * num_offsets + 1}, torch::TensorOptions(device).dtype(torch::kInt32)), 
            });

            kernel_maps.emplace_back(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),   //weight-out-map
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_sources }, torch::TensorOptions(device).dtype(torch::kInt32))),               //source_masks
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets, num_targets }, torch::TensorOptions(device).dtype(torch::kInt32))),               //target_masks
                                    c10::optional<torch::Tensor>(torch::empty({kz * num_offsets }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                                    c10::nullopt, c10::nullopt, c10::nullopt);                                                                    
                                    
            auto dummy1 = [] SPIRA_DEVICE(UIter) {
                return false;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

            auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
            return x / aligned_num_targets;
            };

            CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

            cub::DiscardOutputIterator d_unique_out;
            //cub::DiscardOutputIterator dummy_out;
            cub::DiscardOutputIterator d_num_out;

            // Compute temporary storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceReduce::ReduceByKey(
                    nullptr,
                    d_temp_size_reduce,
                    d_keys_in,
                    d_unique_out,
                    d_values_in,
                    std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),       //nbsizes
                    d_num_out,
                    cub::Sum(),
                    num_offsets * aligned_num_targets * kz
                )
            );

            auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                    return entry != -1;
            };

            auto dummy2 = [] SPIRA_DEVICE(auto index) {
                    return static_cast<std::int64_t>(0);
            };
            
            CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

            // Compute temp storage size
            SPIRA_CHECK_CUDA(
                cub::DeviceSelect::If(
                    nullptr,                          
                    d_temp_size_select,                
                    d_in,                              
                    std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),                             
                    d_num_out,                
                    num_offsets * aligned_num_targets * kz, 
                    IsValidEntry                                           
                )
            );
        }

        total_temp_size += std::max(d_temp_size_reduce, d_temp_size_select);
        temp_offsets.emplace_back(total_temp_size);
    }

    else if(dataflow[i] % 2 == 0){
        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        auto data = generate_indexes_L1(kx, ky, kz, dataflow[i]);

        dense_offsets.emplace_back(data[0]);

        indexes.emplace_back(
            torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(device))
        );

        metadata_buffers.emplace_back(std::pair{
                torch::empty({((kz * num_offsets - dense_offsets[i])/2) * num_targets}, torch::TensorOptions(device).dtype(torch::kInt64)),
                torch::empty({((kz * num_offsets - dense_offsets[i])/2) + 1}, torch::TensorOptions(device).dtype(torch::kInt32)), 
            });

        sizes.emplace_back(c10::nullopt);
            
        kernel_maps.emplace_back(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),   //weight-out-map
                                c10::optional<torch::Tensor>(torch::empty({kz * num_offsets - dense_offsets[i], num_sources }, torch::TensorOptions(device).dtype(torch::kInt32))),                //source_masks
                                c10::optional<torch::Tensor>(torch::empty({kz * num_offsets - dense_offsets[i], num_targets }, torch::TensorOptions(device).dtype(torch::kInt32))),                //target_masks
                                c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets[i])/2)}, torch::TensorOptions(device).dtype(torch::kInt32))),                              //nbsizes
                                c10::nullopt, c10::nullopt, c10::nullopt);                                                    
        
        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
        return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),       //nbsizes
                d_num_out,
                cub::Sum(),
                aligned_num_targets * ((kz * num_offsets - dense_offsets[i])/2)
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
                return entry != -1;
        };

        auto dummy2 = [] SPIRA_DEVICE(auto index) {
                return static_cast<std::int64_t>(0);
        };
        
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),                             
                d_num_out,                
                aligned_num_targets * ((kz * num_offsets - dense_offsets[i])/2), 
                IsValidEntry                                           
            )
        );

        total_temp_size += std::max(d_temp_size_reduce, d_temp_size_select);
        temp_offsets.emplace_back(total_temp_size);
    }

    else{
        auto data = generate_indexes_L1(kx, ky, kz, dataflow[i]);

        dense_offsets.emplace_back(data[0]);

        indexes.emplace_back(
            torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(device))
        );

        metadata_buffers.emplace_back(c10::nullopt);

        sizes.emplace_back(torch::empty({2}, torch::TensorOptions(device).dtype(torch::kInt32)));

        std::int32_t mapsize_val = 0;
        std::int32_t qmapsize_val = 0;

        std::size_t d_temp_size_reduce = 0;
        std::size_t d_temp_size_select = 0;

        kernel_maps.emplace_back(num_sources, num_targets, torch::empty({kz * num_offsets, aligned_num_targets}, torch::TensorOptions(device).dtype(torch::kInt32)),         //weight-out-map
                    c10::optional<torch::Tensor>(torch::empty({((kz * num_offsets - dense_offsets[i]) / 2) * num_targets, 2}, torch::TensorOptions(device).dtype(torch::kInt32))),          //nbmaps
                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets - dense_offsets[i]) / 2 }, torch::TensorOptions(device).dtype(torch::kInt32))),                            //nbsizes
                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets - dense_offsets[i]) / 2 + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //naddrs
                    c10::optional<torch::Tensor>(torch::empty({(kz * num_offsets - dense_offsets[i]) / 2 + 1}, torch::TensorOptions(device).dtype(torch::kInt32))),                         //qnaddrs
                    c10::optional<std::int32_t>(mapsize_val),                                                                                                                //mapsize
                    c10::optional<std::int32_t>(qmapsize_val));                                                                                                              //qmapsize


        auto dummy1 = [] SPIRA_DEVICE(UIter) {
            return false;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, dummy1, d_values_in);

        auto GetKey = [aligned_num_targets] SPIRA_DEVICE(UIter x) {
          return x / aligned_num_targets;
        };

        CUB_TRANSFORMED_INPUT_ITERATOR(UIter, GetKey, d_keys_in);

        cub::DiscardOutputIterator d_unique_out;
        //cub::DiscardOutputIterator dummy_out;
        cub::DiscardOutputIterator d_num_out;

        // Compute temporary storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceReduce::ReduceByKey(
                nullptr,
                d_temp_size_reduce,
                d_keys_in,
                d_unique_out,
                d_values_in,
                std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
                d_num_out,
                cub::Sum(),
                ((kz * num_offsets - dense_offsets[i])/2) * aligned_num_targets 
            )
        );

        auto IsValidEntry = [] SPIRA_DEVICE(const std::int64_t &entry) {
            return entry != -1;
        };


        auto dummy2 = [] SPIRA_DEVICE(auto index) {
            return static_cast<std::int64_t>(0);
        };
    
        CUB_TRANSFORMED_INPUT_ITERATOR(std::int64_t, dummy2, d_in);

        // Compute temp storage size
        SPIRA_CHECK_CUDA(
            cub::DeviceSelect::If(
                nullptr,                          
                d_temp_size_select,                
                d_in,                              
                reinterpret_cast<int64_t*>(std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>()),                             
                d_num_out,                
                ((kz * num_offsets - dense_offsets[i])/2) * aligned_num_targets, 
                IsValidEntry                                           
            )
        ); 

        total_temp_size += std::max(d_temp_size_reduce, d_temp_size_select);
        temp_offsets.emplace_back(total_temp_size);
    }
  }

  auto d_temp_storage = context.NewBuffer(total_temp_size);

  std::size_t current_offsets = 0;

  context.Synchronize();

  for (size_t i = 0; i < num_maps; ++i) {
    const auto num_sources = voxels[i/2].size(0);
    const auto num_targets = voxels[(i+1)/2].size(0);

    auto aligned_num_targets = (num_targets + 127)/128 * 128;

    const uint32_t kx = std::get<0>(kernel_sizes[i]);
    const uint32_t ky = std::get<1>(kernel_sizes[i]);
    const uint32_t kz = std::get<2>(kernel_sizes[i]);

    const uint32_t sz = std::get<2>(source_strides[i]);

    int64_t num_offsets = kx * ky;

    if(dataflow[i] == 0){

      cuda::BinarySearch_Streamed_Os().operator()<CT>(
          num_sources,
          num_targets,
          aligned_num_targets,
          num_offsets,
          TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
          TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]),
          TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                               
          kz,                                                                                                                                                                                                       
          sz,  
          std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
          streams[i].stream());
    }

    else if(dataflow[i] == 1){
        if(i % 2 == 0 && (kz * num_offsets) % 2 == 1){    
            cuda::BinarySearch_Streamed_Ws_Half().operator()<CT>(
                    num_sources,
                    num_targets,
                    aligned_num_targets,
                    num_offsets,
                    TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
                    TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]), 
                    TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                               
                    kz,                                                                                                                                                                                                        
                    sz,  
                    std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
                    reinterpret_cast<int64_t*>(std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>()),
                    std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    std::get<6>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    std::get<7>(kernel_maps[i]).value(),
                    std::get<8>(kernel_maps[i]).value(),
                    d_temp_storage.device_data() + temp_offsets[i],
                    temp_offsets[i+1],
                    sizes[i].value().data_ptr<int32_t>(),
                    streams[i].stream());
        }
        else{    
            cuda::BinarySearch_Streamed_Ws().operator()<CT>(
                    num_sources,
                    num_targets,
                    aligned_num_targets,
                    num_offsets,
                    TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
                    TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]), 
                    TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                                 
                    kz,                                                                                                                                                                                                       
                    sz, 
                    std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
                    reinterpret_cast<int64_t*>(std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>()),
                    std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    std::get<6>(kernel_maps[i]).value().data_ptr<int32_t>(),
                    std::get<7>(kernel_maps[i]).value(),
                    std::get<8>(kernel_maps[i]).value(),
                    d_temp_storage.device_data() + temp_offsets[i],
                    temp_offsets[i+1],
                    sizes[i].value().data_ptr<int32_t>(),
                    streams[i].stream());
        }
    }

    else if(dataflow[i] == 2){
        if(i % 2 == 0 && (kz * num_offsets) % 2 == 1){    
            cuda::BinarySearch_Streamed_GS_Half().operator()<CT>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
            TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]),
            TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                                
            kz,                                                                                                                                                                                                         
            sz, 
            std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
            std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>(),
            std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
            std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),
            metadata_buffers[i].value().first.data_ptr<int64_t>(),
            metadata_buffers[i].value().second.data_ptr<int32_t>(),
            d_temp_storage.device_data() + temp_offsets[i],
            temp_offsets[i+1],
            streams[i].stream());
        }

        else{    
            cuda::BinarySearch_Streamed_GS().operator()<CT>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
            TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]),
            TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                                 
            kz,                                                                                                                                                                                                         
            sz, 
            std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
            std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>(),
            std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
            std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),
            metadata_buffers[i].value().first.data_ptr<int64_t>(),
            metadata_buffers[i].value().second.data_ptr<int32_t>(),
            d_temp_storage.device_data() + temp_offsets[i],
            temp_offsets[i+1],
            streams[i].stream());
        }
    }
    
    else if(dataflow[i] % 2 == 0){
       cuda::BinarySearch_Streamed_Hybrid_GS().operator()<CT>(
            num_sources,
            num_targets,
            aligned_num_targets,
            num_offsets,
            dense_offsets[i],
            TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
            TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]),
            TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                                  
            kz,                                                                                                                                                                                                       
            sz, 
            reinterpret_cast<const uint32_t*>(indexes[i].value().data_ptr<int32_t>()),
            std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
            std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>(),
            std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
            std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),
            metadata_buffers[i].value().first.data_ptr<int64_t>(),
            metadata_buffers[i].value().second.data_ptr<int32_t>(),
            d_temp_storage.device_data() + temp_offsets[i],
            temp_offsets[i+1],
            streams[i].stream());
    }
  
    else{
        cuda::BinarySearch_Streamed_Hybrid_WS().operator()<CT>(
                num_sources,
                num_targets,
                aligned_num_targets,
                num_offsets,
                dense_offsets[i],
                TypeConversion<CT>::GetCppPointer(voxels[i/2]),                                                             
                TypeConversion<CT>::GetCppPointer(voxels[(i+1)/2]),
                TypeConversion<CT>::GetCppPointer(offsets_all) + current_offsets,                                                                                                                                                                                                 
                kz,                                                                                                                                                                                                       
                sz, 
                reinterpret_cast<const uint32_t*>(indexes[i].value().data_ptr<int32_t>()),
                std::get<2>(kernel_maps[i]).data_ptr<int32_t>(),
                reinterpret_cast<int64_t*>(std::get<3>(kernel_maps[i]).value().data_ptr<int32_t>()),
                std::get<4>(kernel_maps[i]).value().data_ptr<int32_t>(),
                std::get<5>(kernel_maps[i]).value().data_ptr<int32_t>(),
                std::get<6>(kernel_maps[i]).value().data_ptr<int32_t>(),
                std::get<7>(kernel_maps[i]).value(),
                std::get<8>(kernel_maps[i]).value(),
                d_temp_storage.device_data() + temp_offsets[i],
                temp_offsets[i+1],
                sizes[i].value().data_ptr<int32_t>(),
                streams[i].stream());        
    }
  
    current_offsets += num_offsets;
}

///////////////////////////////////////////////////////

  for (auto& stream : streams) {
    stream.synchronize();
  }

  for (size_t i = 0; i < num_maps; ++i) {
    if(dataflow[i] % 2 == 1){
        auto& nbmaps = std::get<3>(kernel_maps[i]).value();
        auto mapsize_val = std::get<7>(kernel_maps[i]).value();
        nbmaps = nbmaps.narrow(0, 0, static_cast<int64_t>(mapsize_val)).t().contiguous();
    }

    if(dataflow[i] > 4){
            auto& out_in_map = std::get<2>(kernel_maps[i]);
            out_in_map = out_in_map.narrow(0, 0, static_cast<int64_t>(dense_offsets[i])).t().contiguous();
    }
  }  

  context.Synchronize();

  return kernel_maps;
}

template <typename CT>    
std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>> BinarySearch_No_Z(
        torch::Tensor &sources,
        torch::Tensor &targets,
        const std::tuple<int,int,int> &kernel_size,
        const std::tuple<int,int,int> &source_stride){

    const auto num_sources = sources.size(0);
    const auto num_targets = targets.size(0); 

    auto aligned_num_targets = (num_targets + 127)/128 * 128;        

    auto device = GetTorchDeviceFromTensors({sources});
  
    auto context = GetCUDAContextFromTorchCUDADevice(device);

    context.Synchronize();

    auto offsets = generate_offsets_full<CT>(kernel_size, source_stride).to(device);

    uint32_t kz = std::get<2>(kernel_size);

    int64_t num_offsets = offsets.size(0);

    auto kernel_map = std::make_tuple(num_sources, num_targets,                                                 
        torch::empty({aligned_num_targets, num_offsets},                                                            
        torch::TensorOptions(device).dtype(torch::kInt32)),                                                         
        c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

    cuda::BinarySearch_No_Small_Z().operator()<CT>(                                                                                      
        num_sources,                                                                                            
        num_targets,                                                                                            
        aligned_num_targets,                                                                                    
        num_offsets,
        TypeConversion<CT>::GetCppPointer(sources),                                                             
        TypeConversion<CT>::GetCppPointer(targets),
        TypeConversion<CT>::GetCppPointer(offsets),
        kz,                                                                                                                                                                                                                                                                                                                                                                 
        std::get<2>(kernel_map).data_ptr<int32_t>(),                                                            
        context.stream());
        
    return kernel_map;
}      

#define SPIRA_EXPLICIT_INSTANTIATION(_, CT)                                   \
  template std::tuple<std::int64_t, std::int64_t, torch::Tensor,              \
  c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,                 \
  c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,                 \
  c10::optional<std::int32_t>,                                                \
  c10::optional<std::int32_t>> BinarySearch<CT>(                              \
        torch::Tensor &sources,                                               \
        torch::Tensor &targets,                                               \
        const std::tuple<int,int,int> &kernel_size,                           \
        const std::tuple<int,int,int> &source_stride,                         \
        int dataflow);                                                        \
                                                                              \
  template std::vector<std::tuple<std::int64_t, std::int64_t, torch::Tensor,  \
  c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,                 \
  c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,                 \
  c10::optional<std::int32_t>,                                                \
  c10::optional<std::int32_t>>> BinarySearchStreamed<CT>(                     \
        const std::vector<torch::Tensor> &voxels,                             \
        std::vector<std::tuple<int,int,int>> &kernel_sizes,                   \
        std::vector<std::tuple<int,int,int>> &source_strides,                 \
        std::vector<int> &dataflow);                                          \
                                                                              \
  template std::tuple<std::int64_t, std::int64_t, torch::Tensor,              \
  c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,                 \
  c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,                 \
  c10::optional<std::int32_t>,                                                \
  c10::optional<std::int32_t>> BinarySearch_No_Z<CT>(                         \
        torch::Tensor &sources,                                               \
        torch::Tensor &targets,                                               \
        const std::tuple<int,int,int> &kernel_size,                           \
        const std::tuple<int,int,int> &source_stride)                                       

SPIRA_FOR_ALL_C_TYPES(SPIRA_EXPLICIT_INSTANTIATION);

////////////   Serial    ////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>> CUDABinarySearch(
    torch::Tensor &sources,
    torch::Tensor &targets,
    const std::tuple<int,int,int> &kernel_size,
    const std::tuple<int,int,int> &source_stride,
    int dataflow){

    auto ctype = sources.dtype().toScalarType();

    #define CASE(_, CT)                                                                                                 \
        do {                                                                                                            \
        if (ctype == TypeConversion<CT>::TORCH_DTYPE) {                                                                 \
            return BinarySearch<CT>(                                                                                    \
                sources,                                                                                                \
                targets,                                                                                                \
                kernel_size,                                                                                            \
                source_stride,                                                                                          \
                dataflow);                                                                                              \
        }                                                                                                               \
        } while (false)
        SPIRA_FOR_ALL_C_TYPES(CASE);
    #undef CASE
    SPIRA_ERROR("Cannot find implementation of ", __func__);
}


//////////////////////////////////// Streamed /////////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>>> CUDABinarySearchStreamed(
    const std::vector<torch::Tensor> &voxels,
    std::vector<std::tuple<int,int,int>> &kernel_sizes,
    std::vector<std::tuple<int,int,int>> &source_strides,
    std::vector<int> &dataflow) {

      auto ctype = voxels[0].dtype().toScalarType();

      #define CASE(_, CT)                                                                                               \
        do {                                                                                                            \
        if (ctype == TypeConversion<CT>::TORCH_DTYPE) {                                                                 \
            return BinarySearchStreamed<CT>(                                                                            \
                voxels,                                                                                                 \
                kernel_sizes,                                                                                           \
                source_strides,                                                                                         \
                dataflow);                                                                                              \
        }                                                                                                               \
        } while (false)
        SPIRA_FOR_ALL_C_TYPES(CASE);
    #undef CASE
    SPIRA_ERROR("Cannot find implementation of ", __func__);
}

///////////////////// no small z ////////////////////////////////////////////

std::tuple<std::int64_t, std::int64_t, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>,
c10::optional<torch::Tensor>, c10::optional<std::int32_t>, c10::optional<std::int32_t>> CUDABinarySearchNoSmallZ(
    torch::Tensor &sources,
    torch::Tensor &targets,
    const std::tuple<int,int,int> &kernel_size,
    const std::tuple<int,int,int> &source_stride){

    auto ctype = sources.dtype().toScalarType();

    #define CASE(_, CT)                                                                                             \
     do {                                                                                                           \
      if (ctype == TypeConversion<CT>::TORCH_DTYPE) {                                                               \
        return BinarySearch_No_Z<CT>(                                                                               \
            sources,                                                                                                 \
            targets,                                                                                                \
            kernel_size,                                                                                            \
            source_stride);                                                                                         \
      }                                                                                                             \
    } while (false)
    SPIRA_FOR_ALL_C_TYPES(CASE);
    #undef CASE
    SPIRA_ERROR("Cannot find implementation of ", __func__);
}

//////////////////////////// Helper Convert Tranposed //////////////////////

torch::Tensor CUDAConvertTransposedOs(
    torch::Tensor &out_in_map,
    std::size_t num_sources
){
    auto aligned_num_sources = (num_sources + 127)/128 * 128;

    auto device = GetTorchDeviceFromTensors({out_in_map});

    auto aligned_num_targets = out_in_map.size(0);

    auto num_offsets = out_in_map.size(1);

    auto out_in_map_t = torch::full({static_cast<int64_t>(aligned_num_sources), static_cast<int64_t>(num_offsets)}, -1, torch::TensorOptions(device).dtype(torch::kInt32));

    cuda::Convert_Transposed_Os()(
        reinterpret_cast<const int32_t*>(out_in_map.data_ptr<int32_t>()),
        out_in_map_t.data_ptr<int32_t>(),
        aligned_num_targets,
        num_offsets);

    return out_in_map_t;
}

SPIRA_TORCH_REGISTER(cuda_binary_search_streamed, CUDABinarySearchStreamed);

SPIRA_TORCH_REGISTER(cuda_binary_search, CUDABinarySearch);

SPIRA_TORCH_REGISTER(cuda_convert_transposed_os, CUDAConvertTransposedOs);

SPIRA_TORCH_REGISTER(cuda_binary_search_no_small_z , CUDABinarySearchNoSmallZ);

}  // namespace spira
