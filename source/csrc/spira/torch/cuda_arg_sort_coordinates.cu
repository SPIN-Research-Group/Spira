#include "spira/common/exception.h"
#include "spira/cuda/functions/arg_sort_coodrinates.cuh"
#include "spira/enabled_arguments.h"
#include "spira/torch/cuda_common.cuh"
#include "spira/cuda/helpers.cuh"
#include "spira/compress_layout.h"


namespace spira {

std::pair<torch::Tensor, torch::Tensor> CUDAFlattenSort(const torch::Tensor &coordinates, bool is_double) {
  SPIRA_ENSURE_TENSOR_NDIM(coordinates, 2);

  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto n = coordinates.size(0);
  auto indices = torch::empty({n}, torch::TensorOptions(device).dtype(torch::kInt32));

  if(is_double){
    auto sorted_compressed_coords = torch::empty({n}, torch::TensorOptions(device).dtype(torch::kInt64));

    cuda::FlattenSort().operator()<uint64_t>(                                                                           
            n, coordinates.data_ptr<std::int32_t>(), indices.data_ptr<std::int32_t>(),                                                                
            reinterpret_cast<std::uint64_t*>(sorted_compressed_coords.data_ptr<std::int64_t>()),                                  
            context);
    return {sorted_compressed_coords, indices};   
  }
  
  else{
    auto sorted_compressed_coords = torch::empty({n}, torch::TensorOptions(device).dtype(torch::kInt32));
                                                                                                                      
    cuda::FlattenSort().operator()<uint32_t>(                                                                           
            n, coordinates.data_ptr<std::int32_t>(), indices.data_ptr<std::int32_t>(),                                                                
            reinterpret_cast<std::uint32_t*>(sorted_compressed_coords.data_ptr<std::int32_t>()),                                  
            context);
    return {sorted_compressed_coords, indices};   
  }   
                                                                                                                                                                                                            
}

torch::Tensor CUDAUnFlatten(const torch::Tensor &coordinates) {
  SPIRA_ENSURE_TENSOR_NDIM(coordinates, 1);

  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto n = coordinates.size(0);

  auto ctype = coordinates.dtype().toScalarType();

  auto uncompressed_coords = torch::empty({n, 3}, torch::TensorOptions(device).dtype(torch::kInt32));

  #define CASE(_, CT)                                                                                   \
    do {                                                                                                \
      if (ctype == TypeConversion<CT>::TORCH_DTYPE) {                                                   \
          cuda::UnFlatten().operator()<CT>(static_cast<int>(n),                                        \
          TypeConversion<CT>::GetCppPointer(coordinates),                                               \
          uncompressed_coords.data_ptr<std::int32_t>(),                                                 \
          context);                                                                                     \
          return uncompressed_coords;                                                                   \
      }                                                                                                 \
    } while (false)
    SPIRA_FOR_ALL_C_TYPES(CASE);
  #undef CASE
    SPIRA_ERROR("Cannot find implementation of ", __func__);
  }
  

torch::Tensor CUDADownsample(const torch::Tensor &coordinates, const int stride_x, const int stride_y, const int stride_z) {
  
  SPIRA_ENSURE_TENSOR_NDIM(coordinates, 1);   //coordinates should be already flattened

  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto n = coordinates.size(0);

  auto ctype = coordinates.dtype().toScalarType();
  auto targets = torch::empty({n}, torch::TensorOptions(device).dtype(ctype));

  // Allocate device memory to store number of uniques
  auto d_num_uniques_buf = context.NewBuffer(sizeof(std::uint32_t));
  auto d_num_uniques_ptr = static_cast<std::uint32_t*>(d_num_uniques_buf.device_data());

  #define CASE(_, CT)                                                                                   \
    do {                                                                                                \
      if (ctype == TypeConversion<CT>::TORCH_DTYPE) {                                                   \
          const CT x_mask = Layout<CT>::X_ONES &  ~(static_cast<CT>(stride_x - 1u));                    \
          const CT y_mask = Layout<CT>::Y_ONES &  ~(static_cast<CT>(stride_y - 1u));                    \
          const CT z_mask = Layout<CT>::Z_ONES &  ~(static_cast<CT>(stride_z - 1u));                    \
          const std::make_signed_t<CT> mask = (x_mask << Layout<CT>::X_SHIFT) | (y_mask << Layout<CT>::Y_SHIFT) | z_mask;   \
          auto mask_tensor = torch::full({}, mask, torch::TensorOptions(device).dtype(ctype));          \
          auto sampled_coordinates = coordinates & mask_tensor;                                         \
          cuda::SortUnique().operator()<CT>(static_cast<int>(n),                                        \
          TypeConversion<CT>::GetCppPointer(sampled_coordinates),                                       \
          TypeConversion<CT>::GetCppPointer(targets),                                                   \
          d_num_uniques_ptr, context);                                                                  \
          std::uint32_t num_uniques;                                                                    \
          num_uniques =  context.ReadDeviceData(d_num_uniques_ptr);                                     \
          return targets.slice(0, 0, static_cast<int64_t>(num_uniques));                                \
          return targets;                                                                                 \
      }                                                                                                 \
    } while (false)
    SPIRA_FOR_ALL_C_TYPES(CASE);
  #undef CASE
    SPIRA_ERROR("Cannot find implementation of ", __func__);
}

std::vector<torch::Tensor> CUDAStreamedOutputGenerate(const torch::Tensor &coordinates, const std::vector<std::tuple<int, int, int>> &target_strides) {

  SPIRA_ENSURE_TENSOR_NDIM(coordinates, 1);
  auto device = GetTorchDeviceFromTensors({coordinates});
  auto context = GetCUDAContextFromTorchCUDADevice(device);

  context.Synchronize();

  auto n = coordinates.size(0);

  auto ctype = coordinates.dtype().toScalarType();

  const size_t batch = target_strides.size();

  auto d_num_uniques_buf = context.NewBuffer(sizeof(uint32_t) * batch);
  auto d_num_uniques_ptr = static_cast<uint32_t*>(d_num_uniques_buf.device_data());
  
  auto masks_cpu_tensor = torch::empty({(int64_t)batch}, torch::TensorOptions().dtype(ctype).device(torch::kCPU));

  std::vector<torch::Tensor> outputs;
  outputs.reserve(batch);

  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(batch);

  for (size_t i = 0; i < batch; ++i) {
    streams.emplace_back(c10::cuda::getStreamFromPool(false, device.index()));
    outputs.emplace_back(torch::empty({n}, torch::TensorOptions(device).dtype(ctype)));
  }

  size_t temp_bytes_sort, temp_bytes_unique;

  #define CASE(_, CT)                                                                                   \
    do {                                                                                                \
      if (ctype == TypeConversion<CT>::TORCH_DTYPE) {                                                   \
          std::make_signed_t<CT> *masks_ptr = masks_cpu_tensor.data_ptr<std::make_signed_t<CT>>();       \
          for (size_t i = 0; i < batch; ++i) {                                                          \
            auto [stride_x, stride_y, stride_z] = target_strides[i];                                    \
            const CT x_mask = Layout<CT>::X_ONES &  ~(static_cast<CT>(stride_x - 1u));                  \
            const CT y_mask = Layout<CT>::Y_ONES &  ~(static_cast<CT>(stride_y - 1u));                  \
            const CT z_mask = Layout<CT>::Z_ONES &  ~(static_cast<CT>(stride_z - 1u));                  \
            const std::make_signed_t<CT> mask = (x_mask << Layout<CT>::X_SHIFT) | (y_mask << Layout<CT>::Y_SHIFT) | z_mask; \
            masks_ptr[i] = mask;                                                                        \
          }                                                                                             \
          auto masks_tensor = masks_cpu_tensor.to(device);                                              \
                                                                                                        \
          /*Bitwise AND with broadcasting -> [batch, N] */                                              \
          auto sampled_coords = (coordinates.unsqueeze(0) & masks_tensor.unsqueeze(1)).contiguous();    \
                                                                                                        \
          cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes_sort,                                      \
           (const CT*)nullptr, (CT*)nullptr, n, 0, sizeof(CT) * CHAR_BIT);                              \
                                                                                                        \
          cub::DeviceSelect::Unique(nullptr, temp_bytes_unique,                                         \
          (CT*)nullptr, (CT*)nullptr, (uint32_t*)nullptr, n);                                           \
                                                                                                        \
          size_t temp_bytes = std::max(temp_bytes_sort, temp_bytes_unique);                             \
                                                                                                        \
          auto d_temp_sort_storage = context.NewBuffer(temp_bytes * batch);                             \
                                                                                                        \
          context.Synchronize();                                                                        \
                                                                                                        \
          for (size_t i = 0; i < batch; ++i) {                                                          \
              cuda::SortStreamed().operator()<CT>(                                                      \
              n,                                                                                        \
              TypeConversion<CT>::GetCppPointer(sampled_coords[i]),                                     \
              TypeConversion<CT>::GetCppPointer(outputs[i]),                                            \
              streams[i].stream(),                                                                      \
              d_temp_sort_storage.device_data() + i * temp_bytes,                                       \
              temp_bytes_sort,                                                                          \
              temp_bytes_unique,                                                                        \
              d_num_uniques_ptr + i                                                                     \
            );                                                                                          \
          }                                                                                             \
          for (auto& stream : streams) {                                                                \
              stream.synchronize();                                                                     \
          }                                                                                             \
                                                                                                        \
          std::vector<uint32_t> host_counts(batch);                                                     \
          context.ReadDeviceData(d_num_uniques_ptr, host_counts.data(), batch);                         \
          std::vector<torch::Tensor> final_slices;                                                      \
          final_slices.reserve(batch);                                                                  \
          for (size_t i = 0; i < batch; ++i) {                                                          \
            final_slices.push_back(outputs[i].slice(0, 0, static_cast<int64_t>(host_counts[i])));       \
          }                                                                                             \
                                                                                                        \
          return final_slices;                                                                          \
      }                                                                                                 \
    } while (false)
    SPIRA_FOR_ALL_C_TYPES(CASE);
  #undef CASE
    SPIRA_ERROR("Cannot find implementation of ", __func__);

  ///////////////////////////////////////////////////////////////////////////////////////
}

SPIRA_TORCH_REGISTER(cuda_flatten_sort, CUDAFlattenSort);

SPIRA_TORCH_REGISTER(cuda_unflatten, CUDAUnFlatten);

SPIRA_TORCH_REGISTER(cuda_downsample, CUDADownsample);

SPIRA_TORCH_REGISTER(cuda_streamed_output_generate, CUDAStreamedOutputGenerate);

}  // namespace spira
