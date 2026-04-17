#include "spira/cuda/functions/sparse_convolution.cuh"
#include "spira/enabled_arguments.h"
#include "spira/torch/cuda_common.cuh"

namespace spira {

auto CUDATimeGEMM(std::size_t parallel,                   
                  const std::optional<double> &threshold,
                  const bool is_hybrid,  
                  const torch::Tensor &weights,            
                  const torch::Tensor &source_masks,       
                  const torch::Tensor &target_masks,       
                  const torch::Tensor &kernel_map_sizes) {
                    
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);

  SPIRA_ENSURE_TENSOR_NDIM(source_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(target_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(kernel_map_sizes, 1);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);

  auto device = GetTorchDeviceFromTensors(
      {weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();

#define CASE(_, FT)                                                        \
  do {                                                                     \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE ) {                       \
      return cuda::TimeGEMM().operator()<FT>(                              \
          num_sources, num_targets, num_offsets, is_hybrid,                \
          threshold, parallel,                                             \
          num_source_features, num_target_features,                        \
          kernel_map_sizes.data_ptr<int32_t>(),                            \
          TypeConversion<FT>::GetCppPointer(weights), context);            \
    }                                                                      \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}

auto CUDATimeGather(std::size_t tile_size,                   
                    const std::optional<double> &threshold,  
                    const bool is_hybrid,
                    const torch::Tensor &weights,            
                    const torch::Tensor &source_masks,       
                    const torch::Tensor &target_masks,       
                    const torch::Tensor &kernel_map_sizes) {
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);
  SPIRA_ENSURE_TENSOR_NDIM(source_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(target_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(kernel_map_sizes, 1);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);

  auto device = GetTorchDeviceFromTensors(
      {weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
#define CASE(_, FT)                                                                  \
  do {                                                                               \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                                  \
      return cuda::TimeGather().operator()<FT>(                                      \
          num_sources, num_targets, num_offsets, is_hybrid, threshold,               \
          num_source_features, num_target_features,                                  \
          source_masks.data_ptr<int32_t>(), kernel_map_sizes.data_ptr<int32_t>(),    \
          tile_size, context);                                                       \
    }                                                                                \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}

auto CUDATimeScatter(std::size_t tile_size,                   
                     const std::optional<double> &threshold,  
                     const bool is_hybrid,
                     const torch::Tensor &weights,            
                     const torch::Tensor &source_masks,       
                     const torch::Tensor &target_masks,       
                     const torch::Tensor &kernel_map_sizes) {
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);
  SPIRA_ENSURE_TENSOR_NDIM(source_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(target_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(kernel_map_sizes, 1);

  auto num_sources = source_masks.size(1);
  auto num_targets = target_masks.size(1);
  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);

  auto device = GetTorchDeviceFromTensors(
      {weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
#define CASE(_, FT)                                                                    \
  do {                                                                                 \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                                    \
      return cuda::TimeScatter().operator()<FT>(                                       \
          num_sources, num_targets, num_offsets,                                       \
          is_hybrid, threshold,                                                        \
          num_source_features, num_target_features,                                    \
          target_masks.data_ptr<int32_t>(), kernel_map_sizes.data_ptr<int32_t>(),      \
          tile_size, context);                                                         \
    }                                                                                  \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}


torch::Tensor CUDASparseConvolutionForward_WS(
    std::size_t num_sources,                 
    std::size_t num_targets,
    std::size_t gather_tile_size,                          
    std::size_t scatter_tile_size,                         
    std::size_t parallel,                                  
    const std::optional<double> &threshold,                
    const torch::Tensor &sources,                          
    const torch::Tensor &weights,                          
    const torch::Tensor &source_masks,                     
    const torch::Tensor &target_masks,                     
    const torch::Tensor &kernel_map_sizes) {

  SPIRA_ENSURE_TENSOR_NDIM(source_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(target_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(sources, 2);
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);

  auto num_offsets = target_masks.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets);
  SPIRA_ENSURE_TENSOR_DIM(weights, 0, num_offsets);
  SPIRA_ENSURE_TENSOR_DIM(sources, 1, num_source_features);

  auto device = GetTorchDeviceFromTensors(
      {sources, weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();

  auto targets = torch::empty({static_cast<int64_t>(num_targets), num_target_features},
                              torch::TensorOptions(device).dtype(ftype));

#define CASE(_, FT)                                                          \
  do {                                                                       \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                          \
        cuda::SparseConvolutionForward_WS().                                 \
        operator()<FT>(                                                      \
            num_sources, num_targets, num_offsets, threshold, parallel,      \
            num_source_features, num_target_features,                        \
            TypeConversion<int32_t>::GetCppPointer(source_masks),            \
            TypeConversion<int32_t>::GetCppPointer(target_masks),            \
            TypeConversion<int32_t>::GetCppPointer(kernel_map_sizes),        \
            TypeConversion<FT>::GetCppPointer(sources),                      \
            TypeConversion<FT>::GetCppPointer(weights),                      \
            TypeConversion<FT>::GetCppPointer(targets), gather_tile_size,    \
            scatter_tile_size, context);                                     \
      return targets;                                                        \
    }                                                                        \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}

torch::Tensor CUDASparseConvolutionForward_WS_Merged(
    std::size_t num_sources,                 
    std::size_t num_targets,                              
    const torch::Tensor &sources,                          
    const torch::Tensor &weights,
    const torch::Tensor &naddrs,
    const torch::Tensor &qnaddrs,
    const torch::Tensor &nbmaps,
    int32_t mapsize,
    int32_t qmapsize,
    bool transpose) {

  SPIRA_ENSURE_TENSOR_NDIM(sources, 2);
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);

  auto num_offsets = weights.size(0);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(sources, 1, num_source_features);

  auto device = GetTorchDeviceFromTensors(
      {sources, weights, naddrs, qnaddrs, nbmaps});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();

  auto targets = torch::zeros({static_cast<int64_t>(num_targets), num_target_features},
                              torch::TensorOptions(device).dtype(ftype));

  auto base = TypeConversion<int32_t>::GetCppPointer(nbmaps);

  int32_t* imap;
  int32_t* omap;

  if (transpose) {
      imap = base + mapsize;
      omap = base;
  } else {
      imap = base;
      omap = base + mapsize;
  }


#define CASE(_, FT)                                                          \
  do {                                                                       \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                          \
        cuda::SparseConvolutionForward_WS_Merged().                          \
        operator()<FT>(                                                      \
            num_sources, num_targets, num_offsets,                           \
            num_source_features, num_target_features,                        \
            qmapsize, false,                                                 \
            imap,                                                            \
            omap,                                                            \
            TypeConversion<int32_t>::GetCppPointer(naddrs),                  \
            TypeConversion<int32_t>::GetCppPointer(qnaddrs),                 \
            TypeConversion<FT>::GetCppPointer(sources),                      \
            TypeConversion<FT>::GetCppPointer(weights),                      \
            TypeConversion<FT>::GetCppPointer(targets),                      \
            context.stream());                                               \
      return targets;                                                        \
    }                                                                        \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}


torch::Tensor CUDASparseConvolutionForward_HS(
    std::size_t num_sources,                 
    std::size_t num_targets,
    std::size_t gather_tile_size,                          
    std::size_t scatter_tile_size,                         
    std::size_t parallel,                               
    const std::optional<double> &threshold,                
    const torch::Tensor &sources,                          
    const torch::Tensor &weights,                          
    const torch::Tensor &source_masks,                     
    const torch::Tensor &target_masks,                     
    const torch::Tensor &kernel_map_sizes,
    const std::optional<torch::Tensor> &out_in_map) {

  SPIRA_ENSURE_TENSOR_NDIM(source_masks, 2);
  SPIRA_ENSURE_TENSOR_NDIM(target_masks, 2);

  SPIRA_ENSURE_TENSOR_NDIM(sources, 2);
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);

  std::size_t num_offsets_os;

  if (out_in_map.has_value()) {
     SPIRA_ENSURE_TENSOR_NDIM(out_in_map.value(), 2);
     num_offsets_os = out_in_map.value().size(1);
  }

  else{
    num_offsets_os = 1;           //specialized case
  }

  auto num_offsets_ws = target_masks.size(0);

  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(source_masks, 0, num_offsets_ws);
  SPIRA_ENSURE_TENSOR_DIM(weights, 0, (num_offsets_ws + num_offsets_os));
  SPIRA_ENSURE_TENSOR_DIM(sources, 1, num_source_features);

  auto device = GetTorchDeviceFromTensors(
      {sources, weights, source_masks, target_masks, kernel_map_sizes});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
  auto targets_ws = torch::empty({static_cast<int64_t>(num_targets), num_target_features},
                              torch::TensorOptions(device).dtype(ftype));

  auto targets_os = torch::empty({static_cast<int64_t>(num_targets), num_target_features},
                              torch::TensorOptions(device).dtype(ftype));

#define CASE(_, FT)                                                                                                  \
  do {                                                                                                               \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                                                                  \
      if (out_in_map.has_value()) {                                                                                  \
        cuda::SparseConvolutionForward_HS().                                                                         \
        operator()<FT>(                                                                                              \
            num_sources, num_targets, num_offsets_ws, num_offsets_os, threshold, parallel,                           \
            num_source_features, num_target_features,                                                                \
            TypeConversion<int32_t>::GetCppPointer(source_masks),                                                    \
            TypeConversion<int32_t>::GetCppPointer(target_masks),                                                    \
            TypeConversion<int32_t>::GetCppPointer(kernel_map_sizes),                                                \
            TypeConversion<int32_t>::GetCppPointer(out_in_map.value()),                                              \
            TypeConversion<FT>::GetCppPointer(sources),                                                              \
            TypeConversion<FT>::GetCppPointer(weights) + num_offsets_os * num_source_features * num_target_features, \
            TypeConversion<FT>::GetCppPointer(weights),                                                              \
            TypeConversion<FT>::GetCppPointer(targets_ws), TypeConversion<FT>::GetCppPointer(targets_os), gather_tile_size,  \
            scatter_tile_size, context);                                                                             \
    } else{                                                                                                          \
 cuda::SparseConvolutionForward_HS().                                                                                \
        operator()<FT>(                                                                                              \
            num_sources, num_targets, num_offsets_ws, num_offsets_os, threshold, parallel,                           \
            num_source_features, num_target_features,                                                                \
            TypeConversion<int32_t>::GetCppPointer(source_masks),                                                    \
            TypeConversion<int32_t>::GetCppPointer(target_masks),                                                    \
            TypeConversion<int32_t>::GetCppPointer(kernel_map_sizes),                                                \
            nullptr,                                                                                                 \
            TypeConversion<FT>::GetCppPointer(sources),                                                              \
            TypeConversion<FT>::GetCppPointer(weights) + num_offsets_os * num_source_features * num_target_features, \
            TypeConversion<FT>::GetCppPointer(weights),                                                              \
            TypeConversion<FT>::GetCppPointer(targets_ws), TypeConversion<FT>::GetCppPointer(targets_os), gather_tile_size,  \
            scatter_tile_size, context);                                                                             \
    }                                                                                                                \
    targets_ws += targets_os;                                                                                           \
    return targets_ws;                                                                                                  \
  }                                                                                                                  \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}


torch::Tensor CUDASparseConvolutionForward_HS_Merged(
    std::size_t num_sources,                 
    std::size_t num_targets,                              
    const torch::Tensor &sources,                          
    const torch::Tensor &weights,
    const torch::Tensor &naddrs,
    const torch::Tensor &qnaddrs,
    const torch::Tensor &nbmaps,
    const std::optional<torch::Tensor> &out_in_map,
    int32_t mapsize,
    int32_t qmapsize,
    bool transpose) {

  SPIRA_ENSURE_TENSOR_NDIM(sources, 2);
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);

  auto num_offsets_ws = (naddrs.size(0) - 1) * 2;

  std::size_t num_offsets_os;

  if (out_in_map.has_value()) {
     SPIRA_ENSURE_TENSOR_NDIM(out_in_map.value(), 2);
     num_offsets_os = out_in_map.value().size(1);
  }

  else{
    num_offsets_os = 1;           //specialized case
  }

  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(sources, 1, num_source_features);
  SPIRA_ENSURE_TENSOR_DIM(weights, 0, (num_offsets_ws + num_offsets_os));

  auto device = GetTorchDeviceFromTensors(
      {sources, weights, naddrs, qnaddrs, nbmaps});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();

  auto targets = torch::empty({static_cast<int64_t>(num_targets), num_target_features},
                              torch::TensorOptions(device).dtype(ftype));                         
                          
  auto base = TypeConversion<int32_t>::GetCppPointer(nbmaps);

  int32_t* imap;
  int32_t* omap;

  if (transpose) {
      imap = base + mapsize;
      omap = base;
  } else {
      imap = base;
      omap = base + mapsize;
  }

#define CASE(_, FT)                                                                                                   \
  do {                                                                                                                \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                                                                   \
      if (out_in_map.has_value()) {                                                                                   \
        cuda::SparseConvolutionForward_HS_Merged().                                                                   \
        operator()<FT>(                                                                                               \
            num_sources, num_targets, num_offsets_ws, num_offsets_os,                                                 \
            num_source_features, num_target_features,                                                                 \
            qmapsize,                                                                                                 \
            imap,                                                                                                     \
            omap,                                                                                                     \
            TypeConversion<int32_t>::GetCppPointer(naddrs),                                                           \
            TypeConversion<int32_t>::GetCppPointer(qnaddrs),                                                          \
            TypeConversion<int32_t>::GetCppPointer(out_in_map.value()),                                               \
            TypeConversion<FT>::GetCppPointer(sources),                                                               \
            TypeConversion<FT>::GetCppPointer(weights) + num_offsets_os * num_source_features * num_target_features,  \
            TypeConversion<FT>::GetCppPointer(weights),                                                               \
            TypeConversion<FT>::GetCppPointer(targets),                                                               \
            context);                                                                                                 \
        }                                                                                                             \
        else{                                                                                                         \
          cuda::SparseConvolutionForward_HS_Merged().                                                                 \
            operator()<FT>(                                                                                           \
            num_sources, num_targets, num_offsets_ws, num_offsets_os,                                                 \
            num_source_features, num_target_features,                                                                 \
            qmapsize,                                                                                                 \
            imap,                                                                                                     \
            omap,                                                                                                     \
            TypeConversion<int32_t>::GetCppPointer(naddrs),                                                           \
            TypeConversion<int32_t>::GetCppPointer(qnaddrs),                                                          \
            nullptr,                                                                                                  \
            TypeConversion<FT>::GetCppPointer(sources),                                                               \
            TypeConversion<FT>::GetCppPointer(weights) + num_offsets_os * num_source_features * num_target_features,  \
            TypeConversion<FT>::GetCppPointer(weights),                                                               \
            TypeConversion<FT>::GetCppPointer(targets),                                                               \
            context);                                                                                                 \
        }                                                                                                             \
                                                                                                                      \
        return targets;                                                                                               \
    }                                                                                                                 \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}

torch::Tensor CUDASparseConvolutionForward_OS(
    std::size_t num_sources,                 
    std::size_t num_targets,
    const torch::Tensor &out_in_map,                     
    const torch::Tensor &sources,                          
    const torch::Tensor &weights) {

  SPIRA_ENSURE_TENSOR_NDIM(sources, 2);
  SPIRA_ENSURE_TENSOR_NDIM(out_in_map, 2);
  SPIRA_ENSURE_TENSOR_NDIM(weights, 3);

  auto num_offsets = out_in_map.size(1);
  auto num_source_features = weights.size(1);
  auto num_target_features = weights.size(2);

  SPIRA_ENSURE_TENSOR_DIM(weights, 0, num_offsets);
  SPIRA_ENSURE_TENSOR_DIM(sources, 1, num_source_features);

  auto device = GetTorchDeviceFromTensors(
      {sources, weights, out_in_map});
  auto context = GetCUDAContextFromTorchCUDADevice(device);
  auto ftype = weights.dtype().toScalarType();
  auto targets = torch::empty({static_cast<int64_t>(num_targets), num_target_features},
                              torch::TensorOptions(device).dtype(ftype));

#define CASE(_, FT)                                                          \
  do {                                                                       \
    if (ftype == TypeConversion<FT>::TORCH_DTYPE) {                          \
        cuda::SparseConvolutionForward_OS().                                 \
        operator()<FT>(                                                      \
            num_sources, num_targets, num_offsets,                           \
            num_source_features, num_target_features,                        \
            TypeConversion<int32_t>::GetCppPointer(out_in_map),              \
            TypeConversion<FT>::GetCppPointer(sources),                      \
            TypeConversion<FT>::GetCppPointer(weights),                      \
            TypeConversion<FT>::GetCppPointer(targets),                      \
            context, context.stream());                                      \
      return targets;                                                        \
    }                                                                        \
  } while (false)
  SPIRA_FOR_ALL_F_TYPES(CASE);
#undef CASE
  SPIRA_ERROR("Cannot find implementation of ", __func__);
}








SPIRA_TORCH_REGISTER(cuda_sparse_convolution_forward_ws, CUDASparseConvolutionForward_WS);
SPIRA_TORCH_REGISTER(cuda_sparse_convolution_forward_os, CUDASparseConvolutionForward_OS);
SPIRA_TORCH_REGISTER(cuda_sparse_convolution_forward_hs, CUDASparseConvolutionForward_HS);
SPIRA_TORCH_REGISTER(cuda_sparse_convolution_forward_ws_merged, CUDASparseConvolutionForward_WS_Merged);
SPIRA_TORCH_REGISTER(cuda_sparse_convolution_forward_hs_merged, CUDASparseConvolutionForward_HS_Merged);
SPIRA_TORCH_REGISTER(cuda_time_gather, CUDATimeGather);
SPIRA_TORCH_REGISTER(cuda_time_scatter, CUDATimeScatter);
SPIRA_TORCH_REGISTER(cuda_time_gemm, CUDATimeGEMM);

}  // namespace spira