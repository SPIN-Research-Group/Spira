#include <iomanip>
#include "spira/cuda/event.cuh"
#include "spira/cuda/functions/sparse_convolution.cuh"
#include "spira/cuda/helpers.cuh"
#include "spira/cuda/kernels/padded_gather_warp_optimized.cuh"
#include "spira/cuda/kernels/padded_scatter_warp_optimized.cuh"
#include "spira/cuda/stream.cuh"
#include "spira/enabled_arguments.h"
#include "spira/cuda/kernels/output_stat.cuh"
#include "spira/cuda/kernels/weight_stat.cuh"

namespace spira::cuda {

std::vector<Stream> stream_pool_;

void dispatcher_weight_stationary(
    int32_t num_targets,
    int32_t k_vol,
    bool is_mirror,
    bool is_hybrid,
    int32_t qsum_nnz,
    const float *in_feats,
    const float *kernel_feats,
    float *out_feats,
    const int32_t *kpos_ptr,
    const int32_t *qkpos_ptr,
    const int32_t *in_map_ptr,
    const int32_t *out_map_ptr,
    int32_t in_channel,
    int32_t out_channel, 
    cudaStream_t stream)
{
  
  int32_t true_qsum_nnz; 

  if(is_mirror) true_qsum_nnz = 2 * qsum_nnz - ((num_targets + 127) / 128) * 128;

  else if(is_hybrid) true_qsum_nnz = 2 * qsum_nnz;

  else true_qsum_nnz = qsum_nnz;

    if(in_channel % 4 == 0 && out_channel % 4 ==0){
      if (in_channel <= 16 && out_channel <= 16){
        auto tiles_y = DivCeil(true_qsum_nnz, 64);  

        kernels::fetch_on_demand_gemm_fp32_once<16, 4, 8>                                                         
                    <<<dim3(DivCeil(out_channel, 16), min(tiles_y, 65535), 1), dim3(4, 16, 1), 0, stream>>>(
                    kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 64), in_channel, out_channel, 
                    in_feats,
                    kernel_feats,
                    out_feats,
                    in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));
      }
      else{
          auto tiles_y = DivCeil(true_qsum_nnz, 128);  

          kernels::fetch_on_demand_gemm_fp32<32, 4, 8>                                                          
                  <<<dim3(DivCeil(out_channel, 32), min(tiles_y, 65535), 1), dim3(8, 32, 1), 0, stream>>>(
                  kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 128), in_channel, out_channel, 
                  in_feats,
                  kernel_feats,
                  out_feats,
                  in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));
      }
    }
    else if (in_channel % 2 == 0 && out_channel % 2 == 0){

        auto tiles_y = DivCeil(true_qsum_nnz, 128);  

        kernels::fetch_on_demand_gemm_fp32_2<16, 8, 8>                                                                     
                    <<<dim3(DivCeil(out_channel, 16), min(tiles_y, 65535), 1), dim3(8, 16, 1), 0, stream>>>(
                    kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 128), in_channel, out_channel, 
                    in_feats,
                    kernel_feats,
                    out_feats,
                    in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));
    }
    else{

        auto tiles_y = DivCeil(true_qsum_nnz, 64);  

        kernels::fetch_on_demand_gemm_fp32_1<16, 4, 8>                                                               
                    <<<dim3(DivCeil(out_channel, 16), min(tiles_y, 65535), 1), dim3(16, 16, 1), 0, stream>>>(
                    kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 64), in_channel, out_channel, 
                    in_feats,
                    kernel_feats,
                    out_feats,
                    in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));
    }
}

void dispatcher_weight_stationary_half(
    int32_t num_targets,
    int32_t k_vol,
    bool is_mirror,
    bool is_hybrid,
    int32_t qsum_nnz,
    const half *in_feats,
    const half *kernel_feats,
    half *out_feats,
    const int32_t *kpos_ptr,
    const int32_t *qkpos_ptr,
    const int32_t *in_map_ptr,
    const int32_t *out_map_ptr,
    int32_t in_channel,
    int32_t out_channel, 
    cudaStream_t stream)
{

int32_t true_qsum_nnz; 

if(is_mirror) true_qsum_nnz = 2 * qsum_nnz - ((num_targets + 127) / 128) * 128;

else if(is_hybrid) true_qsum_nnz = 2 * qsum_nnz;

else true_qsum_nnz = qsum_nnz;

if (in_channel % 4 == 0 && out_channel % 4 == 0){    
    if (in_channel <= 16 || out_channel <= 16){

      auto tiles_y = DivCeil(true_qsum_nnz, 64); 

      kernels::fetch_on_demand_gemm_fp16_4_once<16, 4, 8>                                                    
                  <<<dim3(DivCeil(out_channel, 16), min(tiles_y, 65535), 1), dim3(4, 16, 1), 0, stream>>>(
                  kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 64), in_channel, out_channel, 
                  in_feats,
                  kernel_feats,
                  out_feats,
                  in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));
    }
    else{
        auto tiles_y = DivCeil(true_qsum_nnz, 128); 

        kernels::fetch_on_demand_gemm_fp16_tc4<32, 4, 8, 16, 16, 16, 4, 2, 2>                                             
                  <<<dim3(DivCeil(out_channel, 32), min(tiles_y, 65535), 1), dim3(8, 32, 1), 0, stream>>>(
                  kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 128), in_channel, out_channel, 
                  in_feats,
                  kernel_feats,
                  out_feats,
                  in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));
    }
}
else if (in_channel % 2 == 0 && out_channel % 2 == 0){

    auto tiles_y = DivCeil(true_qsum_nnz, 128); 
  
    kernels::fetch_on_demand_gemm_fp16_2<16, 8, 8>
                    <<<dim3(DivCeil(out_channel, 16), min(tiles_y, 65535), 1), dim3(8, 16, 1), 0, stream>>>(
                    kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 128), in_channel, out_channel, 
                    in_feats,
                    kernel_feats,
                    out_feats,
                    in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));   
}
else{
        auto tiles_y = DivCeil(true_qsum_nnz, 64); 

        kernels::fetch_on_demand_gemm_fp16_1<16, 4, 8>                                                                     
                    <<<dim3(DivCeil(out_channel, 16), min(tiles_y, 65535), 1), dim3(16, 16, 1), 0, stream>>>(
                    kpos_ptr, qkpos_ptr, k_vol, DivCeil(qsum_nnz, 64), in_channel, out_channel, 
                    in_feats,
                    kernel_feats,
                    out_feats,
                    in_map_ptr, out_map_ptr, tiles_y, (is_mirror || is_hybrid));  
}

}

void dispatcher_output_stationary(          
    std::size_t num_out_feats,                 
    std::size_t kernel_volume,                 
    const float *in_feats,
    const float *kernel,
    float *out_feats,
    const int32_t *out_in_map,
    std::size_t num_in_channels,
    std::size_t num_out_channels, 
    cudaStream_t stream)
{
    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int block_num_M = (num_out_feats + 127) / 128;
      int block_num_N = num_out_channels / 64;  //j_factors1
      dim3 num_blocks(block_num_M * block_num_N); 
      dim3 threads_per_block(128);
      kernels::conv_forward_cuda_setting3_mode0_f32f32f32<<<num_blocks, threads_per_block, 0, stream>>>(
          num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else if (num_in_channels % 32 == 0 && num_out_channels % 16 == 0)
    {
      int block_num_M = (num_out_feats + 127) / 128;
      int block_num_N = num_out_channels / 16;  //j_factors1
      dim3 num_blocks(block_num_M * block_num_N); 
      dim3 threads_per_block(64);
      kernels::conv_forward_cuda_setting2_mode0_f32f32f32<<<num_blocks, threads_per_block, 0, stream>>>(
          num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else
    {
      int block_num_M = (num_out_feats + 127) / 128;
      int block_num_N = (num_out_channels + 15) / 16;  //j_factors1
      dim3 num_blocks(block_num_M * block_num_N); 
      dim3 threads_per_block(64);
      // conv_forward_cuda_setting1_mode0_tf32tf32f32<<<num_blocks, threads_per_block>>>(
      //     _out_feats.size(0), num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);

      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, false, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 8, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 4, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<16, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<8, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<8, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<8, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<8, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<4, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<4, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<4, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f32f32f32<4, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
    }
}

void dispatcher_output_stationary_half(          
    std::size_t num_out_feats,                 
    std::size_t kernel_volume,                 
    const half *in_feats,
    const half *kernel,
    half *out_feats,
    const int32_t *out_in_map,
    std::size_t num_in_channels,
    std::size_t num_out_channels, 
    cudaStream_t stream)
{
    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int j_factors1 = num_out_channels / 16 / 4;
      dim3 num_blocks((num_out_feats + 127) / 128 * j_factors1);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 4);
      kernels::conv_forward_cuda_setting3_mode0_f16f16f32<<<num_blocks, threads_per_block, 0, stream>>>(
          num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else if (num_in_channels % 32 == 0 && num_out_channels % 16 == 0)
    {
      int j_factors1 = num_out_channels / 16 / 1;
      dim3 num_blocks((num_out_feats + 127) / 128 * j_factors1);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      kernels::conv_forward_cuda_setting2_mode0_f16f16f32<<<num_blocks, threads_per_block, 0, stream>>>(
          num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else
    {
      // throw std::invalid_argument("IC is too small for this kernel");
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dim3 num_blocks((num_out_feats + 127) / 128 * j_factors1);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      // conv_forward_cuda_setting1_mode0_f16f16f32<<<num_blocks, threads_per_block>>>(
      //     _out_feats.size(0), num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, false, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 8, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 4, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 2, false, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 8 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<16, 2, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<8, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<8, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<8, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<8, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<8, 2, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<4, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<4, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<4, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<4, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<4, 2, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<2, 16, true, false><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<2, 16, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<2, 8, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<2, 4, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          kernels::conv_forward_cuda_setting1_mode0_f16f16f32<2, 2, true, true><<<num_blocks, threads_per_block, 0, stream>>>(
              num_out_feats, num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename IT>
std::vector<IT> GeneratePaddingBuckets(std::size_t num_offsets,
                                       std::size_t num_kernel_features,
                                       const std::optional<double> &threshold,
                                       const IT *h_kernel_map_sizes) {
  std::vector<IT> h_buckets;
  h_buckets.push_back(0);
  for (UIter i = 0; i < num_offsets; i++) {
    auto j = i;
    std::size_t actual_size = h_kernel_map_sizes[i];
    std::size_t summit_size = h_kernel_map_sizes[i];
    while (i + 1 < num_offsets) {
      actual_size += h_kernel_map_sizes[i + 1];
      summit_size = std::max(
          summit_size, static_cast<std::size_t>(h_kernel_map_sizes[i + 1]));
      auto padded_size = summit_size * ((i + 1) - j + 1);
      auto redundancy = static_cast<double>(padded_size - actual_size) /
                        static_cast<double>(actual_size);
      if (threshold.has_value() && (threshold < 0 || redundancy > threshold)) {
        break;
      }
      i++;
    }
    h_buckets.push_back(i + 1);
  }
  SPIRA_CHECK(h_buckets.back() == num_offsets);
  return h_buckets;
}

#define SPIRA_GENERATE_TILE_CASES(MACRO) \
  case 512: {                             \
    MACRO(512, 512)                       \
    MACRO(512, 256)                       \
    MACRO(512, 128)                       \
    MACRO(512, 64)                        \
    MACRO(512, 32)                        \
    MACRO(512, 16)                        \
    MACRO(512, 8)                         \
    MACRO(512, 4)                         \
    MACRO(512, 2)                         \
    MACRO(512, 1)                         \
    break;                                \
  }                                       \
  case 384: {                             \
    MACRO(384, 384)                       \
    MACRO(384, 192)                       \
    MACRO(384, 128)                       \
    MACRO(384, 96)                        \
    MACRO(384, 64)                        \
    MACRO(384, 48)                        \
    MACRO(384, 32)                        \
    MACRO(384, 24)                        \
    MACRO(384, 16)                        \
    MACRO(384, 12)                        \
    MACRO(384, 8)                         \
    MACRO(384, 6)                         \
    MACRO(384, 4)                         \
    MACRO(384, 3)                         \
    MACRO(384, 2)                         \
    MACRO(384, 1)                         \
    break;                                \
  }                                       \
  case 256: {                             \
    MACRO(256, 256)                       \
    MACRO(256, 128)                       \
    MACRO(256, 64)                        \
    MACRO(256, 32)                        \
    MACRO(256, 16)                        \
    MACRO(256, 8)                         \
    MACRO(256, 4)                         \
    MACRO(256, 2)                         \
    MACRO(256, 1)                         \
    break;                                \
  }                                       \
  case 192: {                             \
    MACRO(192, 192)                       \
    MACRO(192, 96)                        \
    MACRO(192, 64)                        \
    MACRO(192, 48)                        \
    MACRO(192, 32)                        \
    MACRO(192, 24)                        \
    MACRO(192, 16)                        \
    MACRO(192, 12)                        \
    MACRO(192, 8)                         \
    MACRO(192, 6)                         \
    MACRO(192, 4)                         \
    MACRO(192, 3)                         \
    MACRO(192, 2)                         \
    MACRO(192, 1)                         \
    break;                                \
  }                                       \
  case 128: {                             \
    MACRO(128, 128)                       \
    MACRO(128, 64)                        \
    MACRO(128, 32)                        \
    MACRO(128, 16)                        \
    MACRO(128, 8)                         \
    MACRO(128, 4)                         \
    MACRO(128, 2)                         \
    MACRO(128, 1)                         \
    break;                                \
  }                                       \
  case 96: {                              \
    MACRO(96, 96)                         \
    MACRO(96, 48)                         \
    MACRO(96, 32)                         \
    MACRO(96, 24)                         \
    MACRO(96, 16)                         \
    MACRO(96, 12)                         \
    MACRO(96, 8)                          \
    MACRO(96, 6)                          \
    MACRO(96, 4)                          \
    MACRO(96, 3)                          \
    MACRO(96, 2)                          \
    MACRO(96, 1)                          \
    break;                                \
  }                                       \
  case 64: {                              \
    MACRO(64, 64)                         \
    MACRO(64, 32)                         \
    MACRO(64, 16)                         \
    MACRO(64, 8)                          \
    MACRO(64, 4)                          \
    MACRO(64, 2)                          \
    MACRO(64, 1)                          \
    break;                                \
  }                                       \
  case 32: {                              \
    MACRO(32, 32)                         \
    MACRO(32, 16)                         \
    MACRO(32, 8)                          \
    MACRO(32, 4)                          \
    MACRO(32, 2)                          \
    MACRO(32, 1)                          \
    break;                                \
  }                                       \
  case 16: {                              \
    MACRO(16, 16)                         \
    MACRO(16, 8)                          \
    MACRO(16, 4)                          \
    MACRO(16, 2)                          \
    MACRO(16, 1)                          \
    break;                                \
  }                                       \
  case 8: {                               \
    MACRO(8, 8)                           \
    MACRO(8, 4)                           \
    MACRO(8, 2)                           \
    MACRO(8, 1)                           \
    break;                                \
  }                                       \
  case 4: {                               \
    MACRO(4, 4)                           \
    MACRO(4, 2)                           \
    MACRO(4, 1)                           \
    break;                                \
  }                                       \
  case 2: {                               \
    MACRO(2, 2)                           \
    MACRO(2, 1)                           \
    break;                                \
  }                                       \
  case 1: {                               \
    MACRO(1, 1)                           \
    break;                                \
  }

template <typename IT, typename FT>
void Gather(std::size_t num_sources,                 //
            std::size_t num_offsets,                 //
            std::size_t num_source_features,         //
            const IT *d_cumsum_kernel_padded_sizes,  //
            const IT *d_source_masks,                //
            const FT *d_sources,                     //
            FT *d_source_buffers,                    //
            std::size_t tile_size,                   //
            const Context &context) {
  constexpr std::size_t THREAD_BLOCK_SIZE = 128;
  switch (num_source_features) {
#define SPIRA_GATHER_CASE(NUM_FEATURES, TILE_SIZE)                         \
  if (tile_size == TILE_SIZE) {                                             \
    constexpr const auto BULK_SIZE =                                        \
        std::min(static_cast<std::size_t>(TILE_SIZE),                       \
                 (TILE_SIZE % (SizeOf<int4> / SizeOf<FT>) == 0)             \
                     ? SizeOf<int4> / SizeOf<FT>                            \
                     : SizeOf<int3> / SizeOf<FT>);                          \
    static_assert(NUM_FEATURES % TILE_SIZE == 0);                           \
    constexpr const auto NUM_TILES = NUM_FEATURES / TILE_SIZE;              \
    constexpr const auto FAKE_WARP_SIZE =                                   \
        std::min(static_cast<std::size_t>(NUM_TILES),                       \
                 static_cast<std::size_t>(WARP_SIZE));                      \
    constexpr const auto NUM_WARPS =                                        \
        DivCeil<std::size_t>(NUM_TILES, FAKE_WARP_SIZE);                    \
    context.Launch(                                                         \
        num_sources *NUM_WARPS *FAKE_WARP_SIZE, THREAD_BLOCK_SIZE, 0,       \
        kernels::PaddedGatherWarpOptimized<NUM_TILES, TILE_SIZE, BULK_SIZE, \
                                           FAKE_WARP_SIZE, IT, FT>,         \
        num_sources, num_offsets, d_cumsum_kernel_padded_sizes,             \
        d_source_masks, d_sources, d_source_buffers);                       \
    return;                                                                 \
  }
    SPIRA_GENERATE_TILE_CASES(SPIRA_GATHER_CASE)
#undef SPIRA_GATHER_CASE
    default: {
      break;
    }
  }
  SPIRA_ERROR("Unsupported num_source_features = ", num_source_features,
               " gather_tile_size = ", tile_size);
}

template <typename IT, typename FT>
void Scatter(std::size_t num_targets,                 //
             std::size_t num_offsets,                 //
             std::size_t num_target_features,         //
             const IT *d_cumsum_kernel_padded_sizes,  //
             const IT *d_target_masks,                //
             const FT *d_target_buffers,              //
             FT *d_targets,                           //
             std::size_t tile_size,                   //
             const Context &context) {
  constexpr std::size_t THREAD_BLOCK_SIZE = 128;
  switch (num_target_features) {
#define SPIRA_SCATTER_CASE(NUM_FEATURES, TILE_SIZE)                         \
  if (tile_size == TILE_SIZE) {                                              \
    constexpr const auto BULK_SIZE =                                         \
        std::min(static_cast<std::size_t>(TILE_SIZE),                        \
                 (TILE_SIZE % (SizeOf<int4> / SizeOf<FT>) == 0)              \
                     ? SizeOf<int4> / SizeOf<FT>                             \
                     : SizeOf<int3> / SizeOf<FT>);                           \
    static_assert(NUM_FEATURES % TILE_SIZE == 0);                            \
    constexpr const auto NUM_TILES = NUM_FEATURES / TILE_SIZE;               \
    constexpr const auto FAKE_WARP_SIZE =                                    \
        std::min(static_cast<std::size_t>(1) << CeilLog2(NUM_TILES),         \
                 static_cast<std::size_t>(WARP_SIZE));                       \
    constexpr const auto NUM_WARPS =                                         \
        DivCeil<std::size_t>(NUM_TILES, FAKE_WARP_SIZE);                     \
    context.Launch(                                                          \
        num_targets *NUM_WARPS *FAKE_WARP_SIZE, THREAD_BLOCK_SIZE, 0,        \
        kernels::PaddedScatterWarpOptimized<NUM_TILES, TILE_SIZE, BULK_SIZE, \
                                            FAKE_WARP_SIZE, IT, FT>,         \
        num_targets, num_offsets, d_cumsum_kernel_padded_sizes,              \
        d_target_masks, d_target_buffers, d_targets);                        \
                                                                             \
    return;                                                                  \
  }
    SPIRA_GENERATE_TILE_CASES(SPIRA_SCATTER_CASE)
#undef SPIRA_SCATTER_CASE
    default: {
      break;
    }
  }
  SPIRA_ERROR("Unsupported num_target_features = ", num_target_features,
               "scatter_tile_size = ", tile_size);
}

template <typename IT>
cpu::Memory<IT> ComputeCumsumKernelPaddedSizes(
    std::size_t num_offsets,           //
    const std::vector<IT> &h_buckets,  //
    const cpu::Memory<IT> &h_kernel_map_sizes) {
  // Compute padding strategy & padding indices
  cpu::Memory<IT> h_cumsum_kernel_padded_sizes(num_offsets + 1);
  h_cumsum_kernel_padded_sizes[0] = 0;
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    std::size_t summit_size = 0;
    for (UIter j = h_buckets[i]; j < h_buckets[i + 1]; j++) {
      summit_size = std::max(summit_size,
                             static_cast<std::size_t>(h_kernel_map_sizes[j]));
    }
    for (UIter j = h_buckets[i]; j < h_buckets[i + 1]; j++) {
      h_cumsum_kernel_padded_sizes[j + 1] = h_cumsum_kernel_padded_sizes[j];
      h_cumsum_kernel_padded_sizes[j + 1] += summit_size;
    }
  }
  return h_cumsum_kernel_padded_sizes;
}



template <typename FT>
float TimeGather::operator()(
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
    const Context &context) const {

  const auto num_kernel_features = num_source_features * num_target_features;

  cpu::Memory<int32_t> h_kernel_map_sizes(num_offsets);

  if(is_hybrid || ((num_offsets % 2 == 1) && (num_sources == num_targets))){
    size_t half = (num_offsets + 1) / 2;
    context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), half);

    // Fill the second half using symmetry
    for (size_t i = half; i < num_offsets; ++i) {
        h_kernel_map_sizes[i] = h_kernel_map_sizes[num_offsets - 1 - i];
    }    
  }

  else{
      context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), num_offsets);
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);


  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_sources = context.NewBuffer<FT>(num_sources * num_source_features);

  Event prior, after;

  prior.Record(context.stream());

  Gather(num_sources, num_offsets, num_source_features,
         d_cumsum_kernel_padded_sizes.device_data(), d_source_masks,
         d_sources.device_data(), d_source_buffers.device_data(),
         gather_tile_size, context);


  after.Record(context.stream());
  after.Synchronize();

  return prior.Elapsed(after);
}

template <typename FT>
float TimeScatter::operator()(
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
    const Context &context) const {

  const auto num_kernel_features = num_source_features * num_target_features;
  cpu::Memory<int32_t> h_kernel_map_sizes(num_offsets);

  if(is_hybrid || ((num_offsets % 2 == 1) && (num_sources == num_targets))){
    size_t half = (num_offsets + 1) / 2;
    context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), half);

    // Fill the second half using symmetry
    for (size_t i = half; i < num_offsets; ++i) {
        h_kernel_map_sizes[i] = h_kernel_map_sizes[num_offsets - 1 - i];
    }    
  }

  else{
      context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), num_offsets);
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  //
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);
  auto d_targets = context.NewBuffer<FT>(num_targets * num_target_features);

  Event prior, after;
  prior.Record(context.stream());
  Scatter(num_targets, num_offsets, num_target_features,
          d_cumsum_kernel_padded_sizes.device_data(), d_target_masks,
          d_target_buffers.device_data(), d_targets.device_data(), tile_size,
          context);
  after.Record(context.stream());
  after.Synchronize();
  return prior.Elapsed(after);
}


template <typename FT>
float TimeGEMM::operator()(
    std::size_t num_sources,                 // S
    std::size_t num_targets,                 // T
    std::size_t num_offsets,                 // O
    bool is_hybrid,
    const std::optional<double> &threshold,  //
    std::size_t parallel,
    std::size_t num_source_features,  // C_in
    std::size_t num_target_features,  // C_out
    const int32_t *d_kernel_map_sizes,     // [O]
    const FT *d_weights,              // [O, C_in, C_out]
    const Context &context) const {

  const auto num_kernel_features = num_source_features * num_target_features;
  cpu::Memory<int32_t> h_kernel_map_sizes(num_offsets);


  if(is_hybrid || ((num_offsets % 2 == 1) && (num_sources == num_targets))){
    size_t half = (num_offsets + 1) / 2;
    context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), half);

    // Fill the second half using symmetry
    for (size_t i = half; i < num_offsets; ++i) {
        h_kernel_map_sizes[i] = h_kernel_map_sizes[num_offsets - 1 - i];
    }    
  }

  else{
      context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), num_offsets);
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());
  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  // Create buffer for sources and targets
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);

  parallel = std::min(parallel, h_buckets.size() - 1);
  if (parallel > 1) {
    while (stream_pool_.size() < parallel) {
      stream_pool_.emplace_back(cudaStreamNonBlocking);
    }
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(context.stream());
      SPIRA_CHECK_CUDA(
          cudaStreamWaitEvent(stream_pool_[i].stream(), event.event()));
    }
  }

  Event prior, after;
  prior.Record(context.stream());
  // Perform GEMMs
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    auto padded_base = h_cumsum_kernel_padded_sizes[h_buckets[i]];

    auto m = *std::max_element(h_kernel_map_sizes.data() + h_buckets[i],
                               h_kernel_map_sizes.data() + h_buckets[i + 1]);
    //    auto m = h_kernel_map_sizes[h_buckets[i + 1] - 1];
    if (m == 0) {
      continue;
    }
    auto b = h_buckets[i + 1] - h_buckets[i];

    auto d_a = d_source_buffers.device_data();
    d_a += padded_base * num_source_features;

    auto d_b = d_weights;
    d_b += h_buckets[i] * num_kernel_features;

    auto d_c = d_target_buffers.device_data();
    d_c += padded_base * num_target_features;

    auto stream =
        (parallel > 1) ? stream_pool_[i % parallel].stream() : context.stream();

    BatchedMatMul(b,                    // b
                  m,                    // m
                  num_source_features,  // k
                  num_target_features,  // n
                  false,                // is_a_transposed
                  false,                // is_b_transposed
                  d_a,                  // d_a
                  d_b,                  // d_b
                  d_c,                  // d_c
                  false,                // incremental
                  context,              // context
                  stream);
  }
  if (parallel > 1) {
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(stream_pool_[i].stream());
      SPIRA_CHECK_CUDA(cudaStreamWaitEvent(context.stream(), event.event()));
    }
  }
  // Note that we don't include the last offset (if there is) since it's
  // always the same regardless of `threshold`
  after.Record(context.stream());
  after.Synchronize();
  return prior.Elapsed(after);
}



template <typename FT>
void SparseConvolutionForward_WS::operator()(
    std::size_t num_sources,                 // S
    std::size_t num_targets,                 // T
    std::size_t num_offsets,                 // O
    const std::optional<double> &threshold,  //
    std::size_t parallel,                    //
    std::size_t num_source_features,  // C_in
    std::size_t num_target_features,  // C_out
    const int32_t *d_source_masks,         // [S, O]
    const int32_t *d_target_masks,         // [T, O]
    const int32_t *d_kernel_map_sizes,     // [O]
    const FT *d_sources,              // [S, C_in]
    const FT *d_weights,              // [O, C_in, C_out]
    FT *d_targets,                    // [T, C_out]
    std::size_t gather_tile_size,     //
    std::size_t scatter_tile_size,    //
    const Context &context) const {

  const auto num_kernel_features = num_source_features * num_target_features;

  cpu::Memory<int32_t> h_kernel_map_sizes(num_offsets);
  
  if((num_offsets % 2 == 1) && (num_sources == num_targets)){
    size_t half = (num_offsets + 1) / 2;
    context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), half);

    // Fill the second half using symmetry
    for (size_t i = half; i < num_offsets; ++i) {
        h_kernel_map_sizes[i] = h_kernel_map_sizes[num_offsets - 1 - i];
    }    
  }

  else{
      context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), num_offsets);
  }

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());

  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  // Create buffer for sources and targets
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);

  // Gather from all sources
  Gather(num_sources, num_offsets, num_source_features,
         d_cumsum_kernel_padded_sizes.device_data(), d_source_masks, d_sources,
         d_source_buffers.device_data(), gather_tile_size, context);


  parallel = std::min(parallel, h_buckets.size() - 1);
  if (parallel > 1) {
    while (stream_pool_.size() < parallel) {
      stream_pool_.emplace_back(cudaStreamNonBlocking);
    }
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(context.stream());
      SPIRA_CHECK_CUDA(
          cudaStreamWaitEvent(stream_pool_[i].stream(), event.event()));
    }
  }

  // Perform GEMMs
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    auto padded_base = h_cumsum_kernel_padded_sizes[h_buckets[i]];

    auto m = *std::max_element(h_kernel_map_sizes.data() + h_buckets[i],
                               h_kernel_map_sizes.data() + h_buckets[i + 1]);
    if (m == 0) {
      continue;
    }
    auto b = h_buckets[i + 1] - h_buckets[i];

    auto d_a = d_source_buffers.device_data();
    d_a += padded_base * num_source_features;

    auto d_b = d_weights;
    d_b += h_buckets[i] * num_kernel_features;

    auto d_c = d_target_buffers.device_data();
    d_c += padded_base * num_target_features;

    auto stream =
        (parallel > 1) ? stream_pool_[i % parallel].stream() : context.stream();

    BatchedMatMul(b,                    // b
                  m,                    // m
                  num_source_features,  // k
                  num_target_features,  // n
                  false,                // is_a_transposed
                  false,                // is_b_transposed
                  d_a,                  // d_a
                  d_b,                  // d_b
                  d_c,                  // d_c
                  false,                // incremental
                  context,              //
                  stream);
  }

  if (parallel > 1) {
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(stream_pool_[i].stream());
      SPIRA_CHECK_CUDA(cudaStreamWaitEvent(context.stream(), event.event()));
    }
  }

  Scatter(num_targets, num_offsets, num_target_features,
          d_cumsum_kernel_padded_sizes.device_data(), d_target_masks,
          d_target_buffers.device_data(), d_targets, scatter_tile_size,
          context);

} 

template <typename FT>
void SparseConvolutionForward_HS::operator()(
    std::size_t num_sources,                 
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
    const Context &context) const {

  const auto num_kernel_features = num_source_features * num_target_features;

  cpu::Memory<int32_t> h_kernel_map_sizes(num_offsets_ws);
  
  size_t half = num_offsets_ws / 2;
  context.ReadDeviceData(d_kernel_map_sizes, h_kernel_map_sizes.data(), half);

  for (size_t i = half; i < num_offsets_ws; ++i) {
      h_kernel_map_sizes[i] = h_kernel_map_sizes[num_offsets_ws - 1 - i];
  }    

  // Compute padding strategy & padding indices
  auto h_buckets = GeneratePaddingBuckets(num_offsets_ws, num_kernel_features,
                                          threshold, h_kernel_map_sizes.data());

  auto h_cumsum_kernel_padded_sizes = ComputeCumsumKernelPaddedSizes(
      num_offsets_ws, h_buckets, h_kernel_map_sizes);
  auto d_cumsum_kernel_padded_sizes =
      context.NewBufferFrom(h_cumsum_kernel_padded_sizes);

  // Create buffer for sources and targets
  auto total_size = h_cumsum_kernel_padded_sizes[num_offsets_ws];
  auto d_source_buffers =
      context.NewBuffer<FT>(total_size * num_source_features);
  auto d_target_buffers =
      context.NewBuffer<FT>(total_size * num_target_features);
 
  Stream os_stream(cudaStreamNonBlocking);
  Event os_done(cudaEventDisableTiming);

  SparseConvolutionForward_OS os_forward;

  os_forward(num_sources,
             num_targets,
             num_offsets_os,
             num_source_features,
             num_target_features,
             d_out_in_map,
             d_sources,
             d_weights_os,
             d_targets_os,
             context,
             os_stream.stream());

  os_done.Record(os_stream.stream());

  // Gather from all sources
  Gather(num_sources, num_offsets_ws, num_source_features,
         d_cumsum_kernel_padded_sizes.device_data(), d_source_masks, d_sources,
         d_source_buffers.device_data(), gather_tile_size, context);

  parallel = std::min(parallel, h_buckets.size() - 1);
  if (parallel > 1) {
    while (stream_pool_.size() < parallel) {
      stream_pool_.emplace_back(cudaStreamNonBlocking);
    }
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(context.stream());
      SPIRA_CHECK_CUDA(
          cudaStreamWaitEvent(stream_pool_[i].stream(), event.event()));
    }
  }

  // Perform GEMMs
  for (UIter i = 0; i + 1 < h_buckets.size(); i++) {
    auto padded_base = h_cumsum_kernel_padded_sizes[h_buckets[i]];

    auto m = *std::max_element(h_kernel_map_sizes.data() + h_buckets[i],
                               h_kernel_map_sizes.data() + h_buckets[i + 1]);
    if (m == 0) {
      continue;
    }
    auto b = h_buckets[i + 1] - h_buckets[i];

    auto d_a = d_source_buffers.device_data();
    d_a += padded_base * num_source_features;

    auto d_b = d_weights_ws;
    d_b += h_buckets[i] * num_kernel_features;

    auto d_c = d_target_buffers.device_data();
    d_c += padded_base * num_target_features;

    auto stream =
        (parallel > 1) ? stream_pool_[i % parallel].stream() : context.stream();

    BatchedMatMul(b,                    // b
                  m,                    // m
                  num_source_features,  // k
                  num_target_features,  // n
                  false,                // is_a_transposed
                  false,                // is_b_transposed
                  d_a,                  // d_a
                  d_b,                  // d_b
                  d_c,                  // d_c
                  false,                // incremental
                  context,              //
                  stream);
  }

  if (parallel > 1) {
    for (UIter i = 0; i < parallel; i++) {
      Event event(cudaEventDisableTiming);
      event.Record(stream_pool_[i].stream());
      SPIRA_CHECK_CUDA(cudaStreamWaitEvent(context.stream(), event.event()));
    }
  }      

  Scatter(num_targets, num_offsets_ws, num_target_features,
          d_cumsum_kernel_padded_sizes.device_data(), d_target_masks,
          d_target_buffers.device_data(), d_targets_ws, scatter_tile_size,
          context);

  SPIRA_CHECK_CUDA(cudaStreamWaitEvent(context.stream(), os_done.event()));  

} 

template <typename FT>
void SparseConvolutionForward_WS_Merged::operator()(
    std::size_t num_sources,                 
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
    cudaStream_t stream) const {

    if constexpr (std::is_same_v<FT, float>) {
        dispatcher_weight_stationary(num_targets, num_offsets, ((num_sources == num_targets) && (num_offsets % 2) == 1), is_hybrid, qsum_nnz, d_sources, d_weights, d_targets,
                                          kpos_ptr, qkpos_ptr, in_map_ptr, out_map_ptr, num_source_features, num_target_features,
                                          stream);
    } 
    else if constexpr (std::is_same_v<FT, half>) {
        dispatcher_weight_stationary_half(num_targets, num_offsets, ((num_sources == num_targets) && (num_offsets % 2) == 1), is_hybrid, qsum_nnz, d_sources, d_weights, d_targets,
                                          kpos_ptr, qkpos_ptr, in_map_ptr, out_map_ptr, num_source_features, num_target_features,
                                          stream);
    }
}



template <typename FT>
void SparseConvolutionForward_HS_Merged::operator()(
    std::size_t num_sources,                 
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
    const Context &context) const {

  SparseConvolutionForward_OS os_forward;

  SparseConvolutionForward_WS_Merged ws_forward;

  os_forward(num_sources,
             num_targets,
             num_offsets_os,
             num_source_features,
             num_target_features,
             d_out_in_map,
             d_sources,
             d_weights_os,
             d_targets,
             context,
             context.stream());

  ws_forward(num_sources,
             num_targets,
             num_offsets_ws,
             num_source_features,
             num_target_features,
             qsum_nnz,
             true,
             in_map_ptr,
             out_map_ptr,
             kpos_ptr,
             qkpos_ptr,
             d_sources,
             d_weights_ws,
             d_targets,
             context.stream());     
}



template <typename FT>
void SparseConvolutionForward_OS::operator()(
    std::size_t num_sources,                 
    std::size_t num_targets,                 
    std::size_t num_offsets,                
    std::size_t num_source_features,        
    std::size_t num_target_features,         
    const int32_t *d_out_in_map,                
    const FT *d_sources,                     
    const FT *d_weights,                  
    FT *d_targets, 
    const Context &context,                          
    cudaStream_t stream) const {

    if(num_offsets == 1){                 //optimized case only for 1 dense column
          MatMul(num_targets,         
             num_source_features, 
             num_target_features,  
             false,                
             false,              
             d_sources,            
             d_weights,                  
             d_targets,            
             false,                 
             context,
             stream);
    }


    else if constexpr (std::is_same_v<FT, float>) {
          dispatcher_output_stationary(num_targets, num_offsets, d_sources, d_weights, d_targets,
                                            d_out_in_map, num_source_features, num_target_features,
                                            stream);                                               
    } 
    else if constexpr (std::is_same_v<FT, half>) {
        dispatcher_output_stationary_half(num_targets, num_offsets, d_sources, d_weights, d_targets,
                                          d_out_in_map, num_source_features, num_target_features,
                                          stream);
    }
} 


#define SPIRA_EXPLICIT_INSTANTIATION(_, FT)                        \
  template void SparseConvolutionForward_OS::operator()<FT>(        \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets,                                          \
      std::size_t num_source_features, std::size_t num_target_features, \
      const int32_t *d_out_in_map,                                           \
      const FT *d_sources, const FT *d_weights, FT *d_targets,          \
      const Context &context, cudaStream_t stream) const;                                       \
  template void SparseConvolutionForward_HS::operator()<FT>(        \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets_ws, std::size_t num_offsets_os, const std::optional<double> &threshold,  \
      std::size_t parallel,                                             \
      std::size_t num_source_features, std::size_t num_target_features, \
      const int32_t *d_source_masks, const int32_t *d_target_masks,               \
      const int32_t *d_kernel_map_sizes, const int32_t *d_out_in_map,             \
      const FT *d_sources, const FT *d_weights_ws, const FT *d_weights_os, FT *d_targets_ws, FT *d_targets_os,          \
      std::size_t gather_tile_size, std::size_t scatter_tile_size,      \
      const Context &context) const;                                    \
    template void SparseConvolutionForward_HS_Merged::operator()<FT>(   \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets_ws, std::size_t num_offsets_os,           \
      std::size_t num_source_features, std::size_t num_target_features, \
      int32_t qsum_nnz,                                              \
      const int32_t *in_map_ptr, const int32_t *out_map_ptr,         \
      const int32_t *kpos_ptr, const int32_t *qkpos_ptr, const int32_t *d_out_in_map, \
      const FT *d_sources, const FT *d_weights_ws, const FT *d_weights_os, \
      FT *d_targets,                                                    \
      const Context &context) const;                                    \
    template void SparseConvolutionForward_WS::operator()<FT>(        \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, const std::optional<double> &threshold,  \
      std::size_t parallel,                                             \
      std::size_t num_source_features, std::size_t num_target_features, \
      const int32_t *d_source_masks, const int32_t *d_target_masks,               \
      const int32_t *d_kernel_map_sizes,                                     \
      const FT *d_sources, const FT *d_weights, FT *d_targets,          \
      std::size_t gather_tile_size, std::size_t scatter_tile_size,      \
      const Context &context) const;                                    \
    template void SparseConvolutionForward_WS_Merged::operator()<FT>(   \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets,                                          \
      std::size_t num_source_features, std::size_t num_target_features, \
      int32_t qsum_nnz,                                              \
      bool is_hybrid,                                                \
      const int32_t *in_map_ptr, const int32_t *out_map_ptr,         \
      const int32_t *kpos_ptr, const int32_t *qkpos_ptr,             \
      const FT *d_sources, const FT *d_weights, FT *d_targets,       \
      cudaStream_t stream) const;                                    \
  template float TimeGather::operator()<FT>(                        \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, bool is_hybrid, const std::optional<double> &threshold,  \
      std::size_t num_source_features,      \
      std::size_t num_target_features, const int32_t *d_source_masks,        \
      const int32_t *d_kernel_map_sizes, std::size_t gather_tile_size,       \
      const Context &context) const;                                    \
  template float TimeScatter::operator()<FT>(                       \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, bool is_hybrid, const std::optional<double> &threshold,  \
      std::size_t num_source_features,      \
      std::size_t num_target_features, const int32_t *d_target_masks,        \
      const int32_t *d_kernel_map_sizes, std::size_t tile_size,              \
      const spira::cuda::Context &context) const;                      \
  template float TimeGEMM::operator()<FT>(                          \
      std::size_t num_sources, std::size_t num_targets,                 \
      std::size_t num_offsets, bool is_hybrid, const std::optional<double> &threshold,  \
      std::size_t parallel,                 \
      std::size_t num_source_features, std::size_t num_target_features, \
      const int32_t *d_kernel_map_sizes, const FT *d_weights,                \
      const spira::cuda::Context &context) const

SPIRA_FOR_ALL_F_TYPES(SPIRA_EXPLICIT_INSTANTIATION);

}  // namespace spira::cuda
