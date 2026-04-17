#pragma once

#include <cuda_fp16.h>
#include <mma.h>

namespace spira::cuda::kernels {

__device__ __forceinline__ int binary_search(
                            const int *S_csrRowPtr, const int eid, 
                            const int start, const int end) {
    
  int lo = start, hi = end;
  if (lo == hi){
    return lo;
  }
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (__ldg(S_csrRowPtr + mid) <= eid) {
        lo = mid + 1;
    } else {
        hi = mid;
    }
  }
  if (__ldg(S_csrRowPtr + hi) <= eid) {
    return hi;
  } else {
      return hi - 1;
  }
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen)   
{
    // Block index
    const int bx = blockIdx.x;

    const int by0 = blockIdx.y;
    const int by_stride = gridDim.y;

    for (int byi = by0; byi < tiles_y; byi += by_stride) {

        auto by = byi;

        int widx;
        const float *kw_ptr;

        const int* imap_ptr;
        const int* omap_ptr;

        if (by < margin) {
            imap_ptr = imap;
            omap_ptr = omap;

            auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

            widx = binary_search(qkpos,
                                 by * N_LOOP * BLOCK_SIZE,
                                 0,
                                 search_vol);

            kw_ptr = &kw[widx * c_in * c_out];
        }
        else {
            imap_ptr = omap;
            omap_ptr = imap;

            by -= margin;

            auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

            widx = binary_search(qkpos,
                                 by * N_LOOP * BLOCK_SIZE,
                                 0,
                                 search_vol);

            kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];
        }

        // Thread indices
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int ctx = tx << 2;

        // Coordinates
        const int cx = BLOCK_SIZE * bx + ctx;
        const int y = BLOCK_SIZE * N_LOOP * by + ty  
                      - __ldg(&qkpos[widx])
                      + __ldg(&kpos[widx]);

        float Csub[N_LOOP][4] = {0.0f};
        float padding[4] = {0.0f};

        __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

        // Load kernel weights into Bs
        *((float4*)(&Bs[ty][ctx])) =
            (ty < c_in && cx < c_out)
            ? *((float4*)(kw_ptr + c_out * ty + cx))
            : *((float4*)(&padding[0]));

        // Load input features into As
        for (int n = 0; n < N_LOOP; n++) {

            int y_temp = y + n * BLOCK_SIZE;
            int in_row = (y_temp < __ldg(&kpos[widx + 1]))
                         ? imap_ptr[y_temp]
                         : -1;

            *((float4*)(&As[n][ty][ctx])) =
                (ctx < c_in && in_row > -1)
                ? *((float4*)(&in_f[c_in * in_row + ctx]))
                : *((float4*)(&padding[0]));
        }

        __syncthreads();

        // GEMM
#pragma unroll
        for (int n = 0; n < N_LOOP; n++) {

#pragma unroll
            for (int k = 0; k < c_in; ++k) {
                float Ast = As[n][ty][k];

#pragma unroll
                for (int c = 0; c < 4; c++){
                    Csub[n][c] += Ast * Bs[k][ctx + c];
                }
            }

            int y_temp = y + n * BLOCK_SIZE;
            int out_row = (y_temp < __ldg(&kpos[widx + 1]))
                          ? omap_ptr[y_temp]
                          : -1;

            if (out_row > -1 && cx < c_out) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
                }
            }
        }
    } 
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {
   
    // Block index
  const int bx = blockIdx.x;

  const int by0 = blockIdx.y;
  const int by_stride = gridDim.y;

  for (int byi = by0; byi < tiles_y; byi += by_stride) {

    auto by = byi;

    int widx;
    const float *kw_ptr;

    const int* imap_ptr;
    const int* omap_ptr;

    if(by < margin){
      imap_ptr = imap;
      omap_ptr = omap;

      auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

      widx = binary_search(qkpos,
                            by * N_LOOP * BLOCK_SIZE,
                            0,
                            search_vol);

      kw_ptr = &kw[widx * c_in * c_out];    
    }

    else{
      imap_ptr = omap;
      omap_ptr = imap;
      by -= margin;
      auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

      widx = binary_search(qkpos,
                            by * N_LOOP * BLOCK_SIZE,
                            0,
                            search_vol);

      kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];    
    }
    
    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ctx = tx << 2;

    // Coordinate. x is for rows, y is for columns.
    const int cx = BLOCK_SIZE * bx + ctx;
    const int y = BLOCK_SIZE * N_LOOP * by + ty 
      - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub[N_LOOP][4] = {0.0f};
    float padding[4] = {0.0f};

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int s = 0; s < c_in; s += BLOCK_SIZE) {

      // Load the matrices from device memory
      // to shared memory; each thread loads
      // one element of each matrix

      // Kernel weight to Bs
      *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
        *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
        *((float4*)(&padding[0]));
      
      // Input feature to As
      for (int n = 0; n < N_LOOP; n++){

        int y_temp = y + n * BLOCK_SIZE;

        // The thread deals with the x-th channel of the y-th output
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

        *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
          *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
          *((float4*)(&padding[0]));
      }

      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
  #pragma unroll 
      for (int n = 0; n < N_LOOP; n++){
  #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          float Ast = As[n][ty][k];
  #pragma unroll
          for (int c = 0; c < 4; c++){
            Csub[n][c] += Ast * Bs[k][ctx + c];
          }
        }
      }

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
  #pragma unroll
    for (int n = 0; n < N_LOOP; n++){
      int y_temp = y + n * BLOCK_SIZE;
      int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
      if (out_row > -1 && cx < c_out){
  #pragma unroll
        for (int c = 0; c < 4; c++){
          atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
        }
      }
    }
  }
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {

    // Block index
  const int bx = blockIdx.x;

  const int by0 = blockIdx.y;
  const int by_stride = gridDim.y;

  for (int byi = by0; byi < tiles_y; byi += by_stride) {

    auto by = byi;

    int widx;
    const float *kw_ptr;

    const int* imap_ptr;
    const int* omap_ptr;

    if(by < margin){
      imap_ptr = imap;
      omap_ptr = omap;
      auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

      widx = binary_search(qkpos,
                            by * N_LOOP * BLOCK_SIZE,
                            0,
                            search_vol);
      kw_ptr = &kw[widx * c_in * c_out];    
    }

    else{
      imap_ptr = omap;
      omap_ptr = imap;
      by -= margin;
      auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

      widx = binary_search(qkpos,
                            by * N_LOOP * BLOCK_SIZE,
                            0,
                            search_vol);
      kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];    
    }

    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // const int ctx = tx;

    // Coordinate. x is for rows, y is for columns.
    const int cx = BLOCK_SIZE * bx + tx;
    const int y = BLOCK_SIZE * N_LOOP * by + ty 
      - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub[N_LOOP] = {0.0f};
    float padding = 0.0f;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int s = 0; s < c_in; s += BLOCK_SIZE) {

      // Load the matrices from device memory
      // to shared memory; each thread loads
      // one element of each matrix

      // Kernel weight to Bs
      Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
        *(kw_ptr + c_out * (s + ty) + cx) : 
        padding;
      
      // Input feature to As
      for (int n = 0; n < N_LOOP; n++){

        int y_temp = y + n * BLOCK_SIZE;

        // The thread deals with the x-th channel of the y-th output
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

        As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
          in_f[c_in * in_row + s + tx] : 
          padding;
      }

      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
  #pragma unroll 
      for (int n = 0; n < N_LOOP; n++){
  #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          // float Ast = As[n][ty][k];
          // for (int c = 0; c < 2; c++){
          Csub[n] += As[n][ty][k] * Bs[k][tx];
          // }
        }
      }

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
  #pragma unroll
    for (int n = 0; n < N_LOOP; n++){
      int y_temp = y + n * BLOCK_SIZE;
      int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
      if (out_row > -1 && cx < c_out){
        // for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
        // }
      }
    }
  }
}

template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int margin,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {

    // Block index
  const int bx = blockIdx.x;

  const int by0 = blockIdx.y;
  const int by_stride = gridDim.y;

  for (int byi = by0; byi < tiles_y; byi += by_stride) {
    
    auto by = byi;

    int widx;
    const float *kw_ptr;

    const int* imap_ptr;
    const int* omap_ptr;

    if(by < margin){
      imap_ptr = imap;
      omap_ptr = omap;
      auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

      widx = binary_search(qkpos,
                            by * N_LOOP * BLOCK_SIZE,
                            0,
                            search_vol);
      kw_ptr = &kw[widx * c_in * c_out];    
    }

    else{
      imap_ptr = omap;
      omap_ptr = imap;
      by -= margin;
      auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

      widx = binary_search(qkpos,
                            by * N_LOOP * BLOCK_SIZE,
                            0,
                            search_vol);
      kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];    
    }


    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ctx = tx << 1;


    // Coordinate. x is for rows, y is for columns.
    const int cx = BLOCK_SIZE * bx + ctx;
    const int y = BLOCK_SIZE * N_LOOP * by + ty 
      - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub[N_LOOP][2] = {0.0f};
    float padding[2] = {0.0f};

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int s = 0; s < c_in; s += BLOCK_SIZE) {

      // Load the matrices from device memory
      // to shared memory; each thread loads
      // one element of each matrix

      // Kernel weight to Bs
      *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
        *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
        *((float2*)(&padding[0]));
      
      // Input feature to As
      for (int n = 0; n < N_LOOP; n++){

        int y_temp = y + n * BLOCK_SIZE;

        // The thread deals with the x-th channel of the y-th output
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

        *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
          *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
          *((float2*)(&padding[0]));
      }

      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
  #pragma unroll 
      for (int n = 0; n < N_LOOP; n++){
  #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          float Ast = As[n][ty][k];
  #pragma unroll
          for (int c = 0; c < 2; c++){
            Csub[n][c] += Ast * Bs[k][ctx + c];
          }
        }
      }

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
  #pragma unroll
    for (int n = 0; n < N_LOOP; n++){
      int y_temp = y + n * BLOCK_SIZE;
      int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
      if (out_row > -1 && cx < c_out){
  #pragma unroll
        for (int c = 0; c < 2; c++){
          atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
        }
      }
    }
  }
}

template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_4_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {

  # if __CUDA_ARCH__ >= 700
    // Block index
    const int bx = blockIdx.x;

    const int by0 = blockIdx.y;
    const int by_stride = gridDim.y;

    for (int byi = by0; byi < tiles_y; byi += by_stride) {

      auto by = byi;

      int widx;
      const half *kw_ptr;

      const int* imap_ptr;
      const int* omap_ptr;

      if(by < margin){
        imap_ptr = imap;
        omap_ptr = omap;
        auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

        widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
        kw_ptr = &kw[widx * c_in * c_out];    
      }

      else{
        imap_ptr = omap;
        omap_ptr = imap;
        by -= margin;
        auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

        widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
        kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];    
      }
      
      // Thread index
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int ctx = tx << 2;


      // Coordinate. x is for rows, y is for columns.
      const int cx = BLOCK_SIZE * bx + ctx;
      const int y = BLOCK_SIZE * N_LOOP * by + ty - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

      // Csub is used to store the element of the block sub-matrix
      // that is computed by the thread
      half Csub[N_LOOP][4] = {__float2half(0.0f)};
      half padding[4] = {__float2half(0.0f)};

      // Declaration of the shared memory array As used to
      // store the sub-matrix of A
      __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

      // Declaration of the shared memory array Bs used to
      // store the sub-matrix of B
      __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
      
      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      // In "loop once" version, s = 0
      // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

      // Kernel weight to Bs
      *((float2*)(&Bs[ty][ctx])) = (ty < c_in && cx < c_out) ? 
        *((float2*)(kw_ptr + c_out * ty + cx)) : 
        *((float2*)(&padding[0]));
        
      int y_temp = y;
      // Input feature to As
      for (int n = 0; n < N_LOOP; n++){

        // The thread deals with the x-th channel of the y-th output
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

        *((float2*)(&As[n][ty][ctx])) = (ctx < c_in && in_row > -1) ? 
          *((float2*)(&in_f[c_in * in_row + ctx])) : 
          *((float2*)(&padding[0]));
          
        y_temp += BLOCK_SIZE;
      }

      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
    #pragma unroll 
      for (int n = 0; n < N_LOOP; n++){
    #pragma unroll
        for (int k = 0; k < c_in; ++k){
          half Ast = As[n][ty][k];
    #pragma unroll
          for (int c = 0; c < 4; c++){
            Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
          }
        }
        int y_temp = y + n * BLOCK_SIZE;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
        if (out_row > -1 && cx < c_out){
    #pragma unroll
          for (int c = 0; c < 4; c++){
            atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
          }
        }
      }
    }
  #else
    #pragma message("FP16 kernels will not be compiled.")
  #endif
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_fp16_tc4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {
  # if __CUDA_ARCH__ >= 700             
    // Block index
    const int bx = blockIdx.x;

    const int by0 = blockIdx.y;
    const int by_stride = gridDim.y;

    for (int byi = by0; byi < tiles_y; byi += by_stride) {

      auto by = byi;

      int widx;
      const half *kw_ptr;

      const int* imap_ptr;
      const int* omap_ptr;

      if(by < margin){
        imap_ptr = imap;
        omap_ptr = omap;
        auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

        widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
        kw_ptr = &kw[widx * c_in * c_out];
      }

      else{
        imap_ptr = omap;
        omap_ptr = imap;
        by -= margin;
        auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

        widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);

        kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];
      }

      // Thread index
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int ctx = tx << 2;
      const int tid = ty * blockDim.x + tx;

      // Warp index
      const int warpId = tid / 32;
      // const int laneId = tid % 32;
      const int warp_row = warpId / NS;
      const int warp_col = warpId % NS;
      
      // Coordinate. x is for rows, y is for columns.
      const int cx = BLOCK_SIZE * bx + ctx;
      const int y = BLOCK_SIZE * N_LOOP * by + ty 
        - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

      // Csub is used to store the element of the block sub-matrix
      // that is computed by the thread
      // float Csub[N_LOOP][4] = {0.0f};
      half padding[4] = {__float2half(0.0f)};

      // Declaration of the shared memory array As used to
      // store the sub-matrix of A
      __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

      // Declaration of the shared memory array Bs used to
      // store the sub-matrix of B
      __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

      // Declaration of the shared memeory array Cs used to
      // store the sub-matrix of C
      // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

      // Fragments to store As, Bs and Cs
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

    #pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::fill_fragment(c[n], __float2half(0.0f));
      }
      
      // May not be necessary
      __syncthreads();

      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      for (int s = 0; s < c_in; s += BLOCK_SIZE) {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix

        // Kernel weight to Bs
        *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
          *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
          *((float2*)(&padding[0]));
        
        // Input feature to As
        for (int n = 0; n < N_LOOP; n++){

          int y_temp = y + n * BLOCK_SIZE;

          // The thread deals with the x-th channel of the y-th output
          int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

          *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
            *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
            *((float2*)(&padding[0]));
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together using Tensor Core
        // Load data from shmem to tensor core
        // Just load Bs once
    #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += K){
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major> a[N_LOOP / 2];
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::row_major> b;
          nvcuda::wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
    #pragma unroll
          for (int n = 0; n < N_LOOP / 2; n++){
            nvcuda::wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
            nvcuda::wmma::mma_sync(c[n], a[n], b, c[n]);
          }  
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }

      // Store C fragments to shared memory
      // Note that we reuse As for Cs storing
    #pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        nvcuda::wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
          c[n], BLOCK_SIZE + SKEW, nvcuda::wmma::mem_row_major);
      }

      // Synchronize to make sure that all C fragments are 
      // stored into shared memory
      __syncthreads();

      // Write the block sub-matrix to device memory;
      // each thread writes one element
    #pragma unroll
      for (int n = 0; n < N_LOOP; n++){
        int y_temp = y + n * BLOCK_SIZE;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
        if (out_row > -1 && cx < c_out){
    #pragma unroll
          for (int c = 0; c < 4; c++){
            atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
          }
        }
      }
    }
  #else
    #pragma message("FP16 kernels will not be compiled.")
  #endif
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {
  #if __CUDA_ARCH__ >= 700
    // Block index
    const int bx = blockIdx.x;

    const int by0 = blockIdx.y;
    const int by_stride = gridDim.y;

    for (int byi = by0; byi < tiles_y; byi += by_stride) {

        auto by = byi;

        int widx;
        const half *kw_ptr;

        const int* imap_ptr;
        const int* omap_ptr;

        if(by < margin){
            imap_ptr = imap;
            omap_ptr = omap;
            auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

            widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
            kw_ptr = &kw[widx * c_in * c_out];    
        }

        else{
            imap_ptr = omap;
            omap_ptr = imap;
            by -= margin;
            auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

            widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
            kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];    
        }

      // Thread index
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      // const int ctx = tx << 1;

      // Coordinate. x is for rows, y is for columns.
      const int cx = BLOCK_SIZE * bx + tx;
      const int y = BLOCK_SIZE * N_LOOP * by + ty 
        - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

      // Csub is used to store the element of the block sub-matrix
      // that is computed by the thread
      half Csub[N_LOOP] = {__float2half(0.0f)};
      half padding = __float2half(0.0f);

      // Declaration of the shared memory array As used to
      // store the sub-matrix of A
      __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

      // Declaration of the shared memory array Bs used to
      // store the sub-matrix of B
      __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
      
      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      for (int s = 0; s < c_in; s += BLOCK_SIZE) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix

        // Kernel weight to Bs
        Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
          *(kw_ptr + c_out * (s + ty) + cx) : 
          padding;
        
        // Input feature to As
        for (int n = 0; n < N_LOOP; n++){

          int y_temp = y + n * BLOCK_SIZE;

          // The thread deals with the x-th channel of the y-th output
          int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

          As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
            in_f[c_in * in_row + s + tx] : 
            padding;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
    #pragma unroll 
        for (int n = 0; n < N_LOOP; n++){
    #pragma unroll
          for (int k = 0; k < BLOCK_SIZE; ++k) {
            // half Ast = As[n][ty][k];
            // for (int c = 0; c < 2; c++){
              Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
            // }
          }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }

      // Write the block sub-matrix to device memory;
      // each thread writes one element
    #pragma unroll
      for (int n = 0; n < N_LOOP; n++){
        int y_temp = y + n * BLOCK_SIZE;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
        if (out_row > -1 && cx < c_out){
          // for (int c = 0; c < 2; c++){
          atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
          // }
        }
      }
    }
  #else
    #pragma message("FP16 kernels will not be compiled.")
  #endif
}


template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol,
                const int margin,  
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap,
                const int tiles_y,
                bool halfen) {
  #if __CUDA_ARCH__ >= 700
    // Block index
    const int bx = blockIdx.x;

    const int by0 = blockIdx.y;
    const int by_stride = gridDim.y;

    for (int byi = by0; byi < tiles_y; byi += by_stride) {

        auto by = byi;

        int widx;
        const half *kw_ptr;

        const int* imap_ptr;
        const int* omap_ptr;

        if(by < margin){
            imap_ptr = imap;
            omap_ptr = omap;
            auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

            widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
            kw_ptr = &kw[widx * c_in * c_out];    
        }

        else{
            imap_ptr = omap;
            omap_ptr = imap;
            by -= margin;
            auto search_vol = halfen ? (k_vol+1)/2 : k_vol;

            widx = binary_search(qkpos,
                              by * N_LOOP * BLOCK_SIZE,
                              0,
                              search_vol);
            kw_ptr = &kw[(k_vol - widx - 1) * c_in * c_out];    
        }

      // Thread index
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int ctx = tx << 1;

      // Coordinate. x is for rows, y is for columns.
      const int cx = BLOCK_SIZE * bx + ctx;
      const int y = BLOCK_SIZE * N_LOOP * by + ty 
        - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

      // Csub is used to store the element of the block sub-matrix
      // that is computed by the thread
      half Csub[N_LOOP][2] = {__float2half(0.0f)};
      half padding[2] = {__float2half(0.0f)};

      // Declaration of the shared memory array As used to
      // store the sub-matrix of A
      __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

      // Declaration of the shared memory array Bs used to
      // store the sub-matrix of B
      __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
      
      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      for (int s = 0; s < c_in; s += BLOCK_SIZE) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix

        // Kernel weight to Bs
        *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
          *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
          *((float*)(&padding[0]));
        
        // Input feature to As
        for (int n = 0; n < N_LOOP; n++){

          int y_temp = y + n * BLOCK_SIZE;

          // The thread deals with the x-th channel of the y-th output
          int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap_ptr[y_temp] : -1;

          *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
            *((float*)(&in_f[c_in * in_row + s + ctx])) : 
            *((float*)(&padding[0]));
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
    #pragma unroll 
        for (int n = 0; n < N_LOOP; n++){
    #pragma unroll
          for (int k = 0; k < BLOCK_SIZE; ++k) {
            half Ast = As[n][ty][k];
    #pragma unroll
            for (int c = 0; c < 2; c++){
              Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
            }
          }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }

      // Write the block sub-matrix to device memory;
      // each thread writes one element
    #pragma unroll
      for (int n = 0; n < N_LOOP; n++){
        int y_temp = y + n * BLOCK_SIZE;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap_ptr[y_temp] : -1;
        if (out_row > -1 && cx < c_out){
    #pragma unroll
          for (int c = 0; c < 2; c++){
            atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
          }
        }
      }
    }
  #else
    #pragma message("FP16 kernels will not be compiled.")
  #endif
  }

}    