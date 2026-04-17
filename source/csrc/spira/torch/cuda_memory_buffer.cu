#include "spira/cuda/buffer_pool.cuh"
#include "spira/torch/cuda_common.cuh"

namespace spira {


void CUDAFreeBuffers() {
  auto stream = c10::cuda::getCurrentCUDAStream();
  cuda::BufferPool::Global(stream.stream()).FreeBuffers();
}

void CUDAResetError() { cudaGetLastError(); }


SPIRA_TORCH_REGISTER(cuda_free_buffers, CUDAFreeBuffers);
SPIRA_TORCH_REGISTER(cuda_reset_error, CUDAResetError);

}