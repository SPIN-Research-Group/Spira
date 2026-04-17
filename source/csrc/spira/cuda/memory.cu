#include "spira/cuda/memory.cuh"

namespace spira::cuda {

void *DirectMemoryFactory::Acquire(std::size_t size, std::size_t alignment) {
  void *d_data;
  SPIRA_CHECK(alignment == 0 || CUDA_DEFAULT_ALIGNMENT % alignment == 0,
               "Cannot allocate CUDA memory with alignment ", alignment);
  SPIRA_CHECK_CUDA(cudaMalloc(&d_data, size));
  return d_data;
}

void DirectMemoryFactory::Release(void *data) {
  SPIRA_CHECK_CUDA(cudaFree(data));
}

void DirectMemoryFactory::SilentRelease(void *data) noexcept { cudaFree(data); }

void *AsyncDirectMemoryFactory::Acquire(std::size_t size,
                                        std::size_t alignment) {
  SPIRA_CHECK(alignment == 0 || CUDA_DEFAULT_ALIGNMENT % alignment == 0,
               "Cannot allocate CUDA memory with alignment ", alignment);
  void *d_data;
  SPIRA_CHECK_CUDA(cudaMallocAsync(&d_data, size, stream_));
  return d_data;
}

void AsyncDirectMemoryFactory::Release(void *data) {
  SPIRA_CHECK_CUDA(cudaFreeAsync(data, stream_));
}

void AsyncDirectMemoryFactory::SilentRelease(void *data) noexcept {
  cudaFreeAsync(data, stream_);
}

}  // namespace spira::cuda