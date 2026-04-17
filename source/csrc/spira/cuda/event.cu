#include "spira/cuda/event.cuh"

namespace spira::cuda {

void Event::Record(cudaStream_t stream) {
  SPIRA_CHECK(!closed_, "Event is invalid");
  SPIRA_CHECK_CUDA(cudaEventRecord(event_, stream));
}

void Event::Synchronize() {
  SPIRA_CHECK(!closed_, "Event is invalid");
  SPIRA_CHECK_CUDA(cudaEventSynchronize(event_));
}

float Event::Elapsed(const Event &after) const {
  float result;
  SPIRA_CHECK(!closed_, "Event is invalid");
  SPIRA_CHECK_CUDA(cudaEventElapsedTime(&result, event_, after.event()));
  return result;
}

void Event::Close() {
  if (!closed_) {
    SPIRA_CHECK_CUDA(cudaEventDestroy(event_));
    closed_ = true;
  }
}

void Event::CloseNoExcept() noexcept {
  if (!closed_) {
    cudaEventDestroy(event_);
    closed_ = true;
  }
}

}  // namespace spira::cuda