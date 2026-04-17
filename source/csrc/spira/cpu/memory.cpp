#include "spira/cpu/memory.h"

namespace spira::cpu::detail {

template <>
void *AcquireMemory(std::size_t size) {
  return ::operator new(size);
}

template <>
void ReleaseMemory(void *data) noexcept {
  ::operator delete(data);
}

}  // namespace spira::cpu::detail
