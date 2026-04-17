#pragma once

#include "spira/common/exception.h"
#include "spira/cpu/memory.h"

namespace spira::cpu {

class Context {
 public:
  Context();

  template <typename T = void>
  Memory<T> NewMemory(std::size_t n) const {
    return Memory<T>(n);
  }
};

}  // namespace spira::cpu