#pragma once

#include "spira/cuda/context.cuh"

namespace spira::cuda::device {

// Assume the array is non-decreasing
template <typename IT, typename SourceT>
SPIRA_FORCEINLINE SPIRA_DEVICE IT
BinarySearchMinimize(IT l, IT r, SourceT source,
                     const std::invoke_result_t<SourceT, IT> &target = 0) {
  while (l < r) {
    auto m = (l + r) >> 1;
    if (source(m) >= target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return l;
}

}  // namespace spira::cuda::device