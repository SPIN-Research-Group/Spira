#pragma once

#include "spira/common/functions.h"

namespace spira::cuda::device {

static_assert(SizeOf<char> == 1);

template <std::size_t T_SIZE>
SPIRA_FORCEINLINE SPIRA_DEVICE void MemoryCopy(
    char *__restrict__ targets, const char *__restrict__ sources) {
  Iterate<UIter, T_SIZE>([&](auto k) { targets[k] = sources[k]; });
}

template <std::size_t T_NUM_BYTES, typename T, typename... Ts>
SPIRA_FORCEINLINE SPIRA_DEVICE void MemoryCopy(
    char *__restrict__ targets, const char *__restrict__ sources) {
  constexpr auto SIZE = T_NUM_BYTES / SizeOf<T>;
  Iterate<UIter, SIZE>([&](auto k) {
    reinterpret_cast<T *>(targets)[k] = reinterpret_cast<const T *>(sources)[k];
  });
  MemoryCopy<T_NUM_BYTES - SIZE * SizeOf<T>, Ts...>(targets + SIZE * SizeOf<T>,
                                                    sources + SIZE * SizeOf<T>);
}

template <std::size_t T_SIZE, typename T>
SPIRA_FORCEINLINE SPIRA_DEVICE void Assign(T *__restrict__ targets,
                                             const T *__restrict__ sources) {
  // We do not need type information if we only need to conduct memory copy
  MemoryCopy<T_SIZE * SizeOf<T>, int4, int3, int2, int1, short, char>(
      reinterpret_cast<char *>(targets),
      reinterpret_cast<const char *>(sources));
}

}  // namespace spira::cuda::device