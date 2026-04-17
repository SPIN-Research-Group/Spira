#pragma once

#include <stdexcept>

#include "spira/common/macro.h"
#include "spira/common/stringify.h"

namespace spira {
class SpiraException : public std::runtime_error {
 public:
  explicit SpiraException(const std::string &message)
      : std::runtime_error("[Spira] " + message) {}
};

#define SPIRA_ERROR(...)                                         \
  do {                                                            \
    std::string message =                                         \
        "[" __FILE__ " (" SPIRA_MACRO_STRINGIFY(__LINE__) ")] "; \
    message += Stringify(__VA_ARGS__);                            \
    throw SpiraException(message);                               \
  } while (false)

#define SPIRA_CHECK(COND, ...)  \
  do {                           \
    if (!(COND)) {               \
      SPIRA_ERROR(__VA_ARGS__); \
    }                            \
  } while (false)

#ifdef __NVCC__
#define SPIRA_CHECK_CUDA(stmt)                            \
  do {                                                     \
    auto return_code = (stmt);                             \
    SPIRA_CHECK(return_code == cudaSuccess, "[",          \
                 cudaGetErrorName(return_code), "]", ": ", \
                 cudaGetErrorString(return_code));         \
  } while (false)

#define SPIRA_CHECK_CUDA_WITH_HINT(stmt, ...)                              \
  do {                                                                      \
    auto return_code = (stmt);                                              \
    SPIRA_CHECK(return_code == cudaSuccess, cudaGetErrorName(return_code), \
                 cudaGetErrorString(return_code), " -- ", __VA_ARGS__);     \
  } while (false)
#endif

}  // namespace spira
