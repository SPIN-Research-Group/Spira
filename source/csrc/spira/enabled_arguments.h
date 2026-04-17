#pragma once

#define SPIRA_FOR_ALL_C_TYPES(MACRO, ...) \
  MACRO(__VA_ARGS__, std::uint64_t);      \
  MACRO(__VA_ARGS__, std::uint32_t)

#define SPIRA_FOR_ALL_F_TYPES(MACRO, ...) \
  MACRO(__VA_ARGS__, float);              \
  MACRO(__VA_ARGS__, __half)

