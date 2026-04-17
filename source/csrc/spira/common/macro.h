#pragma once

#ifdef __NVCC__
#define SPIRA_HOST_DEVICE __host__ __device__
#define SPIRA_FORCEINLINE __forceinline__
#define SPIRA_DEVICE __device__
#define SPIRA_HOST __host__
#define SPIRA_RESTRICT __restrict__
#else
#define SPIRA_HOST_DEVICE
#define SPIRA_FORCEINLINE inline __attribute__((always_inline))
#define SPIRA_RESTIRCT __restrict
#endif

#define SPIRA_COMMA ,
#define SPIRA_SEMICOLON ;

namespace spira {

typedef unsigned int UIter;
typedef int Iter;

}  // namespace spira
