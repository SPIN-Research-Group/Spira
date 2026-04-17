#pragma once

#include <cub/cub.cuh>
#include "spira/cuda/context.cuh"

namespace spira::cuda {

#define CUB_TRANSFORMED_INPUT_ITERATOR(TYPE, FUNC, NAME)         \
  cub::TransformInputIterator<TYPE, decltype(FUNC),              \
                              cub::CountingInputIterator<UIter>> \
  NAME(cub::CountingInputIterator<UIter>(0), FUNC)

template <typename FT>
void MatMul(std::size_t m,           //
            std::size_t k,           //
            std::size_t n,           //
            bool is_a_transposed,    //
            bool is_b_transposed,    //
            const FT *d_a,           //
            const FT *d_b,           //
            FT *d_c,                 //
            bool incremental,        //
            const Context &context,  //
            cudaStream_t stream = nullptr);

template <typename FT>
void BatchedMatMul(std::size_t b,           //
                   std::size_t m,           //
                   std::size_t k,           //
                   std::size_t n,           //
                   bool is_a_transposed,    //
                   bool is_b_transposed,    //
                   const FT *d_a,           //
                   const FT *d_b,           //
                   FT *d_c,                 //
                   bool incremental,        //
                   const Context &context,  //
                   cudaStream_t stream = nullptr);

}  // namespace spira::cuda