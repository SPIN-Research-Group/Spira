#pragma once

#include "spira/cuda/context.cuh"
#include "spira/torch/common.h"

namespace spira {

cuda::Context GetCUDAContextFromTorchCUDADevice(const torch::Device &device);

}  // namespace spira
