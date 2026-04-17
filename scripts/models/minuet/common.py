#ref: https://github.com/Kipsora/MinuetArtifacts/blob/bf088077bc5c1019f9b04ca73b31813699a158e1/utils/models/minuet/common.py

__all__ = ['SparseConvBNReLU', 'SparseResidualBlock']

from typing import Optional

import numpy as np
import torch
from minuet import SparseTensor
from minuet.nn import *
from minuet.utils.typing import ScalarOrTuple
from torch.nn import Sequential, Module, Identity


class SparseConvBNReLU(Sequential):

  def __init__(self,
               ndim: int,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               dilation: ScalarOrTuple[int] = 1,
               bias: bool = False,
               transposed: bool = False,
               eps: float = 1e-5,
               momentum: float = 0.1,
               affine: bool = True,
               track_running_stats: bool = True,
               use_relu: bool = True,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None):
    super().__init__(
        SparseConv(ndim=ndim,
                   in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   dilation=dilation,
                   bias=bias,
                   transposed=transposed,
                   dtype=dtype,
                   device=device),
        BatchNorm(num_features=out_channels,
                  eps=eps,
                  momentum=momentum,
                  affine=affine,
                  track_running_stats=track_running_stats,
                  dtype=dtype,
                  device=device),
    )
    if use_relu:
      self.append(ReLU(inplace=True))


class SparseResidualBlock(Module):

  def __init__(self,
               ndim: int,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               dilation: ScalarOrTuple[int] = 1,
               bias: bool = False,
               transposed: bool = False,
               eps: float = 1e-5,
               momentum: float = 0.1,
               affine: bool = True,
               track_running_stats: bool = True,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None):
    super().__init__()
    self.main = Sequential(
        SparseConv(ndim=ndim,
                   in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   dilation=dilation,
                   bias=bias,
                   transposed=transposed,
                   dtype=dtype,
                   device=device),
        BatchNorm(num_features=out_channels,
                  eps=eps,
                  momentum=momentum,
                  affine=affine,
                  track_running_stats=track_running_stats,
                  dtype=dtype,
                  device=device),
        ReLU(inplace=True),
        SparseConv(ndim=ndim,
                   in_channels=out_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   dilation=dilation,
                   bias=bias,
                   transposed=transposed,
                   dtype=dtype,
                   device=device),
    )
    if in_channels != out_channels or np.prod(stride) != 1 or \
            np.prod(dilation) != 1:
      self.shortcut = SparseConvBNReLU(ndim=ndim,
                                       in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=stride,
                                       dilation=dilation,
                                       bias=bias,
                                       transposed=transposed,
                                       eps=eps,
                                       momentum=momentum,
                                       affine=affine,
                                       track_running_stats=track_running_stats,
                                       dtype=dtype,
                                       device=device,
                                       use_relu=False)
    else:
      self.shortcut = Identity()

    self.relu = ReLU()

  def forward(self, x: SparseTensor) -> SparseTensor:
    a = self.main(x)
    b = self.shortcut(x)
    x = SparseTensor(features=a.F + b.F,
                     coordinates=a.C,
                     stride=a.stride,
                     batch_dims=a.batch_dims)
    return self.relu(x)