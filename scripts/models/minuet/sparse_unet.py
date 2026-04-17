#ref: https://github.com/Kipsora/MinuetArtifacts/blob/bf088077bc5c1019f9b04ca73b31813699a158e1/utils/models/minuet/sparse_unet.py

__all__ = ['SparseResUNet42']

from typing import List

from minuet import nn as spnn, SparseTensor
from minuet.nn import functional as MF
from torch import nn
from torch.nn import Module, Sequential

from models.minuet.common import SparseConvBNReLU, SparseResidualBlock


class SparseResUNet(Module):

  def __init__(
      self,
      stem_channels: int,
      encoder_channels: List[int],
      decoder_channels: List[int],
      *,
      in_channels: int = 4,
      width_multiplier: float = 1.0,
  ) -> None:
    super().__init__()
    self.stem_channels = stem_channels
    self.encoder_channels = encoder_channels
    self.decoder_channels = decoder_channels
    self.in_channels = in_channels
    self.width_multiplier = width_multiplier

    num_channels = [stem_channels] + encoder_channels + decoder_channels
    num_channels = [int(width_multiplier * nc) for nc in num_channels]

    self.stem = Sequential(
        spnn.SparseConv3d(in_channels=in_channels,
                          out_channels=num_channels[0],
                          kernel_size=3),
        spnn.BatchNorm(num_features=num_channels[0]),
        spnn.ReLU(True),
        spnn.SparseConv3d(in_channels=num_channels[0],
                          out_channels=num_channels[0],
                          kernel_size=3),
        spnn.BatchNorm(num_features=num_channels[0]),
        spnn.ReLU(True),
    )

    self.encoders = nn.ModuleList()
    for k in range(4):
      self.encoders.append(
          nn.Sequential(
              SparseConvBNReLU(
                  ndim=3,
                  in_channels=num_channels[k],
                  out_channels=num_channels[k],
                  kernel_size=2,
                  stride=2,
              ),
              SparseResidualBlock(ndim=3,
                                  in_channels=num_channels[k],
                                  out_channels=num_channels[k + 1],
                                  kernel_size=3),
              SparseResidualBlock(ndim=3,
                                  in_channels=num_channels[k + 1],
                                  out_channels=num_channels[k + 1],
                                  kernel_size=3),
          ))

    self.decoders = nn.ModuleList()
    for k in range(4):
      self.decoders.append(
          nn.ModuleDict({
              'upsample':
                  SparseConvBNReLU(ndim=3,
                                   in_channels=num_channels[k + 4],
                                   out_channels=num_channels[k + 5],
                                   kernel_size=2,
                                   stride=2,
                                   transposed=True),
              'fuse':
                  nn.Sequential(
                      SparseResidualBlock(ndim=3,
                                          in_channels=num_channels[k + 5]
                                          + num_channels[3 - k],
                                          out_channels=num_channels[k + 5],
                                          kernel_size=3),
                      SparseResidualBlock(ndim=3,
                                          in_channels=num_channels[k + 5],
                                          out_channels=num_channels[k + 5],
                                          kernel_size=3),
                  )
          }))

  def _unet_forward(
      self,
      x: SparseTensor,
      encoders: nn.ModuleList,
      decoders: nn.ModuleList,
  ) -> List[SparseTensor]:
    if not encoders and not decoders:
      return [x]

    # downsample
    xd = encoders[0](x)

    # inner recursion
    outputs = self._unet_forward(xd, encoders[1:], decoders[:-1])
    yd = outputs[-1]

    # upsample and fuse
    u = decoders[-1]['upsample'](yd)
    y = decoders[-1]['fuse'](MF.cat([u, x]))

    return [x] + outputs + [y]

  def forward(self, x: SparseTensor) -> List[SparseTensor]:
    return self._unet_forward(self.stem(x), self.encoders, self.decoders)


class SparseResUNet42(SparseResUNet):

  def __init__(self, **kwargs) -> None:
    super().__init__(
        stem_channels=32,
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=[256, 128, 96, 96],
        **kwargs,
    )