__all__ = ['ReLU', 'ReLU6', 'LeakyReLU']

import torch

from spira import SparseTensor
from spira.utils.apply import fapply

class ReLU(torch.nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

class ReLU6(torch.nn.ReLU6):
  def forward(self, input: SparseTensor) -> SparseTensor:
          return fapply(input, super().forward)

class LeakyReLU(torch.nn.LeakyReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)