__all__ = ['BatchNorm']

import torch

from spira import SparseTensor
from spira.utils.apply import fapply

class BatchNorm(torch.nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)