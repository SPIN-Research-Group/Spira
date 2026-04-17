from typing import List

import torch

from spira import SparseTensor

__all__ = ["cat"]

#for skip connections

def cat(inputs: List[SparseTensor]) -> SparseTensor:
    feats = torch.cat([input._features for input in inputs], dim=1)
    return SparseTensor(features=feats, coordinates=inputs[0]._coordinates, stride=inputs[0]._stride)
