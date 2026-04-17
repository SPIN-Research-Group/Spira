from typing import Callable

import torch

from spira.tensors import SparseTensor

__all__ = ["fapply"]


def fapply(
    input: SparseTensor, fn: Callable[..., torch.Tensor], *args, **kwargs
) -> SparseTensor:
    feats = fn(input._features, *args, **kwargs)
    output = SparseTensor(
        features=feats,
        coordinates=input._coordinates,
        stride=input._stride,
    )
    return output