import numpy as np
from itertools import repeat
from typing import Union
import torch
import random


def ravel_hash(x: np.ndarray) -> np.ndarray:
    """Hash 3D integer coordinates into 1D for uniqueness."""
    assert x.ndim == 2 and x.shape[1] == 3
    x = x.astype(np.uint64)
    return x[:, 0] * 73856093 ^ x[:, 1] * 19349663 ^ x[:, 2] * 83492791

def sparse_quantize(
    coords: np.ndarray,
    voxel_size: Union[float, tuple[float, float, float]] = 1.0
) -> np.ndarray:
    """Voxelize coordinates: floor division, remove duplicates, sort."""
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    voxel_size_np = np.array(voxel_size)
    coords = np.floor(coords / voxel_size_np).astype(np.int32)

    # remove duplicates using hash
    _, indices = np.unique(ravel_hash(coords), return_index=True)
    coords = coords[indices]

    # sort lexicographically x,y,z
    sorted_idx = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    coords = coords[sorted_idx]
    return coords