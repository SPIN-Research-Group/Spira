import glob
import numpy as np
import torch
import os

def make_dataset(path):
    dataset = []

    if os.path.isfile(path) and path.endswith(".npy"):
        npy_files = [path]
    elif os.path.isdir(path):
        npy_files = glob.glob(os.path.join(path, "*.npy"))
    else:
        raise ValueError(f"Invalid path: {path}. Must be a folder or a .npy file.")

    for file in npy_files:
        tensor_data_np = np.load(file)
        np.random.shuffle(tensor_data_np)
        tensor_data = torch.tensor(tensor_data_np, dtype=torch.int32, device="cuda")

        min_coords = torch.min(tensor_data, dim=0).values
        sources = tensor_data - min_coords
        dataset.append(sources)

    return dataset


def torchsparse_dataloader(dataset, in_channels, sort_coordinates=False):
  from torchsparse import SparseTensor
  for td in dataset:

    coords = td[:, [0, 2, 1]]

    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device="cuda")
    coords = torch.cat([batch, coords], dim=1)

    feats = torch.randn((coords.shape[0], in_channels), dtype=torch.float16, device='cuda')

    coords = coords.contiguous()
    feats = feats.contiguous()

    yield SparseTensor(feats=feats, coords=coords)


def minuet_dataloader(dataset, in_channels, sort_coordinates=True):
  from minuet import SparseTensor
  import minuet
  for coords in dataset:
    coords = coords
    feats = torch.randn((coords.shape[0], in_channels), dtype=torch.float16, device='cuda')
    coords = coords.contiguous()
    feats = feats.contiguous()

    if sort_coordinates:
        index = minuet.nn.functional.build_sorted_index(coords)
        coords = coords[index].contiguous()
        feats = feats[index].contiguous()
    
    yield SparseTensor(coordinates=coords, features=feats)


def spira_dataloader(dataset, in_channels, sort_coordinates=True):
  from spira import SparseTensor
  for coords in dataset:
    coords = coords
    feats = torch.randn((coords.shape[0], in_channels), dtype=torch.float16, device='cuda')
    coords = coords.contiguous()
    feats = feats.contiguous()

    x = SparseTensor(features=feats, coordinates=coords, double=False)
    
    if sort_coordinates:

        x = x.order()

    yield x


def spira_64_dataloader(dataset, in_channels, sort_coordinates=True):
  from spira import SparseTensor
  for coords in dataset:
    coords = coords
    feats = torch.randn((coords.shape[0], in_channels), dtype=torch.float16, device='cuda')
    coords = coords.contiguous()
    feats = feats.contiguous()

    x = SparseTensor(features=feats, coordinates=coords, double=True)
    
    if sort_coordinates:

        x = x.order()

    yield x