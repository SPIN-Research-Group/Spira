import os
import numpy as np
from tqdm import tqdm
from helpers import sparse_quantize

DATA_DIR = "kitti"
VOXEL_DIR = "kitti_voxels"
os.makedirs(VOXEL_DIR, exist_ok=True)

# Collect .bin files
bin_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.startswith("coord_") and f.endswith(".bin")
])

print(f"Found {len(bin_files)} KITTI frames")

for idx, fname in enumerate(tqdm(bin_files, desc="Voxelizing KITTI")):
    path = os.path.join(DATA_DIR, fname)

    # Load Velodyne binary (N,4) and keep only XYZ
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

    min_bound = np.array([-60, 50, -4])
    max_bound = np.array([60, 50, 2])
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    xyz = points[mask]

    voxel_size = (0.1, 0.1, 0.1)
    voxel_coords = sparse_quantize(points, voxel_size)

    out_name = f"kitti_{idx:05d}_voxels.npy"
    np.save(os.path.join(VOXEL_DIR, out_name), voxel_coords)