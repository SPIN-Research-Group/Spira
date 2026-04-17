import os
import numpy as np
from tqdm import tqdm
from helpers import sparse_quantize  

DATA_DIR = "scannet"            
VOXEL_DIR = "scannet_voxels"  
os.makedirs(VOXEL_DIR, exist_ok=True)

# Collect exactly the files your bash script created
coord_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.startswith("coord_") and f.endswith(".npy")
])

print(f"Found {len(coord_files)} ScanNet scenes")

for idx, fname in enumerate(tqdm(coord_files, desc="Voxelizing ScanNet")):
    path = os.path.join(DATA_DIR, fname)

    # Load coordinates (N, 3)
    xyz = np.load(path)

    voxel_size = (0.02, 0.02, 0.02)  # example, adjust as needed
    voxel_coords = sparse_quantize(xyz, voxel_size)

    out_name = f"scannet_{idx:03d}_voxels.npy"
    np.save(os.path.join(VOXEL_DIR, out_name), voxel_coords)

