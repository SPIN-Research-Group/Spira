#!/bin/bash

SCENES=("datasets/waymo_voxels/waymo_000000_voxels.npy")

LIBS=("minuet" "torchsparse")

declare -A LIB_NAMES
LIB_NAMES=( ["minuet"]="Minuet" ["torchsparse"]="TorchSparse++")

for scene in "${SCENES[@]}"; do
    for lib in "${LIBS[@]}"; do
        display_name=${LIB_NAMES[$lib]}
        
        echo "Running Output Voxel Data Comparison on $scene against $display_name using FP16 precision"

        python3 scripts/correctness.py \
            --baseline $lib \
            --scene "$scene" 

        echo "Running Output Voxel Data Comparison on $scene against $display_name using FP32 precision"

        python3 scripts/correctness.py \
            --baseline $lib \
            --scene "$scene" \
            --fp32 
    done
done