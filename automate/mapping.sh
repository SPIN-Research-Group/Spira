#!/bin/bash

SCENES=("datasets/waymo_voxels/waymo_000000_voxels.npy" "datasets/scannet_voxels/scannet_039_voxels.npy" "datasets/scannet_voxels/scannet_040_voxels.npy")

mkdir -p results/mapping

LIBS=("minuet" "torchsparse" "simple_bs" "spira")

KERNELS=(3 5)

for scene in "${SCENES[@]}"; do
    scene_name=$(basename "$scene" .npy)

    for k in "${KERNELS[@]}"; do
        for lib in "${LIBS[@]}"; do
            out_file="results/mapping/${scene_name}_k${k}_${lib}.out"
            echo "Running scene=$scene kernel=$k library=$lib -> $out_file"

            python3 scripts/mapping_experiment.py \
                --file "$scene" \
                --kernel_size $k \
                --library $lib \
                > "$out_file" 2>&1
        done
    done
done

python3 scripts/mapping_plot.py