#!/bin/bash

DATASETS=("datasets/waymo_voxels/" "datasets/scannet_voxels/" "datasets/kitti_voxels/")

CONFIGS=(
    "4 32 3"
    "16 32 3"
    "32 32 3"
    "32 64 3"
    "128 128 3"
    "16 16 5"
    "32 32 5"
    "32 64 5"
)

LIBS=("torchsparse" "minuet" "spira")

mkdir -p results/layerwise

for LIB in "${LIBS[@]}"; do
    for DATASET_PATH in "${DATASETS[@]}"; do

        DATASET_NAME=$(basename "${DATASET_PATH}" | sed 's/_voxels//')

        for CONFIG in "${CONFIGS[@]}"; do
            read -r CIN COUT K <<< "${CONFIG}"

            out_file="results/layerwise/${DATASET_NAME}_cin${CIN}_cout${COUT}_k${K}_${LIB}.out"

            echo "=================================================="
            echo "Running lib=${LIB} dataset=${DATASET_NAME} cin=${CIN} cout=${COUT} k=${K}"
            echo "Output -> ${out_file}"
            echo "=================================================="

            python3 scripts/layerwise_experiment.py \
                --library ${LIB} \
                --dataset ${DATASET_NAME} \
                --dataset_path ${DATASET_PATH} \
                --cin ${CIN} \
                --cout ${COUT} \
                --k ${K} \
                > "${out_file}" 2>&1

            echo ""
        done
    done
done

python3 scripts/layerwise_plot.py