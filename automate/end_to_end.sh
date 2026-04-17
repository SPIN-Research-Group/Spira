#!/bin/bash

DATASETS=("datasets/waymo_voxels/" "datasets/scannet_voxels/" "datasets/kitti_voxels/")
MODELS=("ResNet" "ResNetLarge" "UNet")
LIBS=("torchsparse" "minuet" "spira" "spira_64")

mkdir -p results/end_to_end

for LIB in "${LIBS[@]}"; do
    for DATASET_PATH in "${DATASETS[@]}"; do

        DATASET_NAME=$(basename "${DATASET_PATH}" | sed 's/_voxels//')

        for MODEL in "${MODELS[@]}"; do

            out_file="results/end_to_end/${DATASET_NAME}_${MODEL}_${LIB}.out"

            echo "=================================================="
            echo "Running lib=${LIB} dataset=${DATASET_NAME} model=${MODEL}"
            echo "Output -> ${out_file}"
            echo "=================================================="

            python3 scripts/end_to_end_experiment.py \
                --library ${LIB} \
                --dataset ${DATASET_NAME} \
                --dataset_path ${DATASET_PATH} \
                --model ${MODEL} \
                > "${out_file}" 2>&1

            echo ""
        done
    done
done

python3 scripts/end_to_end_plot.py