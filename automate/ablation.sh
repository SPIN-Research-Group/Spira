#!/bin/bash

SCENES=("datasets/random_voxels/")
MODELS=("ResNet" "ResNetLarge" "UNet")
LIBS=("torchsparse" "minuet" "spira")

mkdir -p results/ablation

for LIB in "${LIBS[@]}"; do
    for SCENE_PATH in "${SCENES}"*.npy; do
        SCENE_FILE=$(basename "$SCENE_PATH")
        SCENE_NAME="${SCENE_FILE%.npy}"

        for MODEL in "${MODELS[@]}"; do

            out_file="results/ablation/${SCENE_NAME}_${MODEL}_${LIB}.out"

            echo "=================================================="
            echo "Running lib=${LIB} dataset=${SCENE_NAME} model=${MODEL}"
            echo "Output -> ${out_file}"
            echo "=================================================="

            python3 scripts/ablation_experiment.py \
                --library ${LIB} \
                --scene "$SCENE_PATH" \
                --model ${MODEL} \
                > "${out_file}" 2>&1

            echo ""
        done
    done
done

python3 scripts/ablation_plot.py