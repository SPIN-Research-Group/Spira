#!/usr/bin/env bash
set -e

# -------------------------------
# Config
# -------------------------------
: "${URL:?Please export the KITTI URL first. Example: export URL=.../data_odometry_velodyne.zip}"
ZIP="data_odometry_velodyne.zip"
OUT="kitti_subset"
KITTI_DIR="kitti"
SCENES=("00" "03")
MAX_FRAMES=250

# -------------------------------
# Step 1: Download KITTI if needed
# -------------------------------
if [ -f "$ZIP" ]; then
    echo "[1/6] $ZIP already exists, skipping download."
else
    echo "[1/6] Downloading dataset from $URL..."
    wget -O "$ZIP" "$URL"
fi

# -------------------------------
# Step 2: Create output folders
# -------------------------------
echo "[2/6] Creating directories..."
mkdir -p "$OUT"
mkdir -p "$KITTI_DIR"

# -------------------------------
# Step 3: Extract only scenes 00 and 03
# -------------------------------
for s in "${SCENES[@]}"; do
    # Check for scene inside ZIP (match the common KITTI structure)
    if unzip -l "$ZIP" | grep -q "dataset/sequences/$s/"; then
        echo "Extracting scene $s..."
        unzip -q "$ZIP" "dataset/sequences/$s/*" -d "$OUT"
    else
        echo "Warning: scene $s not found in ZIP, skipping."
    fi
done

# -------------------------------
# Step 4: Keep only first MAX_FRAMES frames per scene
# -------------------------------
echo "[4/6] Moving first $MAX_FRAMES frames per scene to $KITTI_DIR..."
COUNT=0
for s in "${SCENES[@]}"; do
    VEL_DIR="$OUT/dataset/sequences/$s/velodyne"
    i=0
    for f in $(ls "$VEL_DIR"/*.bin | sort); do
        if [ "$i" -ge "$MAX_FRAMES" ]; then
            break
        fi
        cp "$f" "$KITTI_DIR/coord_$(printf "%05d" "$COUNT").bin"
        COUNT=$((COUNT + 1))
        i=$((i + 1))
    done
done

echo "✅ Total $COUNT .bin files copied to $KITTI_DIR"

# -------------------------------
# Step 5: Run prepare_kitti.py
# -------------------------------
echo "[5/6] Running prepare_kitti.py..."
python3 prepare_kitti.py

# -------------------------------
# Step 6: Cleanup
# -------------------------------
echo "[6/6] Cleaning up temporary files..."
rm -f "$ZIP"      # always delete the ZIP
rm -rf "$OUT"     # delete extracted subset folder
rm -rf "$KITTI_DIR"

echo "✅ Cleanup done."
