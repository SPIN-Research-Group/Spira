#!/bin/bash
set -euo pipefail

# ============================================================
# Step 1: Check Hugging Face token
# ============================================================
if [ -z "${HF_TOKEN:-}" ]; then
  echo "❌ Please set HF_TOKEN environment variable first:"
  echo "   export HF_TOKEN='hf_your_token_here'"
  exit 1
fi

# ============================================================
# Step 2: Download ScanNet
# ============================================================
DATA_URL="https://huggingface.co/datasets/Pointcept/scannet-compressed/resolve/main/scannet.tar.gz?download=true"
OUT_TAR="scannet.tar.gz"
SCANNET_DIR="scannet"

echo "📥 Downloading ScanNet dataset..."
curl -L -H "Authorization: Bearer $HF_TOKEN" "$DATA_URL" -o "$OUT_TAR"

# ============================================================
# Step 3: Extract dataset
# ============================================================
echo "📦 Extracting dataset..."
mkdir -p "$SCANNET_DIR"
tar -xzf "$OUT_TAR" -C "$SCANNET_DIR"

rm -f "$OUT_TAR"
echo "✅ Dataset extracted to $SCANNET_DIR"

# ============================================================
# Step 4: Collect coord.npy from test scenes
# ============================================================
TEST_DIR="$SCANNET_DIR/test"

if [ ! -d "$TEST_DIR" ]; then
    echo "❌ Test folder not found at $TEST_DIR"
    exit 1
fi

echo "📂 Preparing ScanNet test coords..."

COUNT=0
for scene in $(ls -1 "$TEST_DIR" | sort | head -n 100); do
    SRC="$TEST_DIR/$scene/coord.npy"
    DST="$SCANNET_DIR/coord_$(printf "%03d" "$COUNT").npy"

    if [ ! -f "$SRC" ]; then
        echo "⚠️  Missing coord.npy in $scene — skipping"
        continue
    fi

    cp "$SRC" "$DST"
    COUNT=$((COUNT + 1))
done

echo "✅ Copied $COUNT coord.npy files to $SCANNET_DIR"

# ============================================================
# Step 5: Cleanup
# ============================================================
echo "🧹 Cleaning up unused data..."
rm -rf "$SCANNET_DIR/train"
rm -rf "$SCANNET_DIR/test"
rm -rf "$SCANNET_DIR/val"
rm -rf "$SCANNET_DIR/tasks"

python3 prepare_scannet.py

rm -rf "$SCANNET_DIR"