#!/usr/bin/env bash

echo "🔐 Step 1: Authenticating..."
# This opens the link for you to log in to your Google account
gcloud auth login --no-launch-browser

echo "📥 Step 2: Downloading Waymo data (this may take a while)..."
# We download directly to the current folder
gsutil -m cp gs://waymo_open_dataset_v_1_4_3/archived_files/testing/testing_0000.tar .


OUTDIR="waymo"
TAR_FILE="testing_0000.tar"

echo "📦 Step 3: Extracting files"
mkdir -p "$OUTDIR"

# '2>/dev/null' hides all the 'Unexpected EOF' and 'Error' messages
tar -xif "$TAR_FILE" -C "$OUTDIR" 2>/dev/null

echo "🧹 Step 4: Filtering to keep only the first 5 sequences..."
cd "$OUTDIR"
# Hiding errors here too in case there are no files to delete
ls -1 *.tfrecord 2>/dev/null | sort | tail -n +6 | xargs rm -f 2>/dev/null
cd ..

echo "🗑️ Step 5: Removing the large tar file..."
rm -f "$TAR_FILE"

echo "✅ Done! Files extracted to $OUTDIR."
ls -1 "$OUTDIR"

python3 prepare_waymo.py

rm -rf "$OUTDIR"