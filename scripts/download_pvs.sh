#!/usr/bin/env bash
# Download PVS dataset CSVs from the UFSC Seafile mirror (no auth required).
# Skips the ~50 GB of video files the fault-detection pipeline never uses.
TOKEN=55ee83c2c9d647eaa252
BASE="https://arquivos.ufsc.br/d/$TOKEN/files/?p="
DEST="${PVS_DATA:-/c/PVS_Dataset}"
FILES=(dataset_mpu_left.csv dataset_mpu_right.csv dataset_labels.csv dataset_settings_left.csv dataset_settings_right.csv)

for i in $(seq 1 9); do
  mkdir -p "$DEST/PVS$i"
  for f in "${FILES[@]}"; do
    out="$DEST/PVS$i/$f"
    if [ -s "$out" ]; then echo "[skip] PVS$i/$f"; continue; fi
    echo "[get ] PVS$i/$f"
    curl -sL --retry 5 --retry-delay 3 --retry-all-errors -C - \
         -o "$out" "${BASE}/PVS%20$i/$f&dl=1" \
      || { echo "[FAIL] PVS$i/$f"; continue; }
    echo "       -> $(du -h "$out" | cut -f1)"
  done
done
echo "=== DONE ==="
du -sh "$DEST"
