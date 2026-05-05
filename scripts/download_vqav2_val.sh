#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs data/vqav2/raw data/vqav2/images data/vqav2/subsets
LOG="logs/download_vqav2_val.log"
exec > >(tee -a "$LOG") 2>&1

echo "ReliableVQA-Lite VQAv2 validation download"
date
echo "Working directory: $ROOT"
echo "Disk before download:"
df -h .
du -sh . data/vqav2 2>/dev/null || true

RAW="data/vqav2/raw"
IMAGE_DIR="data/vqav2/images"

download_file() {
  local url="$1"
  local out="$2"
  echo "Downloading/resuming: $url"
  echo "Output: $out"
  if command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$out"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - "$url" -o "$out"
  else
    echo "ERROR: wget or curl is required." >&2
    exit 1
  fi
}

download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" "$RAW/v2_Questions_Val_mscoco.zip"
download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" "$RAW/v2_Annotations_Val_mscoco.zip"

echo "Unzipping VQAv2 questions and annotations"
unzip -n "$RAW/v2_Questions_Val_mscoco.zip" -d "$RAW"
unzip -n "$RAW/v2_Annotations_Val_mscoco.zip" -d "$RAW"

download_file "http://images.cocodataset.org/zips/val2014.zip" "$RAW/val2014.zip"

echo "Unzipping COCO val2014 images"
unzip -n "$RAW/val2014.zip" -d "$IMAGE_DIR"

QUESTIONS="$RAW/v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS="$RAW/v2_mscoco_val2014_annotations.json"
VAL_DIR="$IMAGE_DIR/val2014"

echo "Expected paths:"
echo "Questions: $QUESTIONS"
echo "Annotations: $ANNOTATIONS"
echo "Images directory: $VAL_DIR"

test -f "$QUESTIONS"
test -f "$ANNOTATIONS"
test -d "$VAL_DIR"

echo "Image count:"
find "$VAL_DIR" -maxdepth 1 -name 'COCO_val2014_*.jpg' | wc -l

echo "Disk after download:"
df -h .
du -sh . data/vqav2 "$RAW" "$IMAGE_DIR" 2>/dev/null || true
echo "Download complete. Log: $LOG"
