#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results

if [[ "${CONDA_DEFAULT_ENV:-}" != "cat-sam" ]]; then
  echo "WARNING: expected conda environment cat-sam, current=${CONDA_DEFAULT_ENV:-<not set>}"
fi

NUM_SAMPLES="${NUM_SAMPLES:-5}"
python scripts/run_self_consistency.py \
  --input data/vqav2/subsets/vqav2_val_100.jsonl \
  --out results/qwen3b_self_consistency_100.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --cache-dir data/models/huggingface \
  --num-samples "$NUM_SAMPLES" \
  --max-new-tokens 32 \
  --log-file logs/qwen3b_self_consistency_100.log \
  2>&1 | tee -a logs/qwen3b_self_consistency_100.log

