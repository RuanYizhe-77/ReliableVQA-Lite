#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "cat-sam" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate cat-sam || true
  fi
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "cat-sam" ]]; then
  echo "WARNING: expected conda environment cat-sam, current=${CONDA_DEFAULT_ENV:-<not set>}"
fi

mkdir -p logs results
python scripts/run_answering.py \
  --input data/vqav2/subsets/vqav2_val_10.jsonl \
  --out results/qwen3b_direct_10.jsonl \
  --method direct \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --cache-dir data/models/huggingface \
  --max-new-tokens 32 \
  --log-file logs/qwen3b_direct_10.log \
  2>&1 | tee -a logs/qwen3b_direct_10.log

