#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results

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

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON="$CONDA_PREFIX/bin/python"
else
  PYTHON="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
"$PYTHON" scripts/run_self_consistency.py \
  --input data/vqav2/subsets/vqav2_val_100.jsonl \
  --out results/qwen3b_self_consistency_100.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --cache-dir data/models/huggingface \
  --torch-dtype float16 \
  --num-samples "$NUM_SAMPLES" \
  --max-new-tokens 32 \
  --log-file logs/qwen3b_self_consistency_100.log \
  2>&1 | tee -a logs/qwen3b_self_consistency_100.log
