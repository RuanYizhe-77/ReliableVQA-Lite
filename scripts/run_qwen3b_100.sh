#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results
exec > >(tee -a logs/qwen3b_100.log) 2>&1

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

"$PYTHON" scripts/run_answering.py \
  --input data/vqav2/subsets/vqav2_val_100.jsonl \
  --out results/qwen3b_direct_100.jsonl \
  --method direct \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --cache-dir data/models/huggingface \
  --torch-dtype float16 \
  --max-new-tokens 32

"$PYTHON" scripts/run_answering.py \
  --input data/vqav2/subsets/vqav2_val_100.jsonl \
  --out results/qwen3b_verbalized_100.jsonl \
  --method verbalized \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --cache-dir data/models/huggingface \
  --torch-dtype float16 \
  --max-new-tokens 96
