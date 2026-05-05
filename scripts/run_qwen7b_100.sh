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
set +e
"$PYTHON" scripts/run_answering.py \
  --input data/vqav2/subsets/vqav2_val_100.jsonl \
  --out results/qwen7b_direct_100.jsonl \
  --method direct \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --cache-dir data/models/huggingface \
  --torch-dtype float16 \
  --max-new-tokens 32 \
  --log-file logs/qwen7b_direct_100.log \
  2>&1 | tee -a logs/qwen7b_direct_100.log
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -ne 0 ]]; then
  echo "Qwen2.5-VL-7B run failed. If this is CUDA OOM, use the 3B config or retry with --load-in-4bit."
  exit "$STATUS"
fi
