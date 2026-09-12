#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Hub cache directory containing models--Qwen--Qwen2.5-Coder-1.5B.
# Change this default if the downloaded cache is stored elsewhere.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/users/congthanh_le/scratch/east/CodeGR/Dense/any-ssr/.cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/home/users/congthanh_le/scratch/east/CodeGR/Dense/any-ssr/.cache
SOURCE_ARGS=()
if [[ -n "${SOURCE_CHECKPOINT:-}" ]]; then
    SOURCE_ARGS=(--source_checkpoint "$SOURCE_CHECKPOINT")
fi

exec "${PYTHON_BIN:-python}" "$REPO_ROOT/Self-Distillation/run_codetask_pairwise.py" \
  "${SOURCE_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR:-$REPO_ROOT/outputs/sdft_pairwise_codetrans}" \
  --num_train "${NUM_TRAIN:-20000}" \
  --num_validation "${NUM_VALIDATION:-1000}" \
  --num_test "${NUM_TEST:-2000}" \
  --seed "${SEED:-42}" \
  --eval_seed "${EVAL_SEED:-1234}" \
  "$@"
