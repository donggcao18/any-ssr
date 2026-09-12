#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
