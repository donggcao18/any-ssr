#!/usr/bin/env bash
# Run in the environment installed from Self-Distillation/requirements.txt.
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR=.
fi
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# NUM_TRAIN can be one count or eight comma-separated counts in the order below:
# CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST
# CLI arguments (including --dry_run) are forwarded to the Python runner.
exec "${PYTHON_BIN:-python}" "$REPO_ROOT/Self-Distillation/run_codetask_sequential.py" \
    --model_name "${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B-Instruct}" \
    --output_dir "${OUTPUT_DIR:-$REPO_ROOT/outputs/sdft_codetask}" \
    --num_train "${NUM_TRAIN:-1000}" \
    --seed "${SEED:-42}" \
    "$@"
