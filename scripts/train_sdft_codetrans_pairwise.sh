#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# 0 allows Hugging Face downloads; 1 uses cached files only.
HF_OFFLINE="${HF_OFFLINE:-0}"
case "$HF_OFFLINE" in
    0|1) ;;
    *) echo "HF_OFFLINE must be 0 (online) or 1 (offline)" >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE="$HF_OFFLINE"
export HF_DATASETS_OFFLINE="$HF_OFFLINE"
export TRANSFORMERS_OFFLINE="$HF_OFFLINE"
# Quadro RTX 8000 (Turing / SM 7.5): no native BF16 or FlashAttention 2.
export SDFT_PRECISION="${SDFT_PRECISION:-float16}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
# Tune these defaults here; the command stays the same for one or more GPUs.
NUM_GPUS="${NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
# Effective batch = NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
EMA_ALPHA="${EMA_ALPHA:-0.01}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
VLLM_MEMORY_FRACTION="${VLLM_MEMORY_FRACTION:-0.3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/research/cbim/vast/qt60/any-ssr/output/CodeTrans/0}"
TASKS="${TASKS:-CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST}"
# A fresh directory lets the same short command work after a failed run.
printf -v RUN_TIMESTAMP '%(%Y%m%d_%H%M%S)T' -1
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/sdft_pairwise_${RUN_TIMESTAMP}_$$}"

exec "${PYTHON_BIN:-python}" "$REPO_ROOT/Self-Distillation/run_codetask_pairwise.py" \
  --source_checkpoint "$SOURCE_CHECKPOINT" \
  --tasks "$TASKS" \
  --num_gpus "$NUM_GPUS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --warmup_ratio "$WARMUP_RATIO" \
  --ref_model_mixup_alpha "$EMA_ALPHA" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --vllm_gpu_memory_utilization "$VLLM_MEMORY_FRACTION" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --save_steps "$SAVE_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --num_train "${NUM_TRAIN:-20000}" \
  --num_validation "${NUM_VALIDATION:-1000}" \
  --num_test "${NUM_TEST:-2000}" \
  --seed "${SEED:-42}" \
  --eval_seed "${EVAL_SEED:-1234}" \
  "$@"
