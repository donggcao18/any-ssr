#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# Tune these defaults here; the command stays the same for one or more GPUs.
NUM_GPUS="${NUM_GPUS:-2}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_LIST=0
    for ((gpu=1; gpu<NUM_GPUS; gpu++)); do GPU_LIST+=",$gpu"; done
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
# Effective batch = NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
EMA_ALPHA="${EMA_ALPHA:-0.01}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
VLLM_MEMORY_FRACTION="${VLLM_MEMORY_FRACTION:-0.3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
SAVE_STEPS="${SAVE_STEPS:-100}"
# Server settings. Both caches were configured in the repository's .cache.
# HF_HUB_CACHE must contain models--Qwen--Qwen2.5-Coder-1.5B;
# HF_DATASETS_CACHE must contain the prepared CodeTask dataset cache.
CACHE_ROOT="/home/users/congthanh_le/scratch/east/CodeGR/Dense/any-ssr/.cache"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/home/users/congthanh_le/scratch/east/CodeGR/Dense/any-ssr/anamoe/CodeTrans/0}"
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
