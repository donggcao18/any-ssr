#!/bin/bash
# Train SeqSSR-LoRA on the executable benchmark (9 tasks, one per deepspeed run).
#
# Usage:
#   bash scripts/executable/train_seqssr_lora_executable.sh
#
# Optional overrides (set before calling the script):
#   ALPHA              mixing coefficient in [0,1]  (default: 0.5)
#   GPU_IDS            comma-separated CUDA device IDs (default: 0,1,2,3)
#   MODEL              HF model name or path        (default: Qwen/Qwen2.5-Coder-1.5B)
#   OUTPUT_DIR         where to save checkpoints    (default: ./output_models/SeqSSRLoRA_Qwen2.5-Coder-1.5B_executable)
#   HF_MODEL_REPO_ID   HF repo for upload           (default: ankhanhtran02/SeqSSRLoRA_Qwen2.5-Coder-1.5B_executable)
#   START_TASK_ID      resume from this task index  (default: 0)

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

set -euo pipefail

ALPHA="${ALPHA:-0.75}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_models/SeqSSRLoRA_Qwen2.5-Coder-1.5B_executable_alpha_${ALPHA}}"
START_TASK_ID="${START_TASK_ID:-4}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Executable benchmark tasks (order must match AllDatasetNameExecutable in params.py)
TASKS=(python cpp swift rust csharp java php typescript shell)
NUM_TASKS=${#TASKS[@]}  # 9

for (( t=START_TASK_ID; t<NUM_TASKS; t++ )); do
  port=$(shuf -i25000-30000 -n1)
  echo "========================================================"
  echo " SeqSSR-LoRA  task $t / $((NUM_TASKS-1)): ${TASKS[$t]}"
  echo " alpha=$ALPHA  output=$OUTPUT_DIR"
  echo "========================================================"

  deepspeed --master_port "$port" training/main_anamoe.py \
    --data_path "" \
    --dataset_name all \
    --benchmark executable \
    --model_name_or_path "$MODEL" \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --learning_rate 1e-4 \
    --CL_method seqssr_lora \
    --alpha "$ALPHA" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --temperature 0.2 \
    --top_p 0.95 \
    --repetition_penalty 1 \
    --do_sample \
    --num_train -1 \
    --num_eval 3 \
    --num_test 1 \
    --run_name "run_task${t}" \
    --group_name "SeqSSRLoRA_a${ALPHA}_${MODEL##*/}_executable" \
    --max_prompt_len 1024,1024,1024,1024,1024,1024,1024,1024,1024 \
    --max_ans_len    2048,2048,2048,2048,2048,2048,2048,2048,2048 \
    --num_train_epochs 3,3,3,3,3,3,3,3,3 \
    --start_task_id "$t" \
    --start_layer 4

done

: "${HF_MODEL_REPO_ID:=ankhanhtran02/SeqSSRLoRA_Qwen2.5-Coder-1.5B_executable_alpha_${ALPHA}}"

python upload_output_to_hf.py \
  --output-dir "$OUTPUT_DIR" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload SeqSSR-LoRA (alpha=${ALPHA}) executable outputs"
