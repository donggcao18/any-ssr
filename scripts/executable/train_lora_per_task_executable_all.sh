#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=5,6,7

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

# Executable benchmark tasks: python, cpp, swift, rust, csharp, java, php, typescript, shell
# Per-language max_prompt_len / max_ans_len.

declare -A MAX_PROMPT_LEN=(
  [python]=1400 [cpp]=1300 [swift]=1400 [rust]=1300 [csharp]=1200
  [java]=1300 [php]=1000 [typescript]=1900 [shell]=1500
)
declare -A MAX_ANS_LEN=(
  [python]=1700 [cpp]=1700 [swift]=2000 [rust]=4000 [csharp]=1600
  [java]=1700 [php]=1500 [typescript]=1700 [shell]=1900
)

for dataset in python cpp swift rust csharp java php typescript shell; do
  deepspeed --master_port "$port" training/main_anamoe.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
    --benchmark executable \
    --data_path "" \
    --dataset_name "$dataset" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 11 \
    --max_prompt_len "${MAX_PROMPT_LEN[$dataset]}" \
    --max_ans_len "${MAX_ANS_LEN[$dataset]}" \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method anamoe \
    --repetition_penalty 1 \
    --output_dir "./output_models/lora_per_task_executable_start_4/${dataset}" \
    --run_name "anamoe_${dataset}" \
    --group_name "anamoe_executable_all" \
    --num_train -1 \
    --num_eval 3 \
    --num_test -1 \
    --logging_steps 10 \
    --start_layer 4 \

done

: "${HF_MODEL_REPO_ID:=ankhanhtran02/lora-per-task-executable-start-4}"

python upload_output_to_hf.py \
  --output-dir "./output_models/lora_per_task_executable_start_4" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload LoRA per-task executable outputs"
