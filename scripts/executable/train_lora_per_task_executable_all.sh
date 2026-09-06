#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=1,2,3

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

# Executable benchmark tasks: python, cpp, swift, rust, csharp, java, php, typescript, shell
# Per-language max_prompt_len / max_ans_len.

declare -A MAX_PROMPT_LEN=(
  [python]=1024 [cpp]=1024 [swift]=1024 [rust]=1024 [csharp]=1024
  [java]=1024 [php]=1024 [typescript]=1024 [shell]=1024
)
declare -A MAX_ANS_LEN=(
  [python]=1024 [cpp]=1024 [swift]=1024 [rust]=1024 [csharp]=1024
  [java]=1024 [php]=1024 [typescript]=1024 [shell]=1024
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
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method anamoe \
    --repetition_penalty 1 \
    --output_dir "./output_models/lora_per_task_executable_start_4_perm_1/${dataset}" \
    --run_name "anamoe_${dataset}" \
    --group_name "anamoe_executable_all" \
    --num_eval 3 \
    --num_test -1 \
    --num_train -1 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --start_layer 4 \
    --num_return_sequences 1

done


: "${HF_MODEL_REPO_ID:=ankhanhtran02/lora-per-task-executable-start-4_perm-1}"

python upload_output_to_hf.py \
  --output-dir "./output_models/lora_per_task_executable_start_4_perm_1" \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload LoRA per-task executable outputs"
