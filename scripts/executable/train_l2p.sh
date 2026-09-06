#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0,4,5,6
# This script uses 1 GPU. Use a larger disk space (56GB) to save the model checkpoints (full model).

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
  --data_path /path/to/LLM-CL-Benchmark_5000 \
  --dataset_name all \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
  --benchmark executable \
  --lr_scheduler_type cosine \
  --num_warmup_steps 0 \
  --seed 1234 \
  --zero_stage 2 \
  --deepspeed \
  --print_loss \
  --offload \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --CL_method L2P \
  --output_dir ./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable_perm_1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --run_name run_1 \
  --group_name L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable_perm_1  \
  --num_train -1 \
  --num_eval 3 \
  --num_test -1 \
  --max_prompt_len 1400,1300,1400,1300,1200,1300,1000,1900,1500 \
  --max_ans_len 1700,1700,2000,4000,1600,1700,1500,1700,1900 \
  --repetition_penalty 1 \
  --num_train_epochs 3

: "${HF_MODEL_REPO_ID:=ankhanhtran02/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable_perm_1 }"

python upload_output_to_hf.py \
  --output-dir "./output_models/L2P_Qwen2.5-Coder-1.5B_with_instruction_pool_executable_perm_1 " \
  --repo-id "$HF_MODEL_REPO_ID" \
  --commit-message "Upload L2P executable outputs"