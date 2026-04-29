#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name all \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --offload \
   --gradient_checkpointing \
   --deepspeed \
   --print_loss \
   --num_train -1 \
   --num_eval 100 \
   --num_test -1 \
   --learning_rate 1e-4 \
   --CL_method LwF \
   --output_dir ./output_models/LwF_Qwen2.5-Coder-1.5B_with_instruction_pool \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 2 \
   --run_name run_1 \
   --group_name LwF_Qwen2.5-Coder-1.5B_with_instruction_pool \