#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

# Permutation 1: A -> B -> H -> C -> G -> D -> F -> E
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name CONCODE,CodeTrans,CoST,CodeSearchNet,TheVault_Csharp,BFP,RunBugRun,KodCode \
   --max_prompt_len 320,320,256,256,256,130,256,512 \
   --max_ans_len 150,256,128,128,128,120,128,300 \
   --num_eval 100 \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --learning_rate 1e-4 \
   --CL_method O-LoRA \
   --output_dir ./output_models/OLoRA_Qwen2.5-Coder-1.5B_permutation_1 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --temperature 0.2 \
   --top_p 0.95 \
   --repetition_penalty 1 \
   --run_name permutation_1 \
   --group_name OLoRA_Qwen2.5-Coder-1.5B_permutations


# Permutation 2: C -> D -> B -> E -> A -> F -> H -> G
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name CodeSearchNet,BFP,CodeTrans,KodCode,CONCODE,RunBugRun,CoST,TheVault_Csharp \
   --max_prompt_len 256,130,320,512,320,256,256,256 \
   --max_ans_len 128,120,256,300,150,128,128,128 \
   --num_eval 100 \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --learning_rate 1e-4 \
   --CL_method O-LoRA \
   --output_dir ./output_models/OLoRA_Qwen2.5-Coder-1.5B_permutation_2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --temperature 0.2 \
   --top_p 0.95 \
   --repetition_penalty 1 \
   --run_name permutation_2 \
   --group_name OLoRA_Qwen2.5-Coder-1.5B_permutations


# Permutation 3: E -> F -> D -> G -> C -> H -> B -> A
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name KodCode,RunBugRun,BFP,TheVault_Csharp,CodeSearchNet,CoST,CodeTrans,CONCODE \
   --max_prompt_len 512,256,130,256,256,256,320,320 \
   --max_ans_len 300,128,120,128,128,128,256,150 \
   --num_eval 100 \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --learning_rate 1e-4 \
   --CL_method O-LoRA \
   --output_dir ./output_models/OLoRA_Qwen2.5-Coder-1.5B_permutation_3 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --temperature 0.2 \
   --top_p 0.95 \
   --repetition_penalty 1 \
   --run_name permutation_3 \
   --group_name OLoRA_Qwen2.5-Coder-1.5B_permutations


# Permutation 4: G -> H -> F -> A -> E -> B -> D -> C
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name TheVault_Csharp,CoST,RunBugRun,CONCODE,KodCode,CodeTrans,BFP,CodeSearchNet \
   --max_prompt_len 256,256,256,320,512,320,130,256 \
   --max_ans_len 128,128,128,150,300,256,120,128 \
   --num_eval 100 \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --learning_rate 1e-4 \
   --CL_method O-LoRA \
   --output_dir ./output_models/OLoRA_Qwen2.5-Coder-1.5B_permutation_4 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --temperature 0.2 \
   --top_p 0.95 \
   --repetition_penalty 1 \
   --run_name permutation_4 \
   --group_name OLoRA_Qwen2.5-Coder-1.5B_permutations


# Permutation 5: H -> A -> G -> B -> F -> C -> E -> D
port=$(shuf -i25000-30000 -n1)

deepspeed --master_port "$port" training/main_anamoe.py \
   --data_path /path/to/LLM-CL-Benchmark_5000 \
   --dataset_name CoST,CONCODE,TheVault_Csharp,CodeTrans,RunBugRun,CodeSearchNet,KodCode,BFP \
   --max_prompt_len 256,320,256,320,256,256,512,130 \
   --max_ans_len 128,150,128,256,128,128,300,120 \
   --num_eval 100 \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --learning_rate 1e-4 \
   --CL_method O-LoRA \
   --output_dir ./output_models/OLoRA_Qwen2.5-Coder-1.5B_permutation_5 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --temperature 0.2 \
   --top_p 0.95 \
   --repetition_penalty 1 \
   --run_name permutation_5 \
   --group_name OLoRA_Qwen2.5-Coder-1.5B_permutations
