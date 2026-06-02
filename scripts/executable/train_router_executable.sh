#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
GAMMA="${GAMMA:-2500}"
GPU_IDS="${GPU_IDS:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
---------- Run (Executable benchmark) ----------
python train_router_ana_continual.py \
    --model_name_or_path "Qwen/Qwen2.5-Coder-1.5B" \
    --benchmark executable \
    --router_weights_path "output_models/router_weights_executable_gamma_${GAMMA}" \
    --dataset_cache_path  "output_models/outputs_router_executable_cache" \
    --max_prompt_len 1024 \
    --max_ans_len    2048 \
    --batch_size     1 \
    --gamma          $GAMMA

python eval_router_ana.py \
    --model_name_or_path "Qwen/Qwen2.5-Coder-1.5B" \
    --benchmark executable \
    --router_weights_path "output_models/router_weights_executable_gamma_${GAMMA}" \
    --dataset_cache_path  "output_models/outputs_router_executable_cache" \
    --max_prompt_len 1024 \
    --max_ans_len    2048 \
    --batch_size          1 \
    --gamma               $GAMMA
