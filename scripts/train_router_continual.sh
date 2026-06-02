#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
GAMMA="${GAMMA:-2500}"
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# # ---------- Run ----------
# python train_router_ana_continual.py \
#     --model_name_or_path          "Qwen/Qwen2.5-Coder-1.5B" \
#     --benchmark non-executable \
#     --gamma          $GAMMA \
#     --router_weights_path "output_models/router_weights_with_pool_codetask_gamma_${GAMMA}" \
#     --dataset_cache_path  "output_models/outputs_router_dataset_with_pool_cache" \
#     --max_prompt_len 512 \
#     --max_ans_len    256 \
#     --batch_size     1 


python eval_router_ana.py \
    --model_name_or_path  "Qwen/Qwen2.5-Coder-1.5B" \
    --benchmark non-executable \
    --gamma               $GAMMA \
    --router_weights_path "output_models/router_weights_with_pool_codetask_gamma_${GAMMA}" \
    --dataset_cache_path  "output_models/outputs_router_dataset_with_pool_cache" \
    --max_prompt_len      512 \
    --max_ans_len         256 \
    --batch_size          1 
