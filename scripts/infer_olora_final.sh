#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

python inference/infer_olora_final.py \
   --checkpoint ./output_models/OLoRA_Qwen2.5-Coder-1.5B_with_instruction_pool/7 \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --per_device_eval_batch_size 8 \
   --lora_alpha 32 \
   --lora_r 16 \
   --seed 1234
