#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/U_PZL2023ZZ0005/rhe/Any-SSR/output_models"
port=$(shuf -i25000-30000 -n1)
GPU_ID="${GPU_ID:-0}"
INFERENCE_BATCH="${INFERENCE_BATCH:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-inference_result_anyssr_executable}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DS_ACCELERATOR=cuda

deepseed_cmd="deepspeed --include=localhost:${GPU_ID} --master_port $port"

# HF code task order (must match router training)
HF_TASKS="python,cpp,swift,rust,csharp,java,php,typescript,shell"

mkdir -p logs

$deepseed_cmd inference/infer_anyssr_total.py \
   --router_weight_path "ankhanhtran02/router_weights_anyssr_executable_Qwen25_Coder_15b" \
   --benchmark executable \
   --data_path "" \
   --inference_tasks $HF_TASKS \
   --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
   --base_path ankhanhtran02/lora-per-task-executable-start-4 \
   --inference_model_path "python/0","cpp/0","swift/0","rust/0","csharp/0","java/0","php/0","typescript/0","shell/0" \
   --seed 1234 \
   --deepspeed \
   --device cuda \
   --inference_output_path "$OUTPUT_DIR" \
   --inference_batch "$INFERENCE_BATCH" \
   --do_sample \
   --max_prompt_len 1024,1024,1024,1024,1024,1024,1024,1024,1024 \
   --max_ans_len 2048,2048,2048,2048,2048,2048,2048,2048,2048 \
   2>&1 | tee logs/inference_result_anyssr_executable.log
