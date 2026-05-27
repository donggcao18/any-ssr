#!/usr/bin/env bash
# Inference for SeqSSR-LoRA on the executable benchmark (9 tasks).
#
# Usage:
#   bash scripts/executable/inference_seqssr_lora_executable.sh
#
# Optional overrides (set before calling the script):
#   ALPHA                mixing coefficient used at training time (default: 0.75)
#   GPU_ID               CUDA device index                        (default: 0)
#   INFERENCE_BATCH      per-device batch size                    (default: 1)
#   MODEL                base model name or path                  (default: Qwen/Qwen2.5-Coder-1.5B)
#   CHECKPOINT_DIR       root training output directory           (default: ./output_models/SeqSSRLoRA_Qwen2.5-Coder-1.5B_executable_alpha_${ALPHA})
#   ROUTER_WEIGHT_PATH   HF repo for FE + router weights          (default: ankhanhtran02/router_weights_anyssr_executable_Qwen25_Coder_15b)
#   OUTPUT_DIR           where to write inference results         (default: ./inference_result/seqssr_lora_executable_alpha_${ALPHA})

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

set -euo pipefail

ALPHA="${ALPHA:-0.5}"
GPU_ID="${GPU_ID:-0}"
INFERENCE_BATCH="${INFERENCE_BATCH:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./output_models/SeqSSRLoRA_Qwen2.5-Coder-1.5B_executable_alpha_${ALPHA}}"
ROUTER_WEIGHT_PATH="${ROUTER_WEIGHT_PATH:-ankhanhtran02/router_weights_anyssr_executable_Qwen25_Coder_15b}"
OUTPUT_DIR="${OUTPUT_DIR:-./inference_result/seqssr_lora_executable_alpha_${ALPHA}}"

port=$(shuf -i25000-30000 -n1)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DS_ACCELERATOR=cuda

mkdir -p $OUTPUT_DIR

deepspeed --include=localhost:"${GPU_ID}" --master_port "$port" inference/infer_seqssr_lora.py \
    --router_weight_path "$ROUTER_WEIGHT_PATH" \
    --benchmark executable \
    --data_path "" \
    --inference_tasks python,cpp,swift,rust,csharp \
    --model_name_or_path "$MODEL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --seed 1234 \
    --deepspeed \
    --device cuda \
    --inference_output_path "$OUTPUT_DIR" \
    --inference_batch "$INFERENCE_BATCH" \
    --do_sample \
    --max_prompt_len 1024,1024,1024,1024,1024,1024,1024,1024,1024 \
    --max_ans_len    2048,2048,2048,2048,2048,2048,2048,2048,2048 \
    2>&1 | tee $OUTPUT_DIR/eval.log
