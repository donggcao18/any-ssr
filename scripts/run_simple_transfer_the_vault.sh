#!/usr/bin/env bash

# Minimal old-expert initialization experiment for target TheVault_Csharp.
# The three default conditions use identical target data, initialization seed,
# batch order, optimizer settings, and validation probe:
#   fresh         - randomly initialized LoRA
#   codesearchnet - initialize from anamoe/CodeSearchNet/0
#   CodeTrans     - initialize from anamoe/CodeTrans/0
#
# Examples:
#   bash scripts/run_simple_transfer_the_vault.sh
#   SEEDS="1234 2024 3407" bash scripts/run_simple_transfer_the_vault.sh
#   CONDITIONS="fresh codesearchnet CodeTrans" bash scripts/run_simple_transfer_the_vault.sh
#   GPU_INDEX=1 ADAPTER_ROOT=/path/to/anamoe bash scripts/run_simple_transfer_the_vault.sh
#   GPU_UUID=GPU-... bash scripts/run_simple_transfer_the_vault.sh

set -euo pipefail

export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"

# Prefer an explicitly supplied UUID, then an explicitly supplied numeric slot,
# then an existing CUDA_VISIBLE_DEVICES setting. Otherwise use physical GPU 0.
if [[ -n "${GPU_UUID:-}" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required to resolve GPU_UUID." >&2
        exit 1
    fi
    GPU_LISTING="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
    RESOLVED_GPU_INDEX="$(
        printf '%s\n' "${GPU_LISTING}" \
            | awk -F',' -v uuid="${GPU_UUID}" '
                {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                }
                $2 == uuid { print $1; exit }
            '
    )"
    if [[ -z "${RESOLVED_GPU_INDEX}" ]]; then
        echo "ERROR: Cannot find GPU with UUID: ${GPU_UUID}" >&2
        echo "Available GPUs:" >&2
        printf '%s\n' "${GPU_LISTING}" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${RESOLVED_GPU_INDEX}"
elif [[ -n "${GPU_INDEX:-}" ]]; then
    if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GPU_INDEX must be one numeric GPU slot." >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

ADAPTER_ROOT="${ADAPTER_ROOT:-./anamoe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_models/simple_transfer/TheVault_Csharp}"
SEEDS="${SEEDS:-12}"
CONDITIONS="${CONDITIONS:-fresh codesearchnet CodeTrans KodCode}"   
NUM_TRAIN="${NUM_TRAIN:-8000}"
NUM_EVAL="${NUM_EVAL:-500}"
NUM_TEST="${NUM_TEST:--1}"
EVAL_STEPS="${EVAL_STEPS:-10}"
EVAL_DATA_SEED="${EVAL_DATA_SEED:-1234}"

adapter_for_condition() {
    case "$1" in
        fresh) echo "" ;;
        codesearchnet|CodeSearchNet) echo "${ADAPTER_ROOT}/CodeSearchNet/0" ;;
        codetrans|CodeTrans) echo "${ADAPTER_ROOT}/CodeTrans/0" ;;
        KodCode|KodCode) echo "${ADAPTER_ROOT}/KodCode/0" ;;
        *)
            echo "Unknown condition: $1" >&2
            return 1
            ;;
    esac
}

for seed in ${SEEDS}; do
    for condition in ${CONDITIONS}; do
        init_path="$(adapter_for_condition "${condition}")"
        init_args=()
        if [[ -n "${init_path}" ]]; then
            if [[ ! -f "${init_path}/adapter_config.json" ]]; then
                echo "Missing adapter_config.json in ${init_path}" >&2
                exit 1
            fi
            if [[ ! -f "${init_path}/adapter_model.bin" && ! -f "${init_path}/adapter_model.safetensors" ]]; then
                echo "Missing adapter_model.bin or adapter_model.safetensors in ${init_path}" >&2
                exit 1
            fi
            init_args=(--init_lora_path "${init_path}")
        fi

        output_dir="${OUTPUT_ROOT}/${condition}/seed_${seed}"
        if [[ -f "${output_dir}/convergence.jsonl" ]]; then
            echo "Refusing to append to existing run: ${output_dir}" >&2
            echo "Choose another OUTPUT_ROOT or run only unfinished CONDITIONS." >&2
            exit 1
        fi

        port="$(shuf -i25000-30000 -n1)"
        echo "Running target=TheVault_Csharp condition=${condition} seed=${seed} output=${output_dir}"

        deepspeed --master_port "${port}" training/main_anamoe.py \
            --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
            --data_path "" \
            --dataset_name TheVault_Csharp \
            --num_train "${NUM_TRAIN}" \
            --num_eval "${NUM_EVAL}" \
            --num_test "${NUM_TEST}" \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --max_prompt_len 256 \
            --max_ans_len 128 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --lr_scheduler_type cosine \
            --num_warmup_steps 0 \
            --seed "${seed}" \
            --eval_data_seed "${EVAL_DATA_SEED}" \
            --zero_stage 2 \
            --deepspeed \
            --print_loss \
            --CL_method anamoe \
            --repetition_penalty 1 \
            --convergence_eval_steps "${EVAL_STEPS}" \
            --output_dir "${output_dir}" \
            --run_name "simple_transfer_TheVault_Csharp_${condition}_seed_${seed}" \
            --group_name simple_transfer_TheVault_Csharp \
            --logging_steps 10 \
            "${init_args[@]}"
    done
done

python scripts/summarize_convergence.py \
    "${OUTPUT_ROOT}" \
    --target TheVault_Csharp \
    --csv "${OUTPUT_ROOT}/summary.csv"
