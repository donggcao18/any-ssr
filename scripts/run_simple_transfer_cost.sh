#!/bin/bash

# Minimal oracle-transfer experiment on the small CoST target task.
# Existing source adapters are loaded from:
#   anamoe/<source-task>/0/{adapter_config.json,adapter_model.bin}
#
# Examples:
#   bash scripts/run_simple_transfer_cost.sh
#   SEEDS="1234 2024 3407" bash scripts/run_simple_transfer_cost.sh
#   CONDITIONS="fresh codetrans" bash scripts/run_simple_transfer_cost.sh
#   ADAPTER_ROOT=/path/to/anamoe bash scripts/run_simple_transfer_cost.sh

set -euo pipefail

export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"
GPU_UUID="${GPU_UUID:-GPU-f65321c7-b1db-a16e-affc-0664b73cf821}"

# DeepSpeed expects a numeric GPU slot rather than a GPU UUID.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not available on this host." >&2
    exit 1
fi

GPU_LISTING="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
GPU_INDEX="$(
    printf '%s\n' "${GPU_LISTING}" \
        | awk -F',' -v uuid="${GPU_UUID}" '
            {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            }
            $2 == uuid { print $1; exit }
        '
)"

if [[ -z "${GPU_INDEX}" ]]; then
    echo "ERROR: Cannot find GPU with UUID: ${GPU_UUID}" >&2
    echo "Available GPUs:" >&2
    printf '%s\n' "${GPU_LISTING}" >&2
    exit 1
fi

echo "Using GPU UUID: ${GPU_UUID}"
echo "Mapped to GPU index: ${GPU_INDEX}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

ADAPTER_ROOT="${ADAPTER_ROOT:-./anamoe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_models/simple_transfer/CoST}"
SEEDS="${SEEDS:-1234}"
CONDITIONS="${CONDITIONS:-CodeTrans RunBugRun BFP}"
NUM_TRAIN="${NUM_TRAIN:-5000}"
# A fixed 200-example validation probe supports dense NLL measurements and is
# safely below CoST's 272-example validation split.
NUM_EVAL="${NUM_EVAL:--1}"
NUM_TEST="${NUM_TEST:--1}"
EVAL_STEPS="${EVAL_STEPS:-10}"
EVAL_DATA_SEED="${EVAL_DATA_SEED:-1234}"

adapter_for_condition() {
    case "$1" in
        fresh) echo "" ;;
        CodeTrans) echo "${ADAPTER_ROOT}/CodeTrans/0" ;;
        RunBugRun) echo "${ADAPTER_ROOT}/RunBugRun/0" ;;
        BFP) echo "${ADAPTER_ROOT}/BFP/0" ;;
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
            if [[ ! -f "${init_path}/adapter_config.json" || ! -f "${init_path}/adapter_model.bin" ]]; then
                echo "Missing source adapter files in ${init_path}" >&2
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
        echo "Running target=CoST condition=${condition} seed=${seed} output=${output_dir}"

        deepspeed --master_port "${port}" training/main_anamoe.py \
            --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
            --data_path "" \
            --dataset_name CoST \
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
            --convergence_eval_steps "${EVAL_STEPS}" \
            --output_dir "${output_dir}" \
            --run_name "simple_transfer_CoST_${condition}_seed_${seed}" \
            --group_name simple_transfer_CoST \
            --logging_steps 10 \
            "${init_args[@]}"
    done
done

python scripts/summarize_convergence.py \
    "${OUTPUT_ROOT}" \
    --csv "${OUTPUT_ROOT}/summary.csv"
