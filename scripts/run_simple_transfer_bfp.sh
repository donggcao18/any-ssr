#!/bin/bash

# Minimal old-expert initialization experiment for the BFP target task.
# Default sources all precede BFP in the repository's continual task order:
#   CONCODE       - strongest available old expert: generates Java code
#   CodeTrans     - partial match: consumes Java and performs code transformation
#   CodeSearchNet - unrelated control: Ruby code summarization
#
# Existing source adapters are expected at:
#   anamoe/<source-task>/0/{adapter_config.json,adapter_model.bin}
#
# Examples:
#   bash scripts/run_simple_transfer_bfp.sh
#   SEEDS="1234 2024 3407" bash scripts/run_simple_transfer_bfp.sh
#   CONDITIONS="fresh concode" bash scripts/run_simple_transfer_bfp.sh
#   ADAPTER_ROOT=/path/to/anamoe bash scripts/run_simple_transfer_bfp.sh

set -euo pipefail

export HF_HOME="${HF_HOME:-./.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./.cache}"
GPU_UUID="${GPU_UUID:-GPU-bf710366-2a19-14b1-a2f3-af0ee303d411}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_models/simple_transfer/BFP}"
SEEDS="${SEEDS:-1234}"
CONDITIONS="${CONDITIONS:-fresh concode codetrans codesearchnet}"
NUM_TRAIN="${NUM_TRAIN:-5000}"
NUM_EVAL="${NUM_EVAL:-500}"
NUM_TEST="${NUM_TEST:-1000}"
EVAL_STEPS="${EVAL_STEPS:-25}"

adapter_for_condition() {
    case "$1" in
        fresh) echo "" ;;
        concode) echo "${ADAPTER_ROOT}/CONCODE/0" ;;
        codetrans) echo "${ADAPTER_ROOT}/CodeTrans/0" ;;
        codesearchnet) echo "${ADAPTER_ROOT}/CodeSearchNet/0" ;;
        # Optional oracle for studying task-type knowledge. RunBugRun is also
        # code repair, but it occurs after BFP in the default continual order.
        runbugrun) echo "${ADAPTER_ROOT}/RunBugRun/0" ;;
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
        echo "Running target=BFP condition=${condition} seed=${seed} output=${output_dir}"

        deepspeed --master_port "${port}" training/main_anamoe.py \
            --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
            --data_path "" \
            --dataset_name BFP \
            --num_train "${NUM_TRAIN}" \
            --num_eval "${NUM_EVAL}" \
            --num_test "${NUM_TEST}" \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --max_prompt_len 130 \
            --max_ans_len 120 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --lr_scheduler_type cosine \
            --num_warmup_steps 0 \
            --seed "${seed}" \
            --zero_stage 2 \
            --deepspeed \
            --print_loss \
            --CL_method anamoe \
            --convergence_eval_steps "${EVAL_STEPS}" \
            --output_dir "${output_dir}" \
            --run_name "simple_transfer_BFP_${condition}_seed_${seed}" \
            --group_name simple_transfer_BFP \
            --logging_steps 10 \
            "${init_args[@]}"
    done
done

python scripts/summarize_convergence.py \
    "${OUTPUT_ROOT}" \
    --target BFP \
    --csv "${OUTPUT_ROOT}/summary.csv"
