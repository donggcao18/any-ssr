#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-1.5B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-./anamoe}"
TASK_ORDER="${TASK_ORDER:-CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST}"
TARGETS="${TARGETS:-CodeSearchNet}"
METHODS="${METHODS:-gmm,gca,oia,slu}"
SMOKE_ROOT="${SMOKE_ROOT:-./output_models/expert_selection/smoke}"
SEED="${SEED:-1234}"
DATA_SEED="${DATA_SEED:-1234}"
METHOD_SEED="${METHOD_SEED:-1234}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${GPU_INDEX:-}" ]]; then
    [[ "${GPU_INDEX}" =~ ^[0-9]+([,][0-9]+)*$ ]] || { echo "GPU_INDEX must contain numeric CUDA slots" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

DRY_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then DRY_ARGS+=(--dry-run); fi

echo 'Running isolated tiny chronological smoke experiment.'
"${PYTHON_BIN}" -m expert_selection.sequence \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --adapter-root "${ADAPTER_ROOT}" \
    --gmm-artifact-root "${SMOKE_ROOT}/gmm_artifacts" \
    --output-root "${SMOKE_ROOT}/runs" \
    --task-order "${TASK_ORDER}" --targets "${TARGETS}" --methods "${METHODS}" \
    --prepare-artifacts gmm \
    --calibration-size 32 --support-size 12 --verification-size 12 --gca-size 8 \
    --source-gmm-cap 32 --gmm-mc-samples 128 --gmm-mc-chunk-size 64 \
    --slu-checkpoints 1,2 --slu-weights 1,1 \
    --seed "${SEED}" --data-seed "${DATA_SEED}" --method-seed "${METHOD_SEED}" \
    "${DRY_ARGS[@]}" "$@"

