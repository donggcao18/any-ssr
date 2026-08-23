#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-1.5B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-./anamoe}"
GMM_ARTIFACT_ROOT="${GMM_ARTIFACT_ROOT:-./output_models/expert_selection/gmm_artifacts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_models/expert_selection/runs}"
TASK_ORDER="${TASK_ORDER:-CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST}"
TARGETS="${TARGETS:-BFP,CoST}"
METHODS="${METHODS:-gmm,gca,oia,slu}"
SEED="${SEED:-1234}"
DATA_SEED="${DATA_SEED:-1234}"
METHOD_SEED="${METHOD_SEED:-1234}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${GPU_INDEX:-}" ]]; then
    [[ "${GPU_INDEX}" =~ ^[0-9]+([,][0-9]+)*$ ]] || { echo "GPU_INDEX must contain numeric CUDA slots" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

DRY_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then DRY_ARGS+=(--dry-run); fi

printf 'wrapper_defaults: mode=chronological_sequence\nmodel=%s\ntask_order=%s\ntargets=%s\nmethods=%s\nadapter_root=%s\ngmm_artifact_root=%s\noutput_root=%s\nseeds=%s,%s,%s\n' \
    "${MODEL_NAME_OR_PATH}" "${TASK_ORDER}" "${TARGETS}" "${METHODS}" "${ADAPTER_ROOT}" \
    "${GMM_ARTIFACT_ROOT}" "${OUTPUT_ROOT}" "${SEED}" "${DATA_SEED}" "${METHOD_SEED}"

"${PYTHON_BIN}" -m expert_selection.sequence \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --adapter-root "${ADAPTER_ROOT}" \
    --gmm-artifact-root "${GMM_ARTIFACT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task-order "${TASK_ORDER}" \
    --targets "${TARGETS}" \
    --methods "${METHODS}" \
    --prepare-artifacts gmm \
    --seed "${SEED}" --data-seed "${DATA_SEED}" --method-seed "${METHOD_SEED}" \
    "${DRY_ARGS[@]}" "$@"
