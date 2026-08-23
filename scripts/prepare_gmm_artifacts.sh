#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-1.5B}"
GMM_ARTIFACT_ROOT="${GMM_ARTIFACT_ROOT:-./output_models/expert_selection/gmm_artifacts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_models/expert_selection/runs}"
TASK_ORDER="${TASK_ORDER:-CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST}"
TASKS="${TASKS:-${TASK_ORDER}}"
SEED="${SEED:-1234}"
DATA_SEED="${DATA_SEED:-1234}"
METHOD_SEED="${METHOD_SEED:-1234}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${GPU_INDEX:-}" ]]; then
    [[ "${GPU_INDEX}" =~ ^[0-9]+([,][0-9]+)*$ ]] || { echo "GPU_INDEX must contain numeric CUDA slots" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

DRY_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then DRY_ARGS+=(--dry-run); fi

echo 'WARNING: offline retrospective bootstrap; this reads named historical task data.'
printf 'tasks=%s\ntask_order=%s\ngmm_artifact_root=%s\n' "${TASKS}" "${TASK_ORDER}" "${GMM_ARTIFACT_ROOT}"

"${PYTHON_BIN}" -m expert_selection.prepare_gmm \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --gmm-artifact-root "${GMM_ARTIFACT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --task-order "${TASK_ORDER}" \
    --tasks "${TASKS}" \
    --seed "${SEED}" --data-seed "${DATA_SEED}" --method-seed "${METHOD_SEED}" \
    "${DRY_ARGS[@]}" "$@"

