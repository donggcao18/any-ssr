#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
RESUME_DIR="${RESUME_DIR:-/research/cbim/vast/qt60/any-ssr/outputs/sdft_pairwise_20260913_090917_3356052}"
exec bash "$SCRIPT_DIR/train_sdft_codetrans_pairwise.sh" --resume "$RESUME_DIR" \
  --resume_runtime_settings --restart_incomplete "$@"
