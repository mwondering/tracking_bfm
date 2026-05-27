#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

MODE="${MODE:-delete}"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic/2026-04-28_15-53-48_multi_gpu_adaptive_nocommandwindow_16384/model_48000.pt}"
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_0427/model_139000.pt
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt
MOTION_TYPE="${MOTION_TYPE:-isaaclab}"
NUM_ENVS="${NUM_ENVS:-16384}"
VIEWER="${VIEWER:-none}"
FAILURE_THRESHOLD="${FAILURE_THRESHOLD:-0.95}"
OUTPUT_FILE="${OUTPUT_FILE:-/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic/2026-04-28_15-53-48_multi_gpu_adaptive_nocommandwindow_16384/filter_report.json}"
REPORT_FILE="${REPORT_FILE:-$OUTPUT_FILE}"
MISSING_OK="${MISSING_OK:-True}"

if [[ "$MODE" == "evaluate" ]]; then
  CMD=(
    uv run data-filtering evaluate "$TASK"
    --history-steps 0
    --future-steps 1
    --motion-path "$MOTION_PATH"
    --motion-type "$MOTION_TYPE"
    --num-envs "$NUM_ENVS"
    --viewer "$VIEWER"
    --failure-threshold "$FAILURE_THRESHOLD"
    --output-file "$OUTPUT_FILE"
    --checkpoint-file "$CHECKPOINT_FILE"
    --gpu_ids "[2,3]"
  )
  "${CMD[@]}"
elif [[ "$MODE" == "delete" ]]; then
  CMD=(
    uv run data-filtering delete
    --report-file "$REPORT_FILE"
    --missing-ok "$MISSING_OK"
  )
  "${CMD[@]}"
else
  echo "Unknown MODE: $MODE" >&2
  echo "Use MODE=evaluate or MODE=delete" >&2
  exit 1
fi
