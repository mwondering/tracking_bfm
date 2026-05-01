#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

MODE="${MODE:-evaluate}"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/DATASETS/Data10k}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt}"
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_0427/model_139000.pt
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt
MOTION_TYPE="${MOTION_TYPE:-isaaclab}"
NUM_ENVS="${NUM_ENVS:-512}"
VIEWER="${VIEWER:-none}"
GPU_IDS="${GPU_IDS:-}"
FAILURE_THRESHOLD="${FAILURE_THRESHOLD:-0.9}"
OUTPUT_FILE="${OUTPUT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/filter_report.json}"
REPORT_FILE="${REPORT_FILE:-$OUTPUT_FILE}"
MISSING_OK="${MISSING_OK:-True}"

if [[ "$MODE" == "evaluate" ]]; then
  EFFECTIVE_VIEWER="$VIEWER"
  if [[ -n "$GPU_IDS" ]]; then
    EFFECTIVE_VIEWER="none"
  fi
  CMD=(
    uv run data-filtering evaluate "$TASK"
    --history-steps 0
    --future-steps 1
    --motion-path "$MOTION_PATH"
    --motion-type "$MOTION_TYPE"
    --num-envs "$NUM_ENVS"
    --viewer "$EFFECTIVE_VIEWER"
    --failure-threshold "$FAILURE_THRESHOLD"
    --output-file "$OUTPUT_FILE"
    --checkpoint-file "$CHECKPOINT_FILE"
  )
  if [[ -n "$GPU_IDS" ]]; then
    CMD+=(--gpu-ids "$GPU_IDS")
  fi
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
