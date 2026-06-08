#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0608_ckpt_bcrl/model_16000.pt}"
TASK_ID="${TASK_ID:-Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/DATASETS/Data10k}"

WBTELEOP_REF_HISTORY_STEPS="${WBTELEOP_REF_HISTORY_STEPS:-${STUDENT_HISTORY_STEPS:-4}}"
WBTELEOP_REF_FUTURE_STEPS="${WBTELEOP_REF_FUTURE_STEPS:-${STUDENT_FUTURE_STEPS:-1}}"
WBTELEOP_ROBOT_HISTORY_LENGTH="${WBTELEOP_ROBOT_HISTORY_LENGTH:-${STUDENT_ROBOT_HISTORY_STEPS:-5}}"

uv run export-tracking-bfm-onnx \
    --checkpoint "$CHECKPOINT" \
    --task-id "$TASK_ID" \
    --motion-path "$MOTION_PATH" \
    --student-history-steps "$WBTELEOP_REF_HISTORY_STEPS" \
    --student-future-steps "$WBTELEOP_REF_FUTURE_STEPS" \
    --student-robot-history-steps "$WBTELEOP_ROBOT_HISTORY_LENGTH"
