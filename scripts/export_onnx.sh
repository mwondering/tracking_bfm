#!/usr/bin/env bash

set -euo pipefail

MOTION_PATH="${MOTION_PATH:-/home/lenovo/DATASETS/Data10k}"
STUDENT_HISTORY_STEPS="${STUDENT_HISTORY_STEPS:-0}"
STUDENT_FUTURE_STEPS="${STUDENT_FUTURE_STEPS:-1}"
STUDENT_ROBOT_HISTORY_STEPS="${STUDENT_ROBOT_HISTORY_STEPS:-20}"
# --task-id Mjlab-Trackingbfm-Flat-Unitree-G1 \
# Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage
uv run export-tracking-bfm-onnx \
    --checkpoint /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0601_5wa_ckpt/model_36000.pt \
    --task-id Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage \
    --motion-path "$MOTION_PATH" \
    --student-history-steps "$STUDENT_HISTORY_STEPS" \
    --student-future-steps "$STUDENT_FUTURE_STEPS" \
    --student-robot-history-steps "$STUDENT_ROBOT_HISTORY_STEPS"
