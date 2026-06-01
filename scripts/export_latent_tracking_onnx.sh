#!/usr/bin/env bash

set -euo pipefail

TASK="${TASK:-Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/DATASETS/Data10k}"
DEVICE="${DEVICE:-gpu}"
CHECKPOINT="${CHECKPOINT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0529_ckpt/latent_tracking_encoder/model_16000.pt}"
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0529_ckpt/latent_distillation_full/model_5000.pt}"

if [[ "$DEVICE" == "gpu" ]]; then
  DEVICE="cuda:0"
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "CHECKPOINT must point to the latent tracking actor .pt checkpoint." >&2
  exit 1
fi

if [[ -z "$DECODER_CHECKPOINT" ]]; then
  echo "DECODER_CHECKPOINT must point to the latent distillation decoder .pt checkpoint." >&2
  exit 1
fi

args=(
  uv run export-latent-tracking-bfm-onnx
  --checkpoint "$CHECKPOINT"
  --decoder-checkpoint "$DECODER_CHECKPOINT"
  --task-id "$TASK"
  --device "$DEVICE"
)

if [[ -n "${MOTION_FILE:-}" ]]; then
  args+=(--motion-file "$MOTION_FILE")
else
  args+=(--motion-path "$MOTION_PATH")
fi

if [[ -n "${OUTPUT_NAME:-}" ]]; then
  args+=(--output-name "$OUTPUT_NAME")
fi

if [[ -n "${ROBOT_NAME:-}" ]]; then
  args+=(--robot-name "$ROBOT_NAME")
fi

if [[ -n "${LATENT_ACTION_CLIP:-}" ]]; then
  args+=(--latent-action-clip "$LATENT_ACTION_CLIP")
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
  args+=(--verbose)
fi

"${args[@]}"
