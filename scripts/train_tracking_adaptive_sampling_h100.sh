#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
NUM_ENVS="${NUM_ENVS:-16384}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-teacher amass lafan noiton sonic}"

RUN_NAME="${RUN_NAME:-multi gpu adaptive}"

uv run train "$TASK" \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.experiment_name "$EXPERIMENT_NAME" \
    --agent.run_name "$RUN_NAME" \
    --agent.wandb_project "tracking_bfm" \
    --env.commands.motion.sampling-mode adaptive \
    --env.commands.motion.adaptive_pre_failure_sample_window_steps 200 \
    --debug False \
    --gpu_ids "[4,5]"
