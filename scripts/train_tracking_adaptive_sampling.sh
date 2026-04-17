#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/Data10k}"
NUM_ENVS="${NUM_ENVS:-2048}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-adaptive_sampling}"

RUN_NAME="${RUN_NAME:-uniform_baseline}"

uv run train "$TASK" \
    --env.commands.motion.motion-file "$MOTION_FILE" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.experiment_name "$EXPERIMENT_NAME" \
    --agent.run_name "$RUN_NAME" \
    --agent.wandb_project "tracking_bfm" \
    --env.commands.motion.sampling-mode uniform \
    --debug False