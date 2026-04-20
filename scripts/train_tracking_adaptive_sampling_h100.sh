#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/AMASS_LAFAN_Qingtong/}"
NUM_ENVS="${NUM_ENVS:-16384}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-adaptive_sampling param}"

RUN_NAME="${RUN_NAME:-adaptive conservative}"

uv run train "$TASK" \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.experiment_name "$EXPERIMENT_NAME" \
    --agent.run_name "$RUN_NAME" \
    --agent.wandb_project "tracking_bfm" \
    --env.commands.motion.sampling-mode adaptive \
    --debug False \
    --env.commands.motion.adaptive_kernel_size 3 \
    --env.commands.motion.adaptive_lambda 0.5 \
    --env.commands.motion.adaptive_uniform_ratio 0.3 \
    --env.commands.motion.adaptive_alpha 0.01 \
    --gpu_ids "[0]"
