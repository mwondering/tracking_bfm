#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/DATASETS/Data10k}"
NUM_ENVS="${NUM_ENVS:-512}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-0518debug}"

RUN_NAME="${RUN_NAME:-adaptive}"

uv run train "$TASK" \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.experiment_name "$EXPERIMENT_NAME" \
    --agent.run_name "$RUN_NAME" \
    --agent.wandb_project "tracking_bfm" \
    --env.commands.motion.sampling-mode adaptive \
    --env.commands.motion.adaptive_pre_failure_sample_window_steps 200 \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --debug True \
    --env.commands.motion.adaptive-failure-rate-window-iterations 400 \
    --env.commands.motion.adaptive_failure_rate_window_chunks 40


    # --env.commands.motion.future_steps 1 \
    # --env.commands.motion.history_steps 0 \