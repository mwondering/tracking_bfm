#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
NUM_ENVS="${NUM_ENVS:-16384}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-teacher_amass_lafan_noiton_sonic_prior}"

RUN_NAME="${RUN_NAME:-teacher_v2_decimation4_5gpu_16384_resume1}"

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
    --agent.save_interval 2000 \
    --debug False \
    --agent.upload-model False \
    --gpu_ids "[3,4,5,6,7]" \
    --agent.resume True \
    --agent.load_run "2026-05-16_15-15-49_teacher_2nd_decimation4_start" \
    --agent.load_checkpoint "model_6000.pt" \
