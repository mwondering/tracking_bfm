#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Distillation-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/DATASETS/Data10k}"
TEACHER_CKPT="${TEACHER_CKPT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt/model_9000.pt}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
BETA_DECAY_STEPS="${BETA_DECAY_STEPS:-2000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_distillation}"
RUN_NAME="${RUN_NAME:-distill_mlp_mixed}"

uv run train "$TASK" \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs "$NUM_ENVS" \
    --env.commands.motion.sampling-mode uniform \
    --env.commands.motion.adaptive_pre_failure_sample_window_steps 200 \
    --agent.teacher_checkpoint_path "$TEACHER_CKPT" \
    --agent.max_iterations "$MAX_ITERATIONS" \
    --agent.num_steps_per_env "$NUM_STEPS_PER_ENV" \
    --agent.save_interval "$SAVE_INTERVAL" \
    --agent.beta_decay_steps "$BETA_DECAY_STEPS" \
    --agent.experiment_name "$EXPERIMENT_NAME" \
    --agent.run_name "$RUN_NAME" \
    --agent.wandb_project "tracking_bfm_distillation" \
    --debug False \
    --agent.upload-model False
