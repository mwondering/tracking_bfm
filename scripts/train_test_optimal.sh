#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

DISABLE_REG_AND_DR="${DISABLE_REG_AND_DR:-True}"

case "${DISABLE_REG_AND_DR,,}" in
  1|true|yes|on)
    DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR"
    DEFAULT_RUN_NAME="test_optimal_global_body_full_obs_no_reg_no_dr"
    ;;
  0|false|no|off)
    DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal"
    DEFAULT_RUN_NAME="test_optimal_global_body_full_obs_with_reg_dr"
    ;;
  *)
    echo "DISABLE_REG_AND_DR must be True or False, got: $DISABLE_REG_AND_DR" >&2
    exit 2
    ;;
esac

TASK="${TASK:-$DEFAULT_TASK}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
NUM_ENVS="${NUM_ENVS:-8192}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_optimal_tracking_bfm}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-tracking_bfm}"
GPU_IDS="${GPU_IDS:-[5,6]}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
DEBUG="${DEBUG:-False}"
UPLOAD_MODEL="${UPLOAD_MODEL:-False}"

cmd=(
  uv run train "$TASK"
  --env.commands.motion.motion-path "$MOTION_PATH"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.experiment_name "$EXPERIMENT_NAME"
  --agent.run_name "$RUN_NAME"
  --agent.wandb_project "$WANDB_PROJECT"
  --env.commands.motion.sampling-mode adaptive
  --env.commands.motion.adaptive_pre_failure_sample_window_steps 200
  --env.commands.motion.history_steps 0
  --env.commands.motion.future_steps 1
  --agent.save_interval "$SAVE_INTERVAL"
  --debug "$DEBUG"
  --agent.upload-model "$UPLOAD_MODEL"
  --gpu_ids "$GPU_IDS"
)

exec "${cmd[@]}"
