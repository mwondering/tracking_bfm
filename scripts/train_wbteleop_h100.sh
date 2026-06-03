#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
TEACHER_CKPT="${TEACHER_CKPT:-/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic_prior/model.pt}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300000}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
HISTORY_STEPS="${HISTORY_STEPS:-0}"
FUTURE_STEPS="${FUTURE_STEPS:-1}"
ROBOT_HISTORY_LENGTH="${ROBOT_HISTORY_LENGTH:-$((HISTORY_STEPS > 0 ? HISTORY_STEPS + 1 : 0))}"
BC_WEIGHT_START="${BC_WEIGHT_START:-0.5}"
BC_WEIGHT_END="${BC_WEIGHT_END:-0.1}"
BC_DECAY_STEPS="${BC_DECAY_STEPS:-10000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_tracking_wbteleop}"
RUN_NAME="${RUN_NAME:-wbteleop_adaptive_bfm}"
WANDB_PROJECT="${WANDB_PROJECT:-tracking_bfm}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
ADAPTIVE_WINDOW_STEPS="${ADAPTIVE_WINDOW_STEPS:-200}"
GPU_IDS="${GPU_IDS:-[0,1]}"
DRY_RUN="${DRY_RUN:-false}"

cmd=(
  uv run train "$TASK"
  --env.commands.motion.motion-path "$MOTION_PATH"
  --env.scene.num-envs "$NUM_ENVS"
  --env.commands.motion.sampling-mode "$SAMPLING_MODE"
  --env.commands.motion.adaptive_pre_failure_sample_window_steps "$ADAPTIVE_WINDOW_STEPS"
  --env.commands.motion.history_steps "$HISTORY_STEPS"
  --env.commands.motion.future_steps "$FUTURE_STEPS"
  --env.observations.actor.terms.projected_gravity.history_length "$ROBOT_HISTORY_LENGTH"
  --env.observations.actor.terms.base_ang_vel.history_length "$ROBOT_HISTORY_LENGTH"
  --env.observations.actor.terms.joint_pos.history_length "$ROBOT_HISTORY_LENGTH"
  --env.observations.actor.terms.joint_vel.history_length "$ROBOT_HISTORY_LENGTH"
  --env.observations.actor.terms.actions.history_length "$ROBOT_HISTORY_LENGTH"
  --agent.algorithm.teacher_checkpoint_path "$TEACHER_CKPT"
  --agent.algorithm.bc_weight_start "$BC_WEIGHT_START"
  --agent.algorithm.bc_weight_end "$BC_WEIGHT_END"
  --agent.algorithm.bc_decay_steps "$BC_DECAY_STEPS"
  --agent.max_iterations "$MAX_ITERATIONS"
  --agent.num_steps_per_env "$NUM_STEPS_PER_ENV"
  --agent.save_interval "$SAVE_INTERVAL"
  --agent.experiment_name "$EXPERIMENT_NAME"
  --agent.run_name "$RUN_NAME"
  --agent.wandb_project "$WANDB_PROJECT"
  --debug False
  --agent.upload-model False
  --gpu_ids "$GPU_IDS"
)

if [[ "$DRY_RUN" == "true" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
