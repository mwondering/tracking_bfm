#!/usr/bin/env bash
export WANDB_API_KEY="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"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
TEACHER_CKPT="${TEACHER_CKPT:-/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic_prior/2026-06-01_12-56-46_teacher_v2_decimation4_4gpu_16384_resume3/model_88000.pt}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300000}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
HISTORY_STEPS="${HISTORY_STEPS:-0}"
FUTURE_STEPS="${FUTURE_STEPS:-1}"
ROBOT_HISTORY_LENGTH="${ROBOT_HISTORY_LENGTH:-10}"
######bc settings
BC_WEIGHT_START="${BC_WEIGHT_START:-0.5}"
BC_WEIGHT_END="${BC_WEIGHT_END:-0.1}"
BC_DECAY_STEPS="${BC_DECAY_STEPS:-10000}"

PURE_BC_ENABLED="${PURE_BC_ENABLED:-True}"
PURE_BC_WEIGHT="${PURE_BC_WEIGHT:-1.0}"
PURE_BC_ROLLOUT="${PURE_BC_ROLLOUT:-student}"

BC_ACTOR_CKPT="${BC_ACTOR_CKPT:-}"

INIT_ACTOR_STD_FROM_TEACHER="${INIT_ACTOR_STD_FROM_TEACHER:-True}"
INIT_CRITIC_FROM_TEACHER="${INIT_CRITIC_FROM_TEACHER:-True}"
STRICT_INIT="${STRICT_INIT:-True}"
#####resume settings
RESUME="${RESUME:-False}"
LOAD_RUN="${LOAD_RUN:-}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_tracking_wbteleop}"
RUN_NAME="${RUN_NAME:-wbteleop_adaptive_bfm}"
#####
WANDB_PROJECT="${WANDB_PROJECT:-tracking_bfm}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
ADAPTIVE_WINDOW_STEPS="${ADAPTIVE_WINDOW_STEPS:-200}"
GPU_IDS="${GPU_IDS:-[0,1,2,3]}"
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
  --agent.algorithm.pure_bc_enabled "$PURE_BC_ENABLED"
  --agent.algorithm.pure_bc_weight "$PURE_BC_WEIGHT"
  --agent.algorithm.pure_bc_rollout "$PURE_BC_ROLLOUT"
  --agent.algorithm.bc_actor_checkpoint_path "$BC_ACTOR_CKPT"
  --agent.algorithm.init_actor_std_from_teacher "$INIT_ACTOR_STD_FROM_TEACHER"
  --agent.algorithm.init_critic_from_teacher "$INIT_CRITIC_FROM_TEACHER"
  --agent.algorithm.strict_init "$STRICT_INIT"
  --agent.max_iterations "$MAX_ITERATIONS"
  --agent.num_steps_per_env "$NUM_STEPS_PER_ENV"
  --agent.save_interval "$SAVE_INTERVAL"
  --agent.experiment_name "$EXPERIMENT_NAME"
  --agent.run_name "$RUN_NAME"
  --agent.wandb_project "$WANDB_PROJECT"
  --debug False
  --agent.upload-model False
  --gpu_ids "$GPU_IDS"
  --agent.resume "$RESUME"
)

if [[ "$RESUME" == "True" || "$RESUME" == "true" ]]; then
  if [[ -n "$LOAD_RUN" ]]; then
    cmd+=(--agent.load_run "$LOAD_RUN")
  fi
  if [[ -n "$LOAD_CHECKPOINT" ]]; then
    cmd+=(--agent.load_checkpoint "$LOAD_CHECKPOINT")
  fi
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
