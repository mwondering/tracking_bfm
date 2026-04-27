#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Distillation-Flat-Unitree-G1}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/g1_distillation/2026-04-27_11-31-08_distill_mlp_mixed/model_2000.pt}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/test_motion/pufu.npz}"
VIEWER="${VIEWER:-viser}"
NUM_ENVS="${NUM_ENVS:-1}"
NO_TERMINATIONS="${NO_TERMINATIONS:-True}"
AGENT="${AGENT:-trained}"
DEVICE="${DEVICE:-}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"
WANDB_CHECKPOINT_NAME="${WANDB_CHECKPOINT_NAME:-}"
DRY_RUN="${DRY_RUN:-false}"
SHOW_REFERENCE_MOTION="${SHOW_REFERENCE_MOTION:-True}"
STUDENT_HISTORY_STEPS="${STUDENT_HISTORY_STEPS:-0}"
STUDENT_FUTURE_STEPS="${STUDENT_FUTURE_STEPS:-1}"
STUDENT_ROBOT_HISTORY_STEPS="${STUDENT_ROBOT_HISTORY_STEPS:-20}"

bool_is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 2
      ;;
  esac
}

case "$AGENT" in
  trained|zero|random) ;;
  *)
    echo "Invalid AGENT: $AGENT" >&2
    exit 2
    ;;
esac

case "$VIEWER" in
  auto|native|viser) ;;
  *)
    echo "Invalid VIEWER: $VIEWER" >&2
    exit 2
    ;;
esac

if [[ "$AGENT" == "trained" ]]; then
  if [[ -z "$CHECKPOINT_FILE" && -z "$WANDB_RUN_PATH" ]]; then
    echo "Trained mode requires CHECKPOINT_FILE or WANDB_RUN_PATH." >&2
    exit 2
  fi
  if [[ -n "$CHECKPOINT_FILE" && ! -f "$CHECKPOINT_FILE" ]]; then
    echo "Checkpoint file not found: $CHECKPOINT_FILE" >&2
    exit 2
  fi
fi

if [[ -n "$MOTION_FILE" && ! -f "$MOTION_FILE" ]]; then
  echo "Motion file not found: $MOTION_FILE" >&2
  exit 2
fi

cmd=(uv run play "$TASK")

if [[ -n "$CHECKPOINT_FILE" ]]; then
  cmd+=(--checkpoint-file "$CHECKPOINT_FILE")
fi

if [[ -n "$MOTION_FILE" ]]; then
  cmd+=(--motion-file "$MOTION_FILE")
fi

cmd+=(--viewer "$VIEWER" --num-envs "$NUM_ENVS" --agent "$AGENT")

cmd+=(
  --env.observations.student_actor.terms.ee_pose.params.history_steps "$STUDENT_HISTORY_STEPS"
  --env.observations.student_actor.terms.ee_pose.params.future_steps "$STUDENT_FUTURE_STEPS"
  --env.observations.student_actor.terms.base_lin_vel_w.params.history_steps "$STUDENT_HISTORY_STEPS"
  --env.observations.student_actor.terms.base_lin_vel_w.params.future_steps "$STUDENT_FUTURE_STEPS"
  --env.observations.student_actor.terms.base_ang_vel_w.params.history_steps "$STUDENT_HISTORY_STEPS"
  --env.observations.student_actor.terms.base_ang_vel_w.params.future_steps "$STUDENT_FUTURE_STEPS"
  --env.observations.student_actor.terms.anchor_height_w.params.history_steps "$STUDENT_HISTORY_STEPS"
  --env.observations.student_actor.terms.anchor_height_w.params.future_steps "$STUDENT_FUTURE_STEPS"
  --env.observations.student_actor.terms.projected_gravity.history_length "$STUDENT_ROBOT_HISTORY_STEPS"
  --env.observations.student_actor.terms.base_ang_vel.history_length "$STUDENT_ROBOT_HISTORY_STEPS"
  --env.observations.student_actor.terms.joint_pos.history_length "$STUDENT_ROBOT_HISTORY_STEPS"
  --env.observations.student_actor.terms.joint_vel.history_length "$STUDENT_ROBOT_HISTORY_STEPS"
  --env.observations.student_actor.terms.actions.history_length "$STUDENT_ROBOT_HISTORY_STEPS"
)

if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi

if [[ -n "$WANDB_RUN_PATH" ]]; then
  cmd+=(--wandb-run-path "$WANDB_RUN_PATH")
fi

if [[ -n "$WANDB_CHECKPOINT_NAME" ]]; then
  cmd+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT_NAME")
fi

if bool_is_true "$NO_TERMINATIONS"; then
  cmd+=(--no-terminations "$NO_TERMINATIONS")
fi

if ! bool_is_true "$SHOW_REFERENCE_MOTION"; then
  cmd+=(--show-reference-motion "False")
fi

if bool_is_true "$DRY_RUN"; then
  printf '[DRY RUN] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf 'Running command: '
printf '%q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
