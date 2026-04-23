#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Distillation-Flat-Unitree-G1}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/g1_distillation/2026-04-23_11-38-59_distill_mlp_mixed/model_5000.pt}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/Data10k/homejrhangmr_dataset_pbhc_contact_maskBMLhandballS08_NoviceTrial_upper_right_left_070_posespkl/motion.npz}"
VIEWER="${VIEWER:-viser}"
NUM_ENVS="${NUM_ENVS:-1}"
NO_TERMINATIONS="${NO_TERMINATIONS:-true}"
AGENT="${AGENT:-trained}"
DEVICE="${DEVICE:-}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"
WANDB_CHECKPOINT_NAME="${WANDB_CHECKPOINT_NAME:-}"
DRY_RUN="${DRY_RUN:-false}"

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
  cmd+=(--no-terminations)
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
