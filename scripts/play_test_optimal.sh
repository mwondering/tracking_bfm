#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR}"
MOTION_FILE="${MOTION_FILE:-}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"
WANDB_CHECKPOINT_NAME="${WANDB_CHECKPOINT_NAME:-}"
NUM_ENVS="${NUM_ENVS:-1}"
VIEWER="${VIEWER:-viser}"
DEVICE="${DEVICE:-}"
AGENT="${AGENT:-trained}"
MOTION_TYPE="${MOTION_TYPE:-isaaclab}"
NO_TERMINATIONS="${NO_TERMINATIONS:-False}"
STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-False}"
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

if [[ -z "$MOTION_FILE" ]]; then
  echo "MOTION_FILE must point to a recorded motion .npz file." >&2
  exit 2
fi

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "Motion file not found: $MOTION_FILE" >&2
  exit 2
fi

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

cmd=(
  uv run play "$TASK"
  --motion-file "$MOTION_FILE"
  --motion-type "$MOTION_TYPE"
  --env.commands.motion.history_steps 0
  --env.commands.motion.future_steps 1
  --viewer "$VIEWER"
  --num-envs "$NUM_ENVS"
  --agent "$AGENT"
)

if [[ -n "$CHECKPOINT_FILE" ]]; then
  cmd+=(--checkpoint-file "$CHECKPOINT_FILE")
fi

if [[ -n "$WANDB_RUN_PATH" ]]; then
  cmd+=(--wandb-run-path "$WANDB_RUN_PATH")
fi

if [[ -n "$WANDB_CHECKPOINT_NAME" ]]; then
  cmd+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT_NAME")
fi

if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi

if bool_is_true "$NO_TERMINATIONS"; then
  cmd+=(--no-terminations "$NO_TERMINATIONS")
fi

if bool_is_true "$STOCHASTIC_POLICY"; then
  cmd+=(--stochastic-policy "$STOCHASTIC_POLICY")
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
