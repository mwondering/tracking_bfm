#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_FILE="${MOTION_FILE:-}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
NUM_ENVS="${NUM_ENVS:-1}"
VIEWER="${VIEWER:-viser}"
DEVICE="${DEVICE:-}"
AGENT="${AGENT:-trained}"
REGISTRY_NAME="${REGISTRY_NAME:-}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"
WANDB_CHECKPOINT_NAME="${WANDB_CHECKPOINT_NAME:-}"
VIDEO="${VIDEO:-false}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
NO_TERMINATIONS="${NO_TERMINATIONS:-false}"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/play_tracking.sh [options]

Options:
  --task TASK
  --motion-file PATH
  --checkpoint-file PATH
  --num-envs N
  --viewer {auto|native|viser}
  --device DEVICE
  --agent {trained|zero|random}
  --registry-name NAME
  --wandb-run-path PATH
  --wandb-checkpoint-name NAME
  --video
  --video-length N
  --no-terminations
  --dry-run
  --help

Environment variable overrides:
  TASK, MOTION_FILE, CHECKPOINT_FILE, NUM_ENVS, VIEWER, DEVICE, AGENT,
  REGISTRY_NAME, WANDB_RUN_PATH, WANDB_CHECKPOINT_NAME, VIDEO, VIDEO_LENGTH,
  NO_TERMINATIONS, DRY_RUN
EOF
}

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --motion-file)
      MOTION_FILE="$2"
      shift 2
      ;;
    --checkpoint-file)
      CHECKPOINT_FILE="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --viewer)
      VIEWER="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --registry-name)
      REGISTRY_NAME="$2"
      shift 2
      ;;
    --wandb-run-path)
      WANDB_RUN_PATH="$2"
      shift 2
      ;;
    --wandb-checkpoint-name)
      WANDB_CHECKPOINT_NAME="$2"
      shift 2
      ;;
    --video)
      VIDEO="true"
      shift
      ;;
    --video-length)
      VIDEO_LENGTH="$2"
      shift 2
      ;;
    --no-terminations)
      NO_TERMINATIONS="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$AGENT" in
  trained|zero|random) ;;
  *)
    echo "Invalid agent: $AGENT" >&2
    exit 2
    ;;
esac

case "$VIEWER" in
  auto|native|viser) ;;
  *)
    echo "Invalid viewer: $VIEWER" >&2
    exit 2
    ;;
esac

if [[ "$AGENT" == "trained" ]]; then
  if [[ -z "$CHECKPOINT_FILE" && -z "$WANDB_RUN_PATH" ]]; then
    echo "Trained mode requires --checkpoint-file or --wandb-run-path." >&2
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

if [[ -n "$MOTION_FILE" ]]; then
  cmd+=(--motion-file "$MOTION_FILE")
fi

if [[ -n "$CHECKPOINT_FILE" ]]; then
  cmd+=(--checkpoint-file "$CHECKPOINT_FILE")
fi

cmd+=(--num-envs "$NUM_ENVS" --viewer "$VIEWER" --agent "$AGENT")

if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi

if [[ -n "$REGISTRY_NAME" ]]; then
  cmd+=(--registry-name "$REGISTRY_NAME")
fi

if [[ -n "$WANDB_RUN_PATH" ]]; then
  cmd+=(--wandb-run-path "$WANDB_RUN_PATH")
fi

if [[ -n "$WANDB_CHECKPOINT_NAME" ]]; then
  cmd+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT_NAME")
fi

if bool_is_true "$VIDEO"; then
  cmd+=(--video --video-length "$VIDEO_LENGTH")
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
