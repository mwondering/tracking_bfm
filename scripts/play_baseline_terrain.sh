#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Velocity-Rough-Unitree-G1}"
POLICY_CKPT="${POLICY_CKPT:-}"
NUM_ENVS="${NUM_ENVS:-1}"
VIEWER="${VIEWER:-viser}"
DEVICE="${DEVICE:-}"
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

case "$VIEWER" in
  auto|native|viser) ;;
  *)
    echo "Invalid VIEWER: $VIEWER" >&2
    exit 2
    ;;
esac

if [[ -z "$POLICY_CKPT" ]]; then
  echo "POLICY_CKPT must point to a baseline terrain policy checkpoint." >&2
  exit 2
fi

if [[ ! -f "$POLICY_CKPT" ]]; then
  echo "Policy checkpoint not found: $POLICY_CKPT" >&2
  exit 2
fi

cmd=(uv run play "$TASK")

cmd+=(
  --checkpoint-file "$POLICY_CKPT"
  --viewer "$VIEWER"
  --num-envs "$NUM_ENVS"
)

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
