#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-LatentRL-Flat-Unitree-G1}"
POLICY_CKPT="${POLICY_CKPT:-/data/wxy/tracking_bfm/logs/rsl_rl/g1_latent_velocity/2026-05-26_06-49-29_latent_rl_flat_g1/model_2000.pt}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-/data/wxy/tracking_bfm/logs/rsl_rl/g1_latent_distillation/2026-05-25_07-57-17_latent_distill_g1/model_3500.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
VIEWER="${VIEWER:-viser}"
AGENT="${AGENT:-trained}"
DEVICE="${DEVICE:-}"
NO_TERMINATIONS="${NO_TERMINATIONS:-False}"
STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-True}"
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
  if [[ -z "$POLICY_CKPT" ]]; then
    echo "POLICY_CKPT must point to a latent velocity policy checkpoint." >&2
    exit 2
  fi
  if [[ ! -f "$POLICY_CKPT" ]]; then
    echo "Policy checkpoint not found: $POLICY_CKPT" >&2
    exit 2
  fi
fi

if [[ -z "$LATENT_DECODER_CKPT" ]]; then
  echo "LATENT_DECODER_CKPT must point to a latent distillation checkpoint." >&2
  exit 2
fi

if [[ ! -f "$LATENT_DECODER_CKPT" ]]; then
  echo "Latent decoder checkpoint not found: $LATENT_DECODER_CKPT" >&2
  exit 2
fi

cmd=(uv run play "$TASK")

if [[ -n "$POLICY_CKPT" ]]; then
  cmd+=(--checkpoint-file "$POLICY_CKPT")
fi

cmd+=(
  --rl.latent-decoder-checkpoint-path "$LATENT_DECODER_CKPT"
  --viewer "$VIEWER"
  --num-envs "$NUM_ENVS"
  --agent "$AGENT"
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
