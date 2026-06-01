#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/test_motion/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1General_c3dA5-pickupbox_posespkl/motion.npz}"
POLICY_CKPT="${POLICY_CKPT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0529_ckpt/latent_tracking_encoder/model_4000.pt}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0529_ckpt/latent_distillation_full/model_5000.pt}"
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

if [[ -z "$MOTION_FILE" ]]; then
  echo "MOTION_FILE must point to a replay motion .npz file." >&2
  exit 2
fi

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "Motion file not found: $MOTION_FILE" >&2
  exit 2
fi

if [[ -z "$POLICY_CKPT" ]]; then
  echo "POLICY_CKPT must point to a latent tracking policy checkpoint." >&2
  exit 2
fi

if [[ ! -f "$POLICY_CKPT" ]]; then
  echo "Policy checkpoint not found: $POLICY_CKPT" >&2
  exit 2
fi

if [[ -z "$LATENT_DECODER_CKPT" ]]; then
  echo "LATENT_DECODER_CKPT must point to a latent distillation checkpoint." >&2
  exit 2
fi

if [[ ! -f "$LATENT_DECODER_CKPT" ]]; then
  echo "Latent decoder checkpoint not found: $LATENT_DECODER_CKPT" >&2
  exit 2
fi

cmd=(
  uv run play "$TASK"
  --motion-file "$MOTION_FILE"
  --checkpoint-file "$POLICY_CKPT"
  --rl.latent-decoder-checkpoint-path "$LATENT_DECODER_CKPT"
  --env.commands.motion.history_steps 0
  --env.commands.motion.future_steps 1
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
