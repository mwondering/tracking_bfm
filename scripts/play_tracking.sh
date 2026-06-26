#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop}"
# MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/sonic_filtered/230324/flip_360_001__A304.npz}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/test_motion/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB19-walktopickupbox_posespkl/motion.npz}"
# MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/generated_motion_data/motion.npz}"
# MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/test_motion/pufu.npz}"

# CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0604_ckpt/model_14000.pt}"
# CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0603_ckpt/model_88000.pt}"
# CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0603_ckpt/model_88000.pt}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0608_ckpt_bc/model_8000.pt}"

# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_0427/model_139000.pt
NUM_ENVS="${NUM_ENVS:-10}"
VIEWER="${VIEWER:-viser}"
DOMAIN_RANDOMIZATION="${DOMAIN_RANDOMIZATION:-true}"
EXTRA_REFERENCE_MOTION_FILE="${EXTRA_REFERENCE_MOTION_FILE:-}"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<'EOF'
Usage: scripts/play_tracking.sh [OPTIONS]

Options:
  --task TASK
  --motion-file PATH
  --checkpoint-file PATH
  --num-envs N
  --viewer auto|native|viser
  --domain-randomization true|false
  --extra-reference-motion-file PATH
  --dry-run
  -h, --help

Environment variables with the same uppercase names are also supported.
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
    --domain-randomization)
      DOMAIN_RANDOMIZATION="$2"
      shift 2
      ;;
    --extra-reference-motion-file)
      EXTRA_REFERENCE_MOTION_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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

if [[ -z "$CHECKPOINT_FILE" ]]; then
  echo "CHECKPOINT_FILE must point to a tracking policy checkpoint." >&2
  exit 2
fi

if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_FILE" >&2
  exit 2
fi

if [[ -n "$EXTRA_REFERENCE_MOTION_FILE" && ! -f "$EXTRA_REFERENCE_MOTION_FILE" ]]; then
  echo "Extra reference motion file not found: $EXTRA_REFERENCE_MOTION_FILE" >&2
  exit 2
fi

    # --env.commands.motion.history_steps 0 \
    # --env.commands.motion.future_steps 1 \
    # --env.observations.actor.terms.ref_limb_ee_pose_b.history_length 5 \
    # --env.observations.actor.terms.robot_limb_ee_pose_b.history_length 5 \
    # --env.observations.actor.terms.projected_gravity.history_length 5 \
    # --env.observations.actor.terms.base_ang_vel.history_length 5 \
    # --env.observations.actor.terms.joint_pos.history_length 5 \
    # --env.observations.actor.terms.joint_vel.history_length 5 \
    # --env.observations.actor.terms.actions.history_length 5 \
cmd=(
  uv run play "$TASK"
  --motion-file "$MOTION_FILE"
  --checkpoint-file "$CHECKPOINT_FILE"
  --env.commands.motion.history_steps 0
  --env.commands.motion.future_steps 1
  --env.observations.actor.terms.ref_limb_ee_pose_b.history_length 5
  --env.observations.actor.terms.robot_limb_ee_pose_b.history_length 5
  --env.observations.actor.terms.projected_gravity.history_length 5
  --env.observations.actor.terms.base_ang_vel.history_length 5
  --env.observations.actor.terms.joint_pos.history_length 5
  --env.observations.actor.terms.joint_vel.history_length 5
  --env.observations.actor.terms.actions.history_length 5
  --env.decimation 4
  --num-envs "$NUM_ENVS"
  --viewer "$VIEWER"
)

if ! bool_is_true "$DOMAIN_RANDOMIZATION"; then
  cmd+=(--domain-randomization False)
fi

if [[ -n "$EXTRA_REFERENCE_MOTION_FILE" ]]; then
  cmd+=(--extra-reference-motion-file "$EXTRA_REFERENCE_MOTION_FILE")
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
