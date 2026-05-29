#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage}"
MOTION_PATH="${MOTION_PATH:-}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-}"
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-300000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_latent_tracking}"
RUN_NAME="${RUN_NAME:-latent_tracking_bfm_1stage_g1}"
WANDB_PROJECT="${WANDB_PROJECT:-tracking_bfm}"
GPU_IDS="${GPU_IDS:-[0]}"
LATENT_DIM="${LATENT_DIM:-64}"
LATENT_ACTION_CLIP="${LATENT_ACTION_CLIP:-15.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
UPLOAD_MODEL="${UPLOAD_MODEL:-False}"
DEBUG="${DEBUG:-False}"
RESUME="${RESUME:-False}"
LOAD_RUN="${LOAD_RUN:-.*}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-model_.*.pt}"
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

if [[ -z "$MOTION_PATH" ]]; then
  echo "MOTION_PATH must point to a directory of tracking motion files." >&2
  exit 2
fi

if [[ ! -d "$MOTION_PATH" ]]; then
  echo "Motion path not found or not a directory: $MOTION_PATH" >&2
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
  uv run train "$TASK"
  --env.commands.motion.motion-path "$MOTION_PATH"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.latent_decoder_checkpoint_path "$LATENT_DECODER_CKPT"
  --agent.latent_dim "$LATENT_DIM"
  --agent.latent_action_clip "$LATENT_ACTION_CLIP"
  --agent.experiment_name "$EXPERIMENT_NAME"
  --agent.run_name "$RUN_NAME"
  --agent.wandb_project "$WANDB_PROJECT"
  --agent.max_iterations "$MAX_ITERATIONS"
  --agent.num_steps_per_env 24
  --agent.save_interval "$SAVE_INTERVAL"
  --agent.upload-model "$UPLOAD_MODEL"
  --env.commands.motion.sampling-mode adaptive
  --env.commands.motion.adaptive_pre_failure_sample_window_steps 200
  --env.commands.motion.history_steps 0
  --env.commands.motion.future_steps 1
  --debug "$DEBUG"
  --gpu_ids "$GPU_IDS"
)

if bool_is_true "$RESUME"; then
  cmd+=(
    --agent.resume True
    --agent.load_run "$LOAD_RUN"
    --agent.load_checkpoint "$LOAD_CHECKPOINT"
  )
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
