#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-LatentRL-Rough-Unitree-G1}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/g1_latent_distillation/2026-05-22_22-38-30_latent_distill_g1/model_4500.pt}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_latent_terrain}"
RUN_NAME="${RUN_NAME:-latent_rl_rough_g1}"
GPU_IDS="${GPU_IDS:-[0]}"
LATENT_DIM="${LATENT_DIM:-64}"
LATENT_ACTION_CLIP="${LATENT_ACTION_CLIP:-15.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.00006}"
INIT_STD="${INIT_STD:-0.5}"

if [[ -z "$LATENT_DECODER_CKPT" ]]; then
  echo "LATENT_DECODER_CKPT must point to a latent distillation checkpoint." >&2
  exit 1
fi

uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.latent_decoder_checkpoint_path "$LATENT_DECODER_CKPT" \
  --agent.latent_dim "$LATENT_DIM" \
  --agent.latent_action_clip "$LATENT_ACTION_CLIP" \
  --agent.experiment_name "$EXPERIMENT_NAME" \
  --agent.run_name "$RUN_NAME" \
  --agent.max_iterations "$MAX_ITERATIONS" \
  --agent.num_steps_per_env 24 \
  --agent.upload-model False \
  --agent.algorithm.entropy_coef "$ENTROPY_COEF" \
  --agent.actor.distribution_cfg.init_std "$INIT_STD" \
  --debug False \
  --gpu_ids "$GPU_IDS"
