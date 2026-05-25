#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Mjlab-LatentRL-Flat-Unitree-G1}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/g1_latent_distillation/2026-05-22_22-38-30_latent_distill_g1/model_4500.pt}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUN_NAME="${RUN_NAME:-latent_rl_flat_g1}"

if [[ -z "$LATENT_DECODER_CKPT" ]]; then
  echo "LATENT_DECODER_CKPT must point to a latent distillation checkpoint." >&2
  exit 1
fi

uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.latent_decoder_checkpoint_path "$LATENT_DECODER_CKPT" \
  --agent.latent_dim 64 \
  --agent.latent_action_clip 12.0 \
  --agent.experiment_name g1_latent_velocity \
  --agent.run_name "$RUN_NAME" \
  --agent.max_iterations "$MAX_ITERATIONS" \
  --agent.num_steps_per_env 24 \
  --agent.upload-model False \
  --debug False
