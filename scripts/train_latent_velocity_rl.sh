#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Mjlab-LatentRL-Flat-Unitree-G1}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-/data/wxy/tracking_bfm/logs/rsl_rl/g1_latent_distillation/2026-05-25_07-57-17_latent_distill_g1/model_3500.pt}"
NUM_ENVS="${NUM_ENVS:-16384}"
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
  --agent.latent_action_clip 15.0 \
  --agent.experiment_name g1_latent_velocity \
  --agent.run_name "$RUN_NAME" \
  --agent.max_iterations "$MAX_ITERATIONS" \
  --agent.num_steps_per_env 24 \
  --agent.upload-model False \
  --agent.algorithm.entropy_coef 0.00006 \
  --agent.actor.distribution_cfg.init_std 0.5 \
  --debug False \
  --gpu_ids "[3]" \