#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Velocity-Flat-Unitree-G1}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_velocity}"
RUN_NAME="${RUN_NAME:-velocity_flat_g1}"
GPU_IDS="${GPU_IDS:-[2]}"

uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.experiment_name "$EXPERIMENT_NAME" \
  --agent.run_name "$RUN_NAME" \
  --agent.max_iterations "$MAX_ITERATIONS" \
  --agent.num_steps_per_env 24 \
  --agent.upload-model False \
  --debug False \
  --gpu_ids "$GPU_IDS"
