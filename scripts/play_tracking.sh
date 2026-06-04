#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
# MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/sonic_filtered/230324/flip_360_001__A304.npz}"
# MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/generated_motion_data/motion.npz}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/test_motion/pufu.npz}"

# CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0604_ckpt/model_14000.pt}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0603_ckpt/model_88000.pt}"

# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_0427/model_139000.pt
NUM_ENVS="${NUM_ENVS:-1}"
VIEWER="${VIEWER:-viser}"

#!/usr/bin/env bash

    # --env.commands.motion.history_steps 0 \
    # --env.commands.motion.future_steps 1 \
uv run play "$TASK" \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --motion-file "$MOTION_FILE" \
    --checkpoint-file "$CHECKPOINT_FILE" \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --env.decimation 4 \
    --num-envs 10 \
    --viewer viser