#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_FILE="${MOTION_FILE:-/home/lenovo/DATASETS/Data10k/mocapblender_take_020_edit_cont_inter05_S0-15_E699-15/motion.npz}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt}"
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_ckpt_0501/model_48000.pt
# /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/teacher_0427/model_139000.pt
NUM_ENVS="${NUM_ENVS:-10}"
VIEWER="${VIEWER:-viser}"

#!/usr/bin/env bash

    # --env.commands.motion.history_steps 0 \
    # --env.commands.motion.future_steps 1 \
uv run play "$TASK" \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --motion-file "$MOTION_FILE" \
    --checkpoint-file "$CHECKPOINT_FILE" \
    --num-envs 10 \
    --viewer viser