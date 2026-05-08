#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage}"
MOTION_FILE="${MOTION_FILE:-/data/zcy/motion_data/AMASS_LAFAN_Qingtong/lafan_qingtong/dance1_subject2.npz}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic_sparse_1stage/2026-05-05_13-07-06_multi_gpu_adaptive_sparse_1stage_16384/model_30000.pt}"
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
    --num-envs 10 \
    --viewer viser