#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

TASK="${TASK:-Mjlab-Trackingbfm-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/AMASS_LAFAN_Qingtong/}"
NUM_ENVS="${NUM_ENVS:-16384}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-adaptive_sampling_sonic}"
RUN_NAME="${RUN_NAME:-adaptive sonic ddp debug}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
WORLD_SIZE="${WORLD_SIZE:-2}"

if [[ "$WORLD_SIZE" != "2" ]]; then
  echo "[ERROR] This debug script currently expects WORLD_SIZE=2." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/tmp/tracking_bfm_ddp_debug_${TIMESTAMP}}"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

COMMON_ARGS=(
  "$TASK"
  --env.commands.motion.motion-path "$MOTION_PATH"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.experiment-name "$EXPERIMENT_NAME"
  --agent.run-name "$RUN_NAME"
  --agent.wandb-project tracking_bfm
  --env.commands.motion.sampling-mode adaptive
  --env.commands.motion.adaptive-pre-failure-sample-window-steps 200
  --debug False
  --gpu-ids "[0]"
)

echo "[INFO] Launching manual 2-process DDP debug run"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] Logs: $LOG_DIR"

IFS=',' read -r -a CUDA_VISIBLE_DEVICES_LIST <<< "$CUDA_VISIBLE_DEVICES_VALUE"
if [[ "${#CUDA_VISIBLE_DEVICES_LIST[@]}" -ne 2 ]]; then
  echo "[ERROR] This debug script expects exactly 2 visible GPUs, got: $CUDA_VISIBLE_DEVICES_VALUE" >&2
  exit 1
fi

cleanup() {
  local exit_code=$?
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
  wait || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

env \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  WORLD_SIZE="$WORLD_SIZE" \
  RANK=0 \
  LOCAL_RANK=0 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST[0]}" \
  uv run train "${COMMON_ARGS[@]}" \
  >"$LOG_DIR/rank0.log" 2>&1 &
PID0=$!

env \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  WORLD_SIZE="$WORLD_SIZE" \
  RANK=1 \
  LOCAL_RANK=0 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST[1]}" \
  uv run train "${COMMON_ARGS[@]}" \
  >"$LOG_DIR/rank1.log" 2>&1 &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

echo "[INFO] rank0 exit code: $STATUS0"
echo "[INFO] rank1 exit code: $STATUS1"
echo "[INFO] rank0 log: $LOG_DIR/rank0.log"
echo "[INFO] rank1 log: $LOG_DIR/rank1.log"

if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "[ERROR] At least one rank failed. Inspect the per-rank logs above." >&2
  exit 1
fi
