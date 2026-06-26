#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

DISABLE_REG_AND_DR="${DISABLE_REG_AND_DR:-True}"
ATTENTION_VARIANT="${ATTENTION_VARIANT:-sparsetrack_full_ref}"

case "${DISABLE_REG_AND_DR,,}" in
  1|true|yes|on)
    case "${ATTENTION_VARIANT,,}" in
      mlp|none|false|off|0)
        DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR"
        DEFAULT_RUN_NAME="test_optimal_global_body_full_obs_no_reg_no_dr"
        ;;
      full_obs_causal)
        DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-FullObsCausalAttn-NoRegNoDR"
        DEFAULT_RUN_NAME="test_optimal_full_obs_causal_attn_no_reg_no_dr"
        ;;
      proprio_ref_cross)
        DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-ProprioRefCrossAttn-NoRegNoDR"
        DEFAULT_RUN_NAME="test_optimal_proprio_ref_cross_attn_no_reg_no_dr"
        ;;
      hist_proprio_cross)
        DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttn-NoRegNoDR"
        DEFAULT_RUN_NAME="test_optimal_hist_proprio_cross_attn_no_reg_no_dr"
        ;;
      sparsetrack_full_ref)
        DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR"
        DEFAULT_RUN_NAME="test_optimal_sparsetrack_full_ref_attn_no_reg_no_dr"
        ;;
      *)
        echo "ATTENTION_VARIANT must be one of mlp, full_obs_causal, proprio_ref_cross, hist_proprio_cross, sparsetrack_full_ref; got: $ATTENTION_VARIANT" >&2
        exit 2
        ;;
    esac
    ;;
  0|false|no|off)
    if [[ "${ATTENTION_VARIANT,,}" != "mlp" ]]; then
      echo "ATTENTION_VARIANT is only supported when DISABLE_REG_AND_DR=True" >&2
      exit 2
    fi
    DEFAULT_TASK="Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal"
    DEFAULT_RUN_NAME="test_optimal_global_body_full_obs_with_reg_dr"
    ;;
  *)
    echo "DISABLE_REG_AND_DR must be True or False, got: $DISABLE_REG_AND_DR" >&2
    exit 2
    ;;
esac

TASK="${TASK:-$DEFAULT_TASK}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/}"
NUM_ENVS="${NUM_ENVS:-8192}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_optimal_tracking_bfm}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-tracking_bfm}"
GPU_IDS="${GPU_IDS:-[5,6]}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
DEBUG="${DEBUG:-False}"
UPLOAD_MODEL="${UPLOAD_MODEL:-False}"

cmd=(
  uv run train "$TASK"
  --env.commands.motion.motion-path "$MOTION_PATH"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.experiment_name "$EXPERIMENT_NAME"
  --agent.run_name "$RUN_NAME"
  --agent.wandb_project "$WANDB_PROJECT"
  --env.commands.motion.sampling-mode adaptive
  --env.commands.motion.adaptive_pre_failure_sample_window_steps 200
  --env.commands.motion.history_steps 0
  --env.commands.motion.future_steps 1
  --agent.save_interval "$SAVE_INTERVAL"
  --debug "$DEBUG"
  --agent.upload-model "$UPLOAD_MODEL"
  --gpu_ids "$GPU_IDS"
)

exec "${cmd[@]}"
