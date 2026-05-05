#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
#/home/lenovo/DATASETS/test_motion
#/home/lenovo/DATASETS/Data10k
TASK="${TASK:-Mjlab-Distillation-Flat-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/data/zcy/motion_data/AMASS_LAFAN_Qingtong}"
TEACHER_CKPT="${TEACHER_CKPT:-/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_filteredsonic/2026-04-25_11-04-39_multi_gpu_adaptive_resume/model_102000.pt}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
BETA_DECAY_STEPS="${BETA_DECAY_STEPS:-1}"
STUDENT_HISTORY_STEPS="${STUDENT_HISTORY_STEPS:-0}"
STUDENT_FUTURE_STEPS="${STUDENT_FUTURE_STEPS:-1}"
STUDENT_ROBOT_HISTORY_STEPS="${STUDENT_ROBOT_HISTORY_STEPS:-20}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_distillation}"
RUN_NAME="${RUN_NAME:-distill_multi_gpu}"

uv run train "$TASK" \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs "$NUM_ENVS" \
    --env.commands.motion.sampling-mode uniform \
    --env.commands.motion.history_steps "$STUDENT_HISTORY_STEPS" \
    --env.commands.motion.future_steps "$STUDENT_FUTURE_STEPS" \
    --env.observations.student_actor.terms.ee_pose.params.history_steps "$STUDENT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.ee_pose.params.future_steps "$STUDENT_FUTURE_STEPS" \
    --env.observations.student_actor.terms.base_lin_vel_b.params.history_steps "$STUDENT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.base_lin_vel_b.params.future_steps "$STUDENT_FUTURE_STEPS" \
    --env.observations.student_actor.terms.base_ang_vel_b.params.history_steps "$STUDENT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.base_ang_vel_b.params.future_steps "$STUDENT_FUTURE_STEPS" \
    --env.observations.student_actor.terms.anchor_height_w.params.history_steps "$STUDENT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.anchor_height_w.params.future_steps "$STUDENT_FUTURE_STEPS" \
    --env.observations.student_actor.terms.projected_gravity.history_length "$STUDENT_ROBOT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.base_ang_vel.history_length "$STUDENT_ROBOT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.joint_pos.history_length "$STUDENT_ROBOT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.joint_vel.history_length "$STUDENT_ROBOT_HISTORY_STEPS" \
    --env.observations.student_actor.terms.actions.history_length "$STUDENT_ROBOT_HISTORY_STEPS" \
    --env.commands.motion.adaptive_pre_failure_sample_window_steps 200 \
    --agent.teacher_checkpoint_path "$TEACHER_CKPT" \
    --agent.max_iterations "$MAX_ITERATIONS" \
    --agent.num_steps_per_env "$NUM_STEPS_PER_ENV" \
    --agent.save_interval "$SAVE_INTERVAL" \
    --agent.beta_decay_steps "$BETA_DECAY_STEPS" \
    --agent.experiment_name "$EXPERIMENT_NAME" \
    --agent.run_name "$RUN_NAME" \
    --agent.wandb_project "tracking_bfm_distillation" \
    --debug False \
    --agent.upload-model False \
    --gpu_ids "[4,5]"
