"""Tests for latent-action tracking task configuration."""

from __future__ import annotations

import os
import subprocess

from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage"
BASE_TASK_ID = "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage"


def test_latent_tracking_1stage_task_is_registered() -> None:
  from mjlab.tasks.latenttracking.rl import LatentTrackingOnPolicyRunner

  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is LatentTrackingOnPolicyRunner
  assert issubclass(LatentTrackingOnPolicyRunner, MotionTrackingOnPolicyRunner)

  rl_cfg = load_rl_cfg(TASK_ID)
  assert rl_cfg.latent_dim == 64
  assert rl_cfg.latent_action_clip == 6.0
  assert rl_cfg.latent_decoder_checkpoint_path == ""
  assert rl_cfg.proprio_obs_group == "proprio_actor"
  assert rl_cfg.resume is False
  assert rl_cfg.load_run == ".*"
  assert rl_cfg.load_checkpoint == "model_.*.pt"


def test_latent_tracking_1stage_keeps_base_non_reward_config() -> None:
  base = load_env_cfg(BASE_TASK_ID)
  cfg = load_env_cfg(TASK_ID)

  assert cfg.commands.keys() == base.commands.keys()
  assert cfg.terminations.keys() == base.terminations.keys()
  assert cfg.events.keys() == base.events.keys()
  assert cfg.observations["actor"].terms.keys() == base.observations["actor"].terms.keys()
  assert cfg.observations["critic"].terms.keys() == base.observations["critic"].terms.keys()


def test_latent_tracking_1stage_uses_sparse_tracking_rewards() -> None:
  from mjlab.tasks.tracking import mdp

  cfg = load_env_cfg(TASK_ID)

  assert tuple(cfg.rewards.keys()) == (
    "sparse_ee_pos",
    "sparse_ee_ori",
    "sparse_root_lin_vel",
    "sparse_root_ang_vel",
    "sparse_root_height",
  )
  assert cfg.rewards["sparse_ee_pos"].func is mdp.motion_relative_body_position_error_exp
  assert cfg.rewards["sparse_ee_ori"].func is mdp.motion_relative_body_orientation_error_exp
  assert (
    cfg.rewards["sparse_root_lin_vel"].func
    is mdp.motion_global_body_linear_velocity_error_exp
  )
  assert (
    cfg.rewards["sparse_root_ang_vel"].func
    is mdp.motion_global_body_angular_velocity_error_exp
  )
  assert cfg.rewards["sparse_root_height"].func is mdp.motion_global_body_height_error_exp

  assert cfg.rewards["sparse_ee_pos"].params["body_names"] == (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  assert cfg.rewards["sparse_ee_ori"].params["body_names"] == (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  assert cfg.rewards["sparse_root_lin_vel"].params["body_names"] == ("pelvis",)
  assert cfg.rewards["sparse_root_ang_vel"].params["body_names"] == ("pelvis",)
  assert cfg.rewards["sparse_root_height"].params["body_name"] == "pelvis"


def test_latent_tracking_1stage_adds_decoder_proprio_observations() -> None:
  cfg = load_env_cfg(TASK_ID)

  assert "proprio_actor" in cfg.observations
  proprio_terms = cfg.observations["proprio_actor"].terms
  assert tuple(proprio_terms.keys()) == (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert cfg.observations["proprio_actor"].concatenate_terms is True
  assert cfg.observations["proprio_actor"].enable_corruption is False
  for term in proprio_terms.values():
    assert term.history_length == 0


def test_latent_tracking_1stage_keeps_asymmetric_critic_observations() -> None:
  base = load_env_cfg(BASE_TASK_ID)
  cfg = load_env_cfg(TASK_ID)

  assert cfg.observations["critic"].terms.keys() == base.observations["critic"].terms.keys()
  assert cfg.observations["critic"].terms.keys() != cfg.observations["actor"].terms.keys()


def test_latent_tracking_1stage_help_exposes_decoder_and_resume_flags() -> None:
  result = subprocess.run(
    ["uv", "run", "train", TASK_ID, "--help"],
    check=True,
    capture_output=True,
    text=True,
    env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache-tracking-bfm-tests"},
  )

  assert "--agent.latent-decoder-checkpoint-path" in result.stdout
  assert "--agent.latent-action-clip" in result.stdout
  assert "--agent.proprio-obs-group" in result.stdout
  assert "--agent.resume" in result.stdout
  assert "--agent.load-run" in result.stdout
  assert "--agent.load-checkpoint" in result.stdout
