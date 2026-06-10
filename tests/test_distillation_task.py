"""Tests specific to distillation tasks."""

from types import SimpleNamespace

import torch
import tyro

import mjlab
import mjlab.tasks.distillation.config.g1  # noqa: F401
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.distillation.mdp import commands as distill_commands
from mjlab.tasks.distillation.mdp.observations import build_student_actor_terms
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls


def test_distillation_task_is_registered() -> None:
  assert "Mjlab-Distillation-Flat-Unitree-G1" in list_tasks()


def test_distillation_wbteleop_obs_task_uses_wbteleop_student_obs() -> None:
  task_id = "Mjlab-DistillationWbteleopObs-Flat-Unitree-G1"

  assert task_id in list_tasks()

  env_cfg = load_env_cfg(task_id)
  wbteleop_cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop")
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)

  student_terms = env_cfg.observations["student_actor"].terms
  wbteleop_terms = wbteleop_cfg.observations["actor"].terms
  assert tuple(student_terms.keys()) == tuple(wbteleop_terms.keys())
  assert "ee_pose" not in student_terms
  assert "base_lin_vel_b" not in student_terms
  assert "motion_ref_ang_vel" in student_terms
  assert env_cfg.observations["student_actor"].enable_corruption is False
  assert rl_cfg.class_name == "DistillationRunner"
  assert rl_cfg.student_obs_group == "student_actor"
  assert runner_cls is not None


def test_latent_distillation_task_is_registered() -> None:
  task_id = "Mjlab-LatentDistillation-Flat-Unitree-G1"

  assert task_id in list_tasks()

  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)

  assert "teacher_actor" in env_cfg.observations
  assert "proprio_actor" in env_cfg.observations
  assert env_cfg.observations["proprio_actor"].enable_corruption is False
  assert rl_cfg.class_name == "DistillationRunner"
  assert rl_cfg.student_model_type == "latent"
  assert rl_cfg.student_obs_group == "student_actor"
  assert rl_cfg.encoder_obs_group == "teacher_actor"
  assert rl_cfg.decoder_obs_group == "proprio_actor"
  assert rl_cfg.latent_regularization == "kl"
  assert rl_cfg.mmd_weight == 0.0
  assert rl_cfg.mmd_max_samples == 1024
  assert rl_cfg.latent_smooth_max_pairs == 2048
  assert rl_cfg.sphere_orthonormal_weight == 0.0
  assert rl_cfg.sphere_knn_smooth_weight == 0.0
  assert rl_cfg.sphere_knn_k == 4
  assert rl_cfg.sphere_knn_max_samples == 2048
  assert runner_cls is not None


def test_distillation_task_loads_cfgs() -> None:
  task_id = "Mjlab-Distillation-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)

  assert "teacher_actor" in env_cfg.observations
  assert "student_actor" in env_cfg.observations
  assert env_cfg.observations["teacher_actor"].enable_corruption is False
  assert env_cfg.observations["student_actor"].enable_corruption is False
  student_actor_terms = env_cfg.observations["student_actor"].terms
  assert student_actor_terms["ee_pose"].params["history_steps"] == 0
  assert student_actor_terms["ee_pose"].params["future_steps"] == 1
  assert runner_cls is not None
  assert rl_cfg.class_name == "DistillationRunner"


def test_proprio_actor_excludes_command_terms() -> None:
  env_cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1")

  terms = set(env_cfg.observations["proprio_actor"].terms.keys())

  assert {"projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"} <= terms
  assert "ee_pose" not in terms
  assert "base_lin_vel_b" not in terms
  assert "base_ang_vel_b" not in terms
  assert "anchor_height_w" not in terms


def test_student_actor_robot_state_history_tracks_student_history_steps() -> None:
  terms = build_student_actor_terms(
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    anchor_body_name="pelvis",
    history_steps=5,
    future_steps=6,
  )

  assert terms["projected_gravity"].history_length == 6
  assert terms["base_ang_vel"].history_length == 6
  assert terms["joint_pos"].history_length == 6
  assert terms["joint_vel"].history_length == 6
  assert terms["actions"].history_length == 6
  assert "base_lin_vel_b" in terms
  assert "base_ang_vel_b" in terms


def test_student_base_velocity_body_terms_rotate_reference_velocity() -> None:
  yaw_90_quat = torch.tensor([[0.70710677, 0.0, 0.0, 0.70710677]])
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=("pelvis",)),
    body_quat_w=yaw_90_quat[:, None, :],
    body_lin_vel_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
    body_ang_vel_w=torch.tensor([[[0.0, 1.0, 0.0]]]),
  )
  env = SimpleNamespace(
    num_envs=1,
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )

  lin_vel_b = distill_commands.student_base_lin_vel_b(env, command_name="motion")
  ang_vel_b = distill_commands.student_base_ang_vel_b(env, command_name="motion")

  torch.testing.assert_close(lin_vel_b, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(ang_vel_b, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5, rtol=1e-5)


def test_distillation_obs_corruption_can_be_configured_independently() -> None:
  args = tyro.cli(
    TrainConfig,
    args=[
      "--env.observations.teacher_actor.enable_corruption",
      "True",
      "--env.observations.student_actor.enable_corruption",
      "False",
    ],
    default=TrainConfig.from_task("Mjlab-Distillation-Flat-Unitree-G1"),
    config=mjlab.TYRO_FLAGS,
  )

  assert args.env.observations["teacher_actor"].enable_corruption is True
  assert args.env.observations["student_actor"].enable_corruption is False
