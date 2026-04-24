"""Tests specific to distillation tasks."""

import mjlab.tasks.distillation.config.g1  # noqa: F401

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.distillation.mdp.observations import build_student_actor_terms


def test_distillation_task_is_registered() -> None:
  assert "Mjlab-Distillation-Flat-Unitree-G1" in list_tasks()


def test_distillation_task_loads_cfgs() -> None:
  task_id = "Mjlab-Distillation-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)

  assert "teacher_actor" in env_cfg.observations
  assert "student_actor" in env_cfg.observations
  student_actor_terms = env_cfg.observations["student_actor"].terms
  assert student_actor_terms["ee_pose"].params["history_steps"] == 0
  assert student_actor_terms["ee_pose"].params["future_steps"] == 1
  assert runner_cls is not None
  assert rl_cfg.class_name == "DistillationRunner"


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
