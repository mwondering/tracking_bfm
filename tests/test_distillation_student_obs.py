"""Tests for distillation student observation definitions."""

import mjlab.tasks.distillation.config.g1  # noqa: F401

from mjlab.tasks.registry import load_env_cfg


def test_student_obs_terms_are_exact() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1")
  terms = set(cfg.observations["student_actor"].terms.keys())

  assert terms == {
    "ee_pose",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "anchor_height_w",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  }


def test_student_obs_body_selection_is_configured_from_env_cfg() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1")
  terms = cfg.observations["student_actor"].terms

  assert terms["ee_pose"].params["ee_body_names"] == (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  assert terms["ee_pose"].params["anchor_body_name"] == "pelvis"
  assert terms["base_lin_vel_w"].params["anchor_body_name"] == "pelvis"
  assert terms["base_ang_vel_w"].params["anchor_body_name"] == "pelvis"
  assert terms["anchor_height_w"].params["anchor_body_name"] == "pelvis"
