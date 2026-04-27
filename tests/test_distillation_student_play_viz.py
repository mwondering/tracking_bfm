"""Tests for distillation student sparse play visualization."""

from types import SimpleNamespace

import mjlab
import numpy as np
import torch
import tyro

import mjlab.tasks.distillation.config.g1  # noqa: F401
from mjlab.tasks.distillation.mdp import commands as distill_cmds
from mjlab.tasks.registry import load_env_cfg
from mjlab.scripts.play import (
  PlayCliConfig,
  PlayConfig,
  _configure_distillation_play_visualization,
)


class _MockCommandManager:
  def __init__(self, command):
    self._command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self._command


class _MockVisualizer:
  def __init__(self):
    self.env_idx = 0
    self.show_all_envs = False
    self.spheres = []
    self.arrows = []
    self.cylinders = []

  def get_env_indices(self, num_envs: int):
    return [0] if num_envs > 0 else []

  def add_sphere(self, center, radius, color, label=None):
    self.spheres.append((center, radius, color, label))

  def add_arrow(self, start, end, color, width=0.015, label=None):
    self.arrows.append((start, end, color, width, label))

  def add_cylinder(self, start, end, radius, color, label=None):
    self.cylinders.append((start, end, radius, color, label))


def _make_mock_env():
  command = SimpleNamespace(
    cfg=SimpleNamespace(
      body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
      history_steps=0,
      future_steps=1,
    ),
    anchor_pos_w=torch.tensor([[0.0, 0.0, 1.2]], dtype=torch.float32),
    anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    body_pos_w=torch.tensor(
      [[[0.0, 0.0, 1.0], [0.2, 0.1, 1.1], [-0.2, 0.15, 1.05]]], dtype=torch.float32
    ),
    body_quat_w=torch.tensor(
      [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
      dtype=torch.float32,
    ),
    body_lin_vel_w=torch.tensor(
      [[[0.5, -0.3, 0.1], [0.3, -0.1, 0.2], [0.2, 0.1, -0.2]]], dtype=torch.float32
    ),
    body_ang_vel_w=torch.tensor(
      [[[0.4, 0.2, -0.1], [0.0, 0.4, -0.2], [0.1, -0.2, 0.3]]], dtype=torch.float32
    ),
    anchor_lin_vel_w=torch.tensor([[0.3, -0.1, 0.2]], dtype=torch.float32),
    anchor_ang_vel_w=torch.tensor([[0.0, 0.4, -0.2]], dtype=torch.float32),
  )
  robot = SimpleNamespace(
    body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor(
        [[[1.0, 0.0, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32
      ),
      body_link_quat_w=torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
      ),
    ),
  )
  return SimpleNamespace(
    num_envs=1,
    command_manager=_MockCommandManager(command),
    scene={"robot": robot},
  )


def test_debug_vis_student_sparse_command_draws_expected_primitives() -> None:
  env = _make_mock_env()
  visualizer = _MockVisualizer()

  distill_cmds.debug_vis_student_sparse_command(env, visualizer)

  assert len(visualizer.spheres) == 2
  assert len(visualizer.arrows) == 2
  assert len(visualizer.cylinders) == 1
  assert visualizer.spheres[0][3] == "student_ref_left_ee_robot_pelvis_0"
  assert visualizer.spheres[1][3] == "student_ref_right_ee_robot_pelvis_0"
  np.testing.assert_allclose(visualizer.spheres[0][0], np.array([1.2, 0.1, 1.0]))
  np.testing.assert_allclose(visualizer.spheres[1][0], np.array([0.8, 0.15, 0.95]))
  np.testing.assert_allclose(visualizer.arrows[0][0], np.array([1.0, 0.0, 1.0]))
  np.testing.assert_allclose(visualizer.arrows[1][0], np.array([1.0, 0.0, 1.0]))
  np.testing.assert_allclose(visualizer.arrows[0][1], np.array([1.1, -0.06, 1.02]))
  np.testing.assert_allclose(visualizer.arrows[1][1], np.array([1.048, 0.024, 0.988]))
  np.testing.assert_allclose(visualizer.cylinders[0][0], np.array([1.0, 0.0, 0.0]))
  np.testing.assert_allclose(visualizer.cylinders[0][1], np.array([1.0, 0.0, 1.0]))
  assert visualizer.cylinders[0][4] == "student_ref_base_height_0"


def test_distillation_play_cfg_enables_motion_and_student_sparse_visualization() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1", play=True)

  assert "student_sparse_vis" in cfg.commands
  assert cfg.commands["motion"].debug_vis
  assert cfg.commands["student_sparse_vis"].debug_vis
  assert cfg.commands["student_sparse_vis"].ee_body_names == (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  assert cfg.commands["student_sparse_vis"].anchor_body_name == "pelvis"


def test_distillation_play_visualization_override_keeps_student_refs() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1", play=True)

  _configure_distillation_play_visualization(
    cfg, show_reference_motion=False
  )

  assert not cfg.commands["motion"].debug_vis
  assert cfg.commands["student_sparse_vis"].debug_vis


def test_play_config_parses_reference_motion_toggle() -> None:
  args = tyro.cli(
    PlayConfig,
    args=["--show-reference-motion", "False"],
    default=PlayConfig(),
    config=mjlab.TYRO_FLAGS,
  )

  assert not args.show_reference_motion


def test_play_cli_config_parses_student_observation_overrides() -> None:
  args = tyro.cli(
    PlayCliConfig,
    args=[
      "--env.observations.student_actor.terms.ee_pose.params.history_steps",
      "3",
      "--env.observations.student_actor.terms.ee_pose.params.future_steps",
      "4",
      "--env.observations.student_actor.terms.projected_gravity.history_length",
      "5",
    ],
    default=PlayCliConfig.from_task("Mjlab-Distillation-Flat-Unitree-G1"),
    config=mjlab.TYRO_FLAGS,
  )

  terms = args.env.observations["student_actor"].terms
  assert terms["ee_pose"].params["history_steps"] == 3
  assert terms["ee_pose"].params["future_steps"] == 4
  assert terms["projected_gravity"].history_length == 5
