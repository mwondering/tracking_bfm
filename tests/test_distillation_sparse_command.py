"""Tests for distillation sparse command extraction."""

from types import SimpleNamespace

import torch

from mjlab.tasks.distillation.mdp import commands as distill_cmds


class _MockCommandManager:
  def __init__(self, command):
    self._command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self._command


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
  env = SimpleNamespace(
    num_envs=1,
    command_manager=_MockCommandManager(command),
  )
  return env


def _make_multistep_env():
  body_names = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")
  body_pos_w = torch.tensor(
    [
      [[0.0, 0.0, 0.8], [0.1, 0.1, 0.9], [-0.1, 0.1, 0.85]],
      [[0.0, 0.0, 0.9], [0.2, 0.1, 1.0], [-0.2, 0.1, 0.95]],
      [[0.0, 0.0, 1.0], [0.3, 0.1, 1.1], [-0.3, 0.1, 1.05]],
      [[0.0, 0.0, 1.1], [0.4, 0.1, 1.2], [-0.4, 0.1, 1.15]],
      [[0.0, 0.0, 1.2], [0.5, 0.1, 1.3], [-0.5, 0.1, 1.25]],
    ],
    dtype=torch.float32,
  )
  body_quat_w = torch.tensor(
    [[[[1.0, 0.0, 0.0, 0.0]] * 3] * 5], dtype=torch.float32
  ).reshape(5, 3, 4)
  body_lin_vel_w = torch.tensor(
    [
      [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    dtype=torch.float32,
  )
  body_ang_vel_w = torch.tensor(
    [
      [[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.0, 0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.0, 0.4, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    dtype=torch.float32,
  )

  command = SimpleNamespace(
    cfg=SimpleNamespace(
      body_names=body_names,
      history_steps=0,
      future_steps=1,
    ),
    time_steps=torch.tensor([2], dtype=torch.long),
    motion_idx=torch.tensor([0], dtype=torch.long),
    _env=SimpleNamespace(
      scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0]]))
    ),
  )

  def _gather_motion_field(field_name: str, motion_idx, time_steps):
    del motion_idx
    field = {
      "body_pos_w": body_pos_w,
      "body_quat_w": body_quat_w,
      "body_lin_vel_w": body_lin_vel_w,
      "body_ang_vel_w": body_ang_vel_w,
    }[field_name]
    return field[time_steps]

  command._gather_motion_field = _gather_motion_field
  env = SimpleNamespace(
    num_envs=1,
    command_manager=_MockCommandManager(command),
  )
  return env


def test_student_sparse_command_dim() -> None:
  env = _make_mock_env()

  out = distill_cmds.student_sparse_command(
    env,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    history_steps=0,
    future_steps=1,
  )

  assert out.shape == (1, 25)


def test_student_anchor_height_matches_anchor_z() -> None:
  env = _make_mock_env()

  out = distill_cmds.student_anchor_height_w(env, command_name="motion")

  assert torch.allclose(out.squeeze(-1), torch.tensor([1.0], dtype=torch.float32))


def test_student_ee_pose_contains_two_end_effectors() -> None:
  env = _make_mock_env()

  out = distill_cmds.student_ee_pose_b(
    env,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
  )

  assert out.shape == (1, 18)
  assert torch.allclose(out[0, :3], torch.tensor([0.2, 0.1, 0.1]))


def test_student_base_velocity_uses_pelvis_body_reference() -> None:
  env = _make_mock_env()

  lin = distill_cmds.student_base_lin_vel_w(env, command_name="motion")
  ang = distill_cmds.student_base_ang_vel_w(env, command_name="motion")

  assert lin.shape == (1, 3)
  assert ang.shape == (1, 3)
  assert torch.allclose(lin[0], torch.tensor([0.5, -0.3, 0.1]))
  assert torch.allclose(ang[0], torch.tensor([0.4, 0.2, -0.1]))


def test_student_command_history_and_future_stack_requested_steps() -> None:
  env = _make_multistep_env()

  heights = distill_cmds.student_anchor_height_w(
    env,
    command_name="motion",
    history_steps=1,
    future_steps=3,
  )
  lin = distill_cmds.student_base_lin_vel_w(
    env,
    command_name="motion",
    history_steps=1,
    future_steps=3,
  )
  ang = distill_cmds.student_base_ang_vel_w(
    env,
    command_name="motion",
    history_steps=1,
    future_steps=3,
  )

  assert heights.shape == (1, 4)
  assert lin.shape == (1, 12)
  assert ang.shape == (1, 12)
  assert torch.allclose(heights[0], torch.tensor([0.9, 1.0, 1.1, 1.2]))
  assert torch.allclose(lin[0], torch.tensor([0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0, 0.0, 0.5, 0.0, 0.0]))
  assert torch.allclose(ang[0], torch.tensor([0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0, 0.0, 0.5, 0.0]))


def test_student_sparse_command_multistep_dim() -> None:
  env = _make_multistep_env()

  out = distill_cmds.student_sparse_command(
    env,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    history_steps=1,
    future_steps=3,
  )

  assert out.shape == (1, 100)


def test_student_command_multistep_is_decoupled_from_motion_cfg_steps() -> None:
  env = _make_multistep_env()
  heights = distill_cmds.student_anchor_height_w(
    env,
    command_name="motion",
    history_steps=1,
    future_steps=3,
  )

  assert torch.allclose(heights[0], torch.tensor([0.9, 1.0, 1.1, 1.2]))
