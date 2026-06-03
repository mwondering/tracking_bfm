"""Observation terms for the G1 BFM wbteleop tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.multi_commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_ref_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return reference anchor angular velocity from the motion command window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.anchor_ang_vel_w
