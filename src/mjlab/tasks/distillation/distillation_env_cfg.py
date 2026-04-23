"""Common environment composition helpers for distillation tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_bfm_env_cfg
from mjlab.tasks.distillation.mdp.commands import StudentSparseCommandVisCfg
from mjlab.tasks.distillation.mdp.observations import build_student_actor_terms


@dataclass(frozen=True)
class StudentCommandConfig:
  command_name: str = "motion"
  ee_body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
  anchor_body_name: str = "pelvis"


def make_distillation_env_cfg(play: bool = False):
  """Build a distillation env cfg by composing the tracking BFM env cfg."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)
  student_command_cfg = StudentCommandConfig()

  teacher_actor = deepcopy(cfg.observations["actor"])
  student_actor = ObservationGroupCfg(
    terms=build_student_actor_terms(
      command_name=student_command_cfg.command_name,
      ee_body_names=student_command_cfg.ee_body_names,
      anchor_body_name=student_command_cfg.anchor_body_name,
    ),
    concatenate_terms=True,
    enable_corruption=not play,
  )

  cfg.observations["teacher_actor"] = teacher_actor
  cfg.observations["student_actor"] = student_actor

  if play:
    cfg.commands["motion"].debug_vis = True
    cfg.commands["student_sparse_vis"] = StudentSparseCommandVisCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      command_name=student_command_cfg.command_name,
      ee_body_names=student_command_cfg.ee_body_names,
      anchor_body_name=student_command_cfg.anchor_body_name,
    )
  return cfg
