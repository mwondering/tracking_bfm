from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import commands as distill_commands

def build_student_actor_terms(
  *,
  command_name: str,
  ee_body_names: tuple[str, str],
  anchor_body_name: str,
) -> dict[str, ObservationTermCfg]:
  return {
    "ee_pose": ObservationTermCfg(
      func=distill_commands.student_ee_pose_b,
      params={
        "command_name": command_name,
        "ee_body_names": ee_body_names,
        "anchor_body_name": anchor_body_name,
      },
    ),
    "base_lin_vel_w": ObservationTermCfg(
      func=distill_commands.student_base_lin_vel_w,
      params={
        "command_name": command_name,
        "anchor_body_name": anchor_body_name,
      },
    ),
    "base_ang_vel_w": ObservationTermCfg(
      func=distill_commands.student_base_ang_vel_w,
      params={
        "command_name": command_name,
        "anchor_body_name": anchor_body_name,
      },
    ),
    "anchor_height_w": ObservationTermCfg(
      func=distill_commands.student_anchor_height_w,
      params={
        "command_name": command_name,
        "anchor_body_name": anchor_body_name,
      },
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
