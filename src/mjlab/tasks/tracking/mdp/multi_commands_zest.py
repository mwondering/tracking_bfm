from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]


class MultiMotionLoader:
  def __init__(
    self, motion_files: list[str], body_indexes: torch.Tensor, device: str = "cpu"
  ):
    assert len(motion_files) > 0, "motion_files cannot be empty"
    self.num_files = len(motion_files)
    self._body_indexes = body_indexes
    self.device = device

    # Keep each motion separate so variable-length clips stay ragged.
    self.joint_pos_list = []
    self.joint_vel_list = []
    self.body_pos_w_list = []
    self.body_quat_w_list = []
    self.body_lin_vel_w_list = []
    self.body_ang_vel_w_list = []
    self.fps_list = []
    self.file_lengths = []

    self.isaaclab_joint_names = [
      "left_hip_pitch_joint",
      "right_hip_pitch_joint",
      "waist_yaw_joint",
      "left_hip_roll_joint",
      "right_hip_roll_joint",
      "waist_roll_joint",
      "left_hip_yaw_joint",
      "right_hip_yaw_joint",
      "waist_pitch_joint",
      "left_knee_joint",
      "right_knee_joint",
      "left_shoulder_pitch_joint",
      "right_shoulder_pitch_joint",
      "left_ankle_pitch_joint",
      "right_ankle_pitch_joint",
      "left_shoulder_roll_joint",
      "right_shoulder_roll_joint",
      "left_ankle_roll_joint",
      "right_ankle_roll_joint",
      "left_shoulder_yaw_joint",
      "right_shoulder_yaw_joint",
      "left_elbow_joint",
      "right_elbow_joint",
      "left_wrist_roll_joint",
      "right_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "right_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_wrist_yaw_joint",
    ]

    self.mujoco_joint_names = [
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ]

    self.isaaclab_body_names = [
      "pelvis",
      "left_hip_pitch_link",
      "right_hip_pitch_link",
      "waist_yaw_link",
      "left_hip_roll_link",
      "right_hip_roll_link",
      "waist_roll_link",
      "left_hip_yaw_link",
      "right_hip_yaw_link",
      "torso_link",
      "left_knee_link",
      "right_knee_link",
      "left_shoulder_pitch_link",
      "right_shoulder_pitch_link",
      "left_ankle_pitch_link",
      "right_ankle_pitch_link",
      "left_shoulder_roll_link",
      "right_shoulder_roll_link",
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_shoulder_yaw_link",
      "right_shoulder_yaw_link",
      "left_elbow_link",
      "right_elbow_link",
      "left_wrist_roll_link",
      "right_wrist_roll_link",
      "left_wrist_pitch_link",
      "right_wrist_pitch_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ]
    self.mujoco_body_names = [
      "pelvis",
      "left_hip_pitch_link",
      "left_hip_roll_link",
      "left_hip_yaw_link",
      "left_knee_link",
      "left_ankle_pitch_link",
      "left_ankle_roll_link",
      "right_hip_pitch_link",
      "right_hip_roll_link",
      "right_hip_yaw_link",
      "right_knee_link",
      "right_ankle_pitch_link",
      "right_ankle_roll_link",
      "waist_yaw_link",
      "waist_roll_link",
      "torso_link",
      "left_shoulder_pitch_link",
      "left_shoulder_roll_link",
      "left_shoulder_yaw_link",
      "left_elbow_link",
      "left_wrist_roll_link",
      "left_wrist_pitch_link",
      "left_wrist_yaw_link",
      "right_shoulder_pitch_link",
      "right_shoulder_roll_link",
      "right_shoulder_yaw_link",
      "right_elbow_link",
      "right_wrist_roll_link",
      "right_wrist_pitch_link",
      "right_wrist_yaw_link",
    ]

    self.isaaclab_to_mujoco_joint_reindex = [
      self.isaaclab_joint_names.index(name) for name in self.mujoco_joint_names
    ]
    self.mujoco_to_isaaclab_joint_reindex = [
      self.mujoco_joint_names.index(name) for name in self.isaaclab_joint_names
    ]
    self.isaaclab_to_mujoco_body_reindex = [
      self.isaaclab_body_names.index(name) for name in self.mujoco_body_names
    ]
    self.mujoco_to_isaaclab_body_reindex = [
      self.mujoco_body_names.index(name) for name in self.isaaclab_body_names
    ]

    for motion_file in motion_files:
      assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
      data = np.load(motion_file)

      self.fps_list.append(float(np.asarray(data["fps"]).reshape(-1)[0]))

      jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)[
        :, self.isaaclab_to_mujoco_joint_reindex
      ]
      jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)[
        :, self.isaaclab_to_mujoco_joint_reindex
      ]
      bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)[
        :, self.isaaclab_to_mujoco_body_reindex, :
      ]
      bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)[
        :, self.isaaclab_to_mujoco_body_reindex, :
      ]
      blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)[
        :, self.isaaclab_to_mujoco_body_reindex, :
      ]
      bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)[
        :, self.isaaclab_to_mujoco_body_reindex, :
      ]

      self.joint_pos_list.append(jp)
      self.joint_vel_list.append(jv)
      self.body_pos_w_list.append(bp[:, self._body_indexes])
      self.body_quat_w_list.append(bq[:, self._body_indexes])
      self.body_lin_vel_w_list.append(blv[:, self._body_indexes])
      self.body_ang_vel_w_list.append(bav[:, self._body_indexes])
      self.file_lengths.append(jp.shape[0])

    self.file_lengths = torch.tensor(
      self.file_lengths, dtype=torch.long, device=self.device
    )
    self.frame_offsets = torch.cumsum(self.file_lengths, dim=0) - self.file_lengths
    self.max_length = int(self.file_lengths.max().item())
    self.total_frames = int(self.file_lengths.sum().item())
    self.joint_pos_flat = torch.cat(self.joint_pos_list, dim=0)
    self.joint_vel_flat = torch.cat(self.joint_vel_list, dim=0)
    self.body_pos_w_flat = torch.cat(self.body_pos_w_list, dim=0)
    self.body_quat_w_flat = torch.cat(self.body_quat_w_list, dim=0)
    self.body_lin_vel_w_flat = torch.cat(self.body_lin_vel_w_list, dim=0)
    self.body_ang_vel_w_flat = torch.cat(self.body_ang_vel_w_list, dim=0)
    self.joint_pos_list = []
    self.joint_vel_list = []
    self.body_pos_w_list = []
    self.body_quat_w_list = []
    self.body_lin_vel_w_list = []
    self.body_ang_vel_w_list = []

class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    motion_files = self._resolve_motion_files(self.cfg.motion_file)
    motion_tensor_device = (
      self.device if self.cfg.motion_storage_device == "device" else "cpu"
    )
    self.motions = MultiMotionLoader(
      motion_files,
      self.body_indexes.to(motion_tensor_device),
      device=motion_tensor_device,
    )
    self.motion_fps = torch.tensor(
      self.motions.fps_list, dtype=torch.float32, device=self.device
    )
    self.motion_lengths = self.motions.file_lengths.to(self.device)
    self.motion_frame_starts = self.motions.frame_offsets.to(self.device)
    self.motion_frame_ends = self.motion_frame_starts + self.motion_lengths
    self.motion_durations_s = self.motion_lengths.float() / self.motion_fps
    self.max_episode_steps = (
      int(self._env.max_episode_length_s / self._env.step_dt)
      if self._env.max_episode_length_s > 0
      else int(self.motion_lengths.max().item())
    )
    self._all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    self._validate_actor_obs_mask_cfg()
    self.actor_obs_mask: dict[str, torch.Tensor] = {}
    self.actor_obs_mask_steps_remaining = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._init_actor_obs_masks()
    self.motion_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.buffer_start_time = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0
    self.sampled_bin_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.segment_max_steps = torch.ones(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.segment_similarity_sum = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.segment_step_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._initialize_adaptive_sampler()
    self._init_motion_buffers()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _resolve_motion_files(self, motion_files: str | list[str]) -> list[str]:
    if isinstance(motion_files, (str, os.PathLike)):
      motion_path = os.fspath(motion_files)
      if os.path.isdir(motion_path):
        resolved_motion_files = []
        for root, _, files in os.walk(motion_path):
          for filename in files:
            if filename.lower().endswith(".npz"):
              resolved_motion_files.append(os.path.join(root, filename))
        resolved_motion_files.sort()
      elif os.path.isfile(motion_path):
        resolved_motion_files = [motion_path]
      else:
        raise ValueError(f"motion_file path does not exist: {motion_path}")
    elif isinstance(motion_files, (list, tuple)):
      resolved_motion_files = list(motion_files)
    else:
      raise TypeError("motion_file must be a path or list of paths")

    if len(resolved_motion_files) == 0:
      raise ValueError("motion_file did not resolve to any .npz motion files")
    return resolved_motion_files

  def _validate_actor_obs_mask_cfg(self) -> None:
    if not self.cfg.actor_obs_mask_enabled:
      return
    if len(self.cfg.actor_obs_mask_probabilities) == 0:
      raise ValueError("actor_obs_mask_probabilities must be set when masking is enabled")
    min_steps, max_steps = self.cfg.actor_obs_mask_steps_range
    if min_steps <= 0 or max_steps <= 0 or min_steps > max_steps:
      raise ValueError(
        "actor_obs_mask_steps_range must be positive and satisfy min <= max"
      )
    for term_name, prob in self.cfg.actor_obs_mask_probabilities.items():
      if not 0.0 <= prob <= 1.0:
        raise ValueError(
          f"actor_obs_mask probability for '{term_name}' must be in [0, 1]"
        )

  def _init_actor_obs_masks(self) -> None:
    mask_terms = self.cfg.actor_obs_mask_probabilities.keys()
    self.actor_obs_mask = {
      term_name: torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
      for term_name in mask_terms
    }
    self._resample_actor_obs_masks(self._all_env_ids)

  def _resample_actor_obs_masks(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    if not self.cfg.actor_obs_mask_enabled:
      self.actor_obs_mask_steps_remaining[env_ids] = 1
      for mask in self.actor_obs_mask.values():
        mask[env_ids] = True
      return

    term_names = tuple(self.cfg.actor_obs_mask_probabilities.keys())
    for term_name in term_names:
      prob = self.cfg.actor_obs_mask_probabilities[term_name]
      self.actor_obs_mask[term_name][env_ids] = (
        torch.rand(len(env_ids), device=self.device) >= prob
      )

    if term_names:
      stacked_mask = torch.stack(
        [self.actor_obs_mask[term_name][env_ids] for term_name in term_names], dim=1
      )
      fully_masked = ~stacked_mask.any(dim=1)
      if fully_masked.any():
        replacement_term_ids = torch.randint(
          0, len(term_names), (int(fully_masked.sum().item()),), device=self.device
        )
        replacement_env_ids = env_ids[fully_masked]
        for i, term_id in enumerate(replacement_term_ids.tolist()):
          self.actor_obs_mask[term_names[term_id]][replacement_env_ids[i]] = True

    min_steps, max_steps = self.cfg.actor_obs_mask_steps_range
    self.actor_obs_mask_steps_remaining[env_ids] = torch.randint(
      min_steps, max_steps + 1, (len(env_ids),), device=self.device
    )

  def mask_actor_observation(
    self, term_name: str, observation: torch.Tensor
  ) -> torch.Tensor:
    visible = self.actor_obs_mask.get(term_name)
    if visible is None:
      visible_tensor = torch.ones(
        self.num_envs, 1, dtype=observation.dtype, device=observation.device
      )
    else:
      visible_tensor = visible.to(
        dtype=observation.dtype, device=observation.device
      ).unsqueeze(-1)

    masked_observation = observation * visible_tensor
    if self.cfg.actor_obs_append_mask:
      return torch.cat([masked_observation, visible_tensor], dim=-1)
    return masked_observation

  def _init_motion_buffers(self) -> None:
    self.buffer_length = max(
      1, min(self.cfg.motion_buffer_size, int(self.motion_lengths.max().item()))
    )
    self._buffer_offsets = torch.arange(
      self.buffer_length, dtype=torch.long, device=self.device
    )
    num_bodies = len(self.cfg.body_names)
    joint_dim = self.motions.joint_pos_flat.shape[1]
    self.joint_pos_buffer = torch.zeros(
      self.num_envs, self.buffer_length, joint_dim, device=self.device
    )
    self.joint_vel_buffer = torch.zeros(
      self.num_envs, self.buffer_length, joint_dim, device=self.device
    )
    self.body_pos_w_buffer = torch.zeros(
      self.num_envs, self.buffer_length, num_bodies, 3, device=self.device
    )
    self.body_quat_w_buffer = torch.zeros(
      self.num_envs, self.buffer_length, num_bodies, 4, device=self.device
    )
    self.body_lin_vel_w_buffer = torch.zeros(
      self.num_envs, self.buffer_length, num_bodies, 3, device=self.device
    )
    self.body_ang_vel_w_buffer = torch.zeros(
      self.num_envs, self.buffer_length, num_bodies, 3, device=self.device
    )
    self.body_quat_w_buffer[..., 0] = 1.0

  def _current_buffer_indices(self) -> torch.Tensor:
    return torch.clamp(
      self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
    )

  def _gather_motion_tensor(
    self, flat_tensor: torch.Tensor, global_frame_indices: torch.Tensor
  ) -> torch.Tensor:
    index_tensor = (
      global_frame_indices.cpu()
      if flat_tensor.device.type == "cpu"
      else global_frame_indices.to(flat_tensor.device)
    )
    gathered = flat_tensor[index_tensor]
    if gathered.device != torch.device(self.device):
      gathered = gathered.to(self.device)
    return gathered

  def _refresh_motion_buffers(
    self, env_ids: torch.Tensor, start_steps: torch.Tensor | None = None
  ) -> None:
    if env_ids.numel() == 0:
      return

    motion_ids = self.motion_index[env_ids]
    motion_lengths = self.motion_lengths[motion_ids]
    if start_steps is None:
      start_steps = self.time_steps[env_ids]
    start_steps = torch.minimum(start_steps, motion_lengths - 1)
    self.buffer_start_time[env_ids] = start_steps

    frame_steps = start_steps[:, None] + self._buffer_offsets[None, :]
    frame_steps = torch.minimum(frame_steps, motion_lengths[:, None] - 1)
    global_frame_indices = self.motion_frame_starts[motion_ids, None] + frame_steps

    self.joint_pos_buffer[env_ids] = self._gather_motion_tensor(
      self.motions.joint_pos_flat, global_frame_indices
    )
    self.joint_vel_buffer[env_ids] = self._gather_motion_tensor(
      self.motions.joint_vel_flat, global_frame_indices
    )
    self.body_pos_w_buffer[env_ids] = self._gather_motion_tensor(
      self.motions.body_pos_w_flat, global_frame_indices
    )
    self.body_quat_w_buffer[env_ids] = self._gather_motion_tensor(
      self.motions.body_quat_w_flat, global_frame_indices
    )
    self.body_lin_vel_w_buffer[env_ids] = self._gather_motion_tensor(
      self.motions.body_lin_vel_w_flat, global_frame_indices
    )
    self.body_ang_vel_w_buffer[env_ids] = self._gather_motion_tensor(
      self.motions.body_ang_vel_w_flat, global_frame_indices
    )

  def _current_motion_lengths(self) -> torch.Tensor:
    return self.motion_lengths[self.motion_index]

  def _current_time_bins(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    time_s = time_steps.float() / self.motion_fps[motion_ids]
    bin_indices = torch.floor(time_s / self.bin_width_s).long()
    valid_bin_counts = self.motion_valid_bin_counts[motion_ids]
    return torch.minimum(bin_indices, torch.clamp(valid_bin_counts - 1, min=0))

  def _set_motion_frames(
    self, env_ids: torch.Tensor, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> None:
    self.motion_index[env_ids] = motion_ids
    self.time_steps[env_ids] = time_steps

  def _initialize_adaptive_sampler(self) -> None:
    min_duration_s = self.motion_durations_s.min()
    self.bin_width_s = min(self.cfg.adaptive_bin_width_s, float(min_duration_s.item()))
    if self.bin_width_s <= 0.0:
      raise ValueError("adaptive_bin_width_s must be positive")
    if self.cfg.adaptive_base_temperature <= 0.0:
      raise ValueError("adaptive_base_temperature must be positive")
    if not 0.0 <= self.cfg.adaptive_uniform_floor < 1.0:
      raise ValueError("adaptive_uniform_floor must be in [0, 1)")

    self.bin_count = max(
      1, int(torch.ceil(self.motion_durations_s.max() / self.bin_width_s).item())
    )
    bin_starts = (
      torch.arange(self.bin_count, device=self.device, dtype=torch.float32)
      * self.bin_width_s
    )
    self.valid_bin_mask = bin_starts.unsqueeze(0) < self.motion_durations_s.unsqueeze(1)
    self.motion_valid_bin_counts = self.valid_bin_mask.sum(dim=1)
    self.failure_levels = torch.zeros(
      (self.motions.num_files, self.bin_count),
      dtype=torch.float32,
      device=self.device,
    )
    self.failure_levels[~self.valid_bin_mask] = -torch.inf
    self.valid_pairs = self.valid_bin_mask.nonzero(as_tuple=False)
    if self.valid_pairs.numel() == 0:
      raise RuntimeError("Adaptive sampler found no valid motion bins.")
    self.valid_bin_count = int(self.valid_pairs.shape[0])
    self.adaptive_temperature = self.cfg.adaptive_base_temperature / math.log(
      1.0 + self.valid_bin_count
    )
    self.adaptive_similarity_std = self.cfg.adaptive_similarity_std
    if self.adaptive_similarity_std is None:
      self.adaptive_similarity_std = 0.15 * math.sqrt(self.robot.num_joints)

  def _compute_adaptive_similarity(self) -> torch.Tensor:
    joint_error = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
    return torch.exp(
      -joint_error / (float(self.adaptive_similarity_std) ** 2 + 1e-8)
    )

  def _update_failure_levels(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    valid_envs = env_ids[self.segment_step_count[env_ids] > 0]
    if valid_envs.numel() == 0:
      return

    similarity = self.segment_similarity_sum[valid_envs] / torch.clamp(
      self.segment_max_steps[valid_envs].float(), min=1.0
    )
    failure_score = 1.0 - similarity
    motion_ids = self.motion_index[valid_envs]
    bin_ids = self.sampled_bin_idx[valid_envs]
    old_levels = self.failure_levels[motion_ids, bin_ids]
    self.failure_levels[motion_ids, bin_ids] = (
      (1.0 - self.cfg.adaptive_alpha) * old_levels
      + self.cfg.adaptive_alpha * failure_score
    )

  def _reset_segment_stats(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self.segment_similarity_sum[env_ids] = 0.0
    self.segment_step_count[env_ids] = 0

  def _sample_adaptive_pairs(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    logits = self.failure_levels[self.valid_bin_mask] / self.adaptive_temperature
    softmax_probs = torch.softmax(logits, dim=0)
    floor = self.cfg.adaptive_uniform_floor / float(self.valid_bin_count)
    probabilities = (1.0 - self.cfg.adaptive_uniform_floor) * softmax_probs + floor
    sampled_pair_indices = torch.multinomial(
      probabilities, len(env_ids), replacement=True
    )
    sampled_pairs = self.valid_pairs[sampled_pair_indices]
    sampled_motion_ids = sampled_pairs[:, 0]
    sampled_bin_ids = sampled_pairs[:, 1]

    entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    entropy_denom = math.log(self.valid_bin_count) if self.valid_bin_count > 1 else 1.0
    entropy_norm = entropy / entropy_denom
    top_prob, _ = probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = entropy_norm
    self.metrics["sampling_top1_prob"][:] = top_prob
    return sampled_motion_ids, sampled_bin_ids

  def _sample_start_motion_indices(self, count: int) -> torch.Tensor:
    return torch.randint(0, self.motions.num_files, (count,), device=self.device)
  ###command term 
  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.joint_pos_buffer[self._all_env_ids, self._current_buffer_indices()]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.joint_vel_buffer[self._all_env_ids, self._current_buffer_indices()]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.body_pos_w_buffer[self._all_env_ids, self._current_buffer_indices()]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.body_quat_w_buffer[self._all_env_ids, self._current_buffer_indices()]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w_buffer[
      self._all_env_ids, self._current_buffer_indices()
    ]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w_buffer[
      self._all_env_ids, self._current_buffer_indices()
    ]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.body_pos_w_buffer[
        self._all_env_ids, self._current_buffer_indices(), self.motion_anchor_body_index
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w_buffer[
      self._all_env_ids, self._current_buffer_indices(), self.motion_anchor_body_index
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w_buffer[
      self._all_env_ids, self._current_buffer_indices(), self.motion_anchor_body_index
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w_buffer[
      self._all_env_ids, self._current_buffer_indices(), self.motion_anchor_body_index
    ]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )
    if self.cfg.sampling_mode == "adaptive":
      similarity = self._compute_adaptive_similarity()
      self.segment_similarity_sum += similarity
      self.segment_step_count += 1

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    sampled_motion_ids, sampled_bin_ids = self._sample_adaptive_pairs(env_ids)
    bin_start_s = sampled_bin_ids.float() * self.bin_width_s
    bin_end_s = torch.minimum(
      (sampled_bin_ids.float() + 1.0) * self.bin_width_s,
      self.motion_durations_s[sampled_motion_ids],
    )
    sampled_time_s = sample_uniform(
      bin_start_s, bin_end_s, (len(env_ids),), device=self.device
    )
    sampled_time_steps = torch.floor(
      sampled_time_s * self.motion_fps[sampled_motion_ids]
    ).long()
    sampled_time_steps = torch.minimum(
      sampled_time_steps, self.motion_lengths[sampled_motion_ids] - 1
    )
    self._set_motion_frames(env_ids, sampled_motion_ids, sampled_time_steps)
    self.sampled_bin_idx[env_ids] = sampled_bin_ids

  def _uniform_sampling(self, env_ids: torch.Tensor):
    sampled_frames = torch.randint(
      0, self.motions.total_frames, (len(env_ids),), device=self.device
    )
    sampled_motion_ids = torch.searchsorted(self.motion_frame_ends, sampled_frames, right=True)
    sampled_time_steps = sampled_frames - self.motion_frame_starts[sampled_motion_ids]
    self._set_motion_frames(env_ids, sampled_motion_ids, sampled_time_steps)
    self.sampled_bin_idx[env_ids] = self._current_time_bins(
      sampled_motion_ids, sampled_time_steps
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.valid_bin_count

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if env_ids.numel() == 0:
      return
    if self.cfg.sampling_mode == "adaptive":
      self._update_failure_levels(env_ids)
    self._reset_segment_stats(env_ids)

    if self.cfg.sampling_mode == "start":
      self._set_motion_frames(
        env_ids,
        self._sample_start_motion_indices(len(env_ids)),
        torch.zeros(len(env_ids), dtype=torch.long, device=self.device),
      )
      self.sampled_bin_idx[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)
    self._refresh_motion_buffers(env_ids)
    remaining_steps = torch.clamp(self._current_motion_lengths()[env_ids] - self.time_steps[env_ids], min=1)
    episode_limit = torch.full_like(remaining_steps, self.max_episode_steps)
    self.segment_max_steps[env_ids] = torch.minimum(remaining_steps, episode_limit)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._resample_actor_obs_masks(env_ids)

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self):
    self.time_steps += 1
    self.actor_obs_mask_steps_remaining -= 1
    env_ids = torch.where(self.time_steps >= self._current_motion_lengths())[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)
    actor_mask_env_ids = torch.where(self.actor_obs_mask_steps_remaining <= 0)[0]
    if actor_mask_env_ids.numel() > 0:
      self._resample_actor_obs_masks(actor_mask_env_ids)
    refresh_env_ids = torch.where(
      self.time_steps - self.buffer_start_time >= self.buffer_length
    )[0]
    if refresh_env_ids.numel() > 0:
      self._refresh_motion_buffers(refresh_env_ids)

    self.update_relative_body_poses()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = self.motions.max_length - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        max_idx = int(self._current_motion_lengths()[idx].item()) - 1
        self.time_steps[idx] = min(int(scrubber.value), max_idx)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    frame_tensor = torch.full((len(env_ids),), frame, dtype=torch.long, device=self.device)
    max_frames = self._current_motion_lengths()[env_ids] - 1
    self.time_steps[env_ids] = torch.minimum(frame_tensor, max_frames)
    self.sampled_bin_idx[env_ids] = self._current_time_bins(
      self.motion_index[env_ids], self.time_steps[env_ids]
    )
    remaining_steps = torch.clamp(self._current_motion_lengths()[env_ids] - self.time_steps[env_ids], min=1)
    episode_limit = torch.full_like(remaining_steps, self.max_episode_steps)
    self.segment_max_steps[env_ids] = torch.minimum(remaining_steps, episode_limit)
    self._reset_segment_stats(env_ids)
    self._refresh_motion_buffers(env_ids)
    self._resample_actor_obs_masks(env_ids)
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str | list[str]
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  motion_storage_device: Literal["cpu", "device"] = "cpu"
  motion_buffer_size: int = 64
  actor_obs_mask_enabled: bool = False
  actor_obs_mask_probabilities: dict[str, float] = field(default_factory=dict)
  actor_obs_mask_steps_range: tuple[int, int] = (8, 32)
  actor_obs_append_mask: bool = True
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  adaptive_bin_width_s: float = 4.0
  adaptive_base_temperature: float = 1.0
  adaptive_uniform_floor: float = 0.15
  adaptive_similarity_std: float | None = None
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
