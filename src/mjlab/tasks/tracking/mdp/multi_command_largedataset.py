from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist

from mjlab.managers import CommandTerm
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from .multi_commands import (
  MultiMotionCommand,
  MultiMotionCommandCfg,
  MotionLoader,
  _ISAACLAB_TO_MUJOCO_BODY_REINDEX,
  _ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
)


def _bootstrap_debug(message: str) -> None:
  debug_dir = os.environ.get("MJLAB_BOOTSTRAP_DEBUG_DIR", "")
  if not debug_dir:
    return
  rank = os.environ.get("RANK", "unknown")
  local_rank = os.environ.get("LOCAL_RANK", "unknown")
  pid = os.getpid()
  line = (
    f"[BOOT][{time.strftime('%Y-%m-%d %H:%M:%S')}] "
    f"rank={rank} local_rank={local_rank} pid={pid}: large_motion: {message}"
  )
  try:
    os.makedirs(debug_dir, exist_ok=True)
    log_file = os.path.join(debug_dir, f"rank_{rank}_local_{local_rank}_pid_{pid}.log")
    with open(log_file, "a", encoding="utf-8") as f:
      f.write(line + "\n")
      f.flush()
  except Exception:
    pass


@dataclass(frozen=True)
class SubsetRefreshResult:
  replaced_slot_ids: torch.Tensor
  old_motion_ids: torch.Tensor
  new_motion_ids: torch.Tensor

  @property
  def num_replaced(self) -> int:
    return int(self.new_motion_ids.numel())


class ActiveMotionSubset:
  """Bookkeeping for the unique per-rank active motion subset."""

  def __init__(
    self,
    *,
    total_motion_count: int,
    subset_size: int,
    min_resident_iterations: int,
    device: str | torch.device,
  ) -> None:
    if total_motion_count <= 0:
      raise ValueError("total_motion_count must be positive")
    if subset_size <= 0:
      raise ValueError("subset_size must be positive")
    self.total_motion_count = int(total_motion_count)
    self.subset_size = min(int(subset_size), self.total_motion_count)
    self.min_resident_iterations = max(int(min_resident_iterations), 0)
    self.device = torch.device(device)

    self.active_motion_ids = torch.empty(
      self.subset_size, dtype=torch.long, device=self.device
    )
    self.active_mask = torch.zeros(
      self.total_motion_count, dtype=torch.bool, device=self.device
    )
    self.pending_mask = torch.zeros_like(self.active_mask)
    self.motion_to_slot = torch.full(
      (self.total_motion_count,), -1, dtype=torch.long, device=self.device
    )
    self.slot_loaded_iteration = torch.zeros(
      self.subset_size, dtype=torch.long, device=self.device
    )
    self.slot_ref_count = torch.zeros(
      self.subset_size, dtype=torch.long, device=self.device
    )
    self._initialized = False

  def initialize(self, motion_ids: torch.Tensor, *, iteration: int) -> None:
    motion_ids = self._normalize_motion_ids(motion_ids)
    if motion_ids.numel() != self.subset_size:
      raise ValueError(
        f"Expected {self.subset_size} initial motion ids, got {motion_ids.numel()}"
      )
    if torch.unique(motion_ids).numel() != motion_ids.numel():
      raise ValueError("Initial active subset must contain unique motion ids")

    self.active_motion_ids.copy_(motion_ids)
    self.active_mask.zero_()
    self.active_mask[motion_ids] = True
    self.pending_mask.zero_()
    self.motion_to_slot.fill_(-1)
    self.motion_to_slot[motion_ids] = torch.arange(
      self.subset_size, dtype=torch.long, device=self.device
    )
    self.slot_loaded_iteration.fill_(int(iteration))
    self.slot_ref_count.zero_()
    self._initialized = True

  def mark_pending(self, motion_ids: torch.Tensor) -> None:
    motion_ids = self._normalize_motion_ids(motion_ids)
    self.pending_mask[motion_ids] = True

  def clear_pending(self, motion_ids: torch.Tensor) -> None:
    motion_ids = self._normalize_motion_ids(motion_ids)
    self.pending_mask[motion_ids] = False

  def available_motion_ids(self) -> torch.Tensor:
    unavailable = self.active_mask | self.pending_mask
    return torch.where(~unavailable)[0]

  def set_slot_ref_counts_from_motion_ids(self, motion_ids: torch.Tensor) -> None:
    self.slot_ref_count.zero_()
    if motion_ids.numel() == 0:
      return
    motion_ids = self._normalize_motion_ids(motion_ids)
    slot_ids = self.motion_to_slot[motion_ids]
    slot_ids = slot_ids[slot_ids >= 0]
    if slot_ids.numel() == 0:
      return
    counts = torch.bincount(slot_ids, minlength=self.subset_size)
    self.slot_ref_count.copy_(counts.to(dtype=torch.long, device=self.device))

  def eligible_slot_ids(self, *, iteration: int) -> torch.Tensor:
    if not self._initialized:
      return torch.empty(0, dtype=torch.long, device=self.device)
    resident_iterations = int(iteration) - self.slot_loaded_iteration
    eligible = (
      (resident_iterations >= self.min_resident_iterations)
      & (self.slot_ref_count == 0)
    )
    return torch.where(eligible)[0]

  def refresh(
    self,
    replacement_motion_ids: torch.Tensor,
    *,
    iteration: int,
    max_replacements: int,
    generator: torch.Generator | None = None,
  ) -> SubsetRefreshResult:
    if not self._initialized:
      raise RuntimeError("ActiveMotionSubset.initialize() must be called first")
    if max_replacements <= 0:
      return self._empty_refresh_result()

    replacement_motion_ids = self._filter_replacement_ids(replacement_motion_ids)
    if replacement_motion_ids.numel() == 0:
      return self._empty_refresh_result()

    eligible_slots = self.eligible_slot_ids(iteration=iteration)
    if eligible_slots.numel() == 0:
      return self._empty_refresh_result()

    num_replacements = min(
      int(max_replacements),
      int(replacement_motion_ids.numel()),
      int(eligible_slots.numel()),
    )
    slot_order = torch.randperm(
      eligible_slots.numel(), generator=generator, device=self.device
    )
    selected_slots = eligible_slots[slot_order[:num_replacements]]
    selected_replacements = replacement_motion_ids[:num_replacements]
    old_motion_ids = self.active_motion_ids[selected_slots].clone()

    self.active_mask[old_motion_ids] = False
    self.motion_to_slot[old_motion_ids] = -1

    self.active_motion_ids[selected_slots] = selected_replacements
    self.active_mask[selected_replacements] = True
    self.pending_mask[selected_replacements] = False
    self.motion_to_slot[selected_replacements] = selected_slots
    self.slot_loaded_iteration[selected_slots] = int(iteration)
    self.slot_ref_count[selected_slots] = 0

    return SubsetRefreshResult(
      replaced_slot_ids=selected_slots,
      old_motion_ids=old_motion_ids,
      new_motion_ids=selected_replacements,
    )

  def _filter_replacement_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
    motion_ids = self._normalize_motion_ids(motion_ids)
    if motion_ids.numel() == 0:
      return motion_ids
    unique_ids = torch.unique(motion_ids, sorted=False)
    available = ~(self.active_mask[unique_ids] | self.pending_mask[unique_ids])
    return unique_ids[available]

  def _normalize_motion_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
    if motion_ids.ndim != 1:
      motion_ids = motion_ids.reshape(-1)
    if motion_ids.numel() == 0:
      return motion_ids
    if motion_ids.min() < 0 or motion_ids.max() >= self.total_motion_count:
      raise IndexError("Motion id is outside the full dataset range")
    return motion_ids

  def _empty_refresh_result(self) -> SubsetRefreshResult:
    empty = torch.empty(0, dtype=torch.long, device=self.device)
    return SubsetRefreshResult(empty, empty, empty)


@dataclass
class LargeDatasetMotionBuffer:
  global_motion_ids: torch.Tensor
  file_lengths: torch.Tensor
  length_starts: torch.Tensor
  fps: float
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor

  @property
  def num_files(self) -> int:
    return int(self.global_motion_ids.numel())


class LargeDatasetMotionSlotBuffer:
  """Per-slot GPU cache that can replace a few motions without rebuilding all slots."""

  _FIELD_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )

  def __init__(
    self,
    *,
    global_motion_ids: torch.Tensor,
    chunks: dict[str, list[torch.Tensor]],
    file_lengths: torch.Tensor,
    fps: float,
  ) -> None:
    self.global_motion_ids = global_motion_ids
    self._chunks = chunks
    self.file_lengths = file_lengths
    self.fps = fps
    self._refresh_length_starts()

  @property
  def num_files(self) -> int:
    return int(self.global_motion_ids.numel())

  @property
  def length_starts(self) -> torch.Tensor:
    return self._length_starts

  def gather(
    self,
    field_name: str,
    slot_ids: torch.Tensor,
    time_steps: torch.Tensor,
  ) -> torch.Tensor:
    chunks = self._chunks[field_name]
    flat_slot_ids, flat_time_steps, output_shape = self._flatten_indices(
      slot_ids, time_steps
    )
    tail_shape = chunks[0].shape[1:]
    output = torch.empty(
      (*flat_time_steps.shape, *tail_shape),
      dtype=chunks[0].dtype,
      device=chunks[0].device,
    )
    for slot in torch.unique(flat_slot_ids):
      slot_int = int(slot.item())
      mask = flat_slot_ids == slot
      output[mask] = chunks[slot_int][flat_time_steps[mask]]
    return output.reshape(*output_shape, *tail_shape)

  def replace_slots(
    self,
    slot_ids: torch.Tensor,
    new_motion_ids: torch.Tensor,
    store: "LargeDatasetMotionStore",
  ) -> None:
    if slot_ids.numel() == 0:
      return
    loaded = store.load_motion_chunks(new_motion_ids)
    for offset, slot in enumerate(slot_ids.tolist()):
      self.global_motion_ids[slot] = loaded["global_motion_ids"][offset]
      self.file_lengths[slot] = loaded["file_lengths"][offset]
      for field_name in self._FIELD_NAMES:
        self._chunks[field_name][slot] = loaded[field_name][offset]
    self._refresh_length_starts()

  def _flatten_indices(
    self, slot_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    if time_steps.ndim == 1:
      return slot_ids.reshape(-1), time_steps.reshape(-1), time_steps.shape
    expanded_slots = slot_ids.unsqueeze(-1).expand_as(time_steps)
    return expanded_slots.reshape(-1), time_steps.reshape(-1), time_steps.shape

  def _refresh_length_starts(self) -> None:
    self._length_starts = torch.cat(
      [
        torch.zeros(1, dtype=torch.long, device=self.file_lengths.device),
        self.file_lengths[:-1].cumsum(dim=0),
      ]
    )

  def __getattr__(self, name: str) -> torch.Tensor:
    if name in self._FIELD_NAMES:
      return torch.cat(self._chunks[name], dim=0)
    raise AttributeError(name)


class LargeDatasetMotionStore:
  """CPU/disk-side motion store that only stages requested motions on device."""

  _FIELD_NAMES = LargeDatasetMotionSlotBuffer._FIELD_NAMES

  def __init__(
    self,
    motion_files: list[str],
    body_indexes: torch.Tensor,
    motion_type: Literal["isaaclab", "mujoco"] = "isaaclab",
    device: str | torch.device = "cpu",
  ) -> None:
    if len(motion_files) == 0:
      raise ValueError("motion_files cannot be empty")
    start = time.perf_counter()
    _bootstrap_debug(
      f"LargeDatasetMotionStore init start num_motion_files={len(motion_files)} device={device}"
    )
    self.motion_files = list(motion_files)
    self.num_files = len(self.motion_files)
    self.device = torch.device(device)
    self.motion_type = motion_type
    self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long).cpu()
    self._joint_reindex: list[int] | None = None
    self._body_reindex: list[int] | None = None
    if motion_type == "isaaclab":
      self._joint_reindex = _ISAACLAB_TO_MUJOCO_JOINT_REINDEX
      self._body_reindex = _ISAACLAB_TO_MUJOCO_BODY_REINDEX
    elif motion_type != "mujoco":
      raise ValueError(f"Unsupported motion_type: {motion_type}")

    file_lengths: list[int] = []
    fps_values: list[float] = []
    non_scalar_fps_count = 0
    for index, motion_file in enumerate(self.motion_files):
      if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Invalid motion file path: {motion_file}")
      with np.load(motion_file) as data:
        file_lengths.append(int(data["joint_pos"].shape[0]))
        fps_value, is_scalar_fps = self._extract_fps_value(data["fps"], motion_file)
        fps_values.append(fps_value)
        if not is_scalar_fps:
          non_scalar_fps_count += 1
      if (index + 1) % 5000 == 0:
        _bootstrap_debug(
          f"LargeDatasetMotionStore metadata progress {index + 1}/{len(self.motion_files)}"
        )
    self.file_lengths = torch.tensor(
      file_lengths, dtype=torch.long, device=self.device
    )
    self.fps_list = fps_values
    self.fps = fps_values[0]
    _bootstrap_debug(
      "LargeDatasetMotionStore init done "
      f"num_files={self.num_files} total_frames={int(sum(file_lengths))} "
      f"non_scalar_fps_count={non_scalar_fps_count} "
      f"elapsed={time.perf_counter() - start:.3f}s"
    )

  @staticmethod
  def _extract_fps_value(fps_data: np.ndarray, motion_file: str) -> tuple[float, bool]:
    fps_array = np.asarray(fps_data, dtype=np.float32)
    if fps_array.size == 0:
      raise ValueError(f"Motion file has an empty fps array: {motion_file}")
    return float(fps_array.reshape(-1)[0]), fps_array.size == 1

  def load_motion_ids(self, motion_ids: torch.Tensor) -> LargeDatasetMotionBuffer:
    loaded = self.load_motion_chunks(motion_ids)
    length_starts = torch.cat(
      [
        torch.zeros(1, dtype=torch.long, device=self.device),
        loaded["file_lengths"][:-1].cumsum(dim=0),
      ]
    )
    return LargeDatasetMotionBuffer(
      global_motion_ids=loaded["global_motion_ids"],
      file_lengths=loaded["file_lengths"],
      length_starts=length_starts,
      fps=self.fps,
      joint_pos=torch.cat(loaded["joint_pos"], dim=0),
      joint_vel=torch.cat(loaded["joint_vel"], dim=0),
      body_pos_w=torch.cat(loaded["body_pos_w"], dim=0),
      body_quat_w=torch.cat(loaded["body_quat_w"], dim=0),
      body_lin_vel_w=torch.cat(loaded["body_lin_vel_w"], dim=0),
      body_ang_vel_w=torch.cat(loaded["body_ang_vel_w"], dim=0),
    )

  def load_slot_buffer(self, motion_ids: torch.Tensor) -> LargeDatasetMotionSlotBuffer:
    loaded = self.load_motion_chunks(motion_ids)
    chunks = {field_name: loaded[field_name] for field_name in self._FIELD_NAMES}
    return LargeDatasetMotionSlotBuffer(
      global_motion_ids=loaded["global_motion_ids"],
      chunks=chunks,
      file_lengths=loaded["file_lengths"],
      fps=self.fps,
    )

  def load_motion_chunks(self, motion_ids: torch.Tensor) -> dict[str, object]:
    start = time.perf_counter()
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
    if motion_ids.ndim != 1:
      motion_ids = motion_ids.reshape(-1)
    if motion_ids.numel() == 0:
      raise ValueError("motion_ids cannot be empty")
    if motion_ids.min() < 0 or motion_ids.max() >= self.num_files:
      raise IndexError("Motion id is outside the full dataset range")
    should_log = motion_ids.numel() >= 100
    if should_log:
      _bootstrap_debug(
        f"load_motion_chunks start count={int(motion_ids.numel())} "
        f"first_ids={motion_ids[:5].detach().cpu().tolist()}"
      )

    loaded: dict[str, object] = {
      "global_motion_ids": motion_ids.clone(),
      "file_lengths": self.file_lengths[motion_ids].clone(),
    }
    for field_name in self._FIELD_NAMES:
      loaded[field_name] = []

    motion_id_list = motion_ids.detach().cpu().tolist()
    for offset, motion_id in enumerate(motion_id_list):
      fields = self._load_one_motion(motion_id)
      for field_name in self._FIELD_NAMES:
        loaded[field_name].append(fields[field_name])
      if should_log and (offset + 1) % 1000 == 0:
        _bootstrap_debug(
          f"load_motion_chunks progress {offset + 1}/{len(motion_id_list)} "
          f"elapsed={time.perf_counter() - start:.3f}s"
        )
    if should_log:
      total_frames = int(self.file_lengths[motion_ids].sum().item())
      allocated = (
        torch.cuda.memory_allocated(self.device)
        if self.device.type == "cuda"
        else 0
      )
      reserved = (
        torch.cuda.memory_reserved(self.device)
        if self.device.type == "cuda"
        else 0
      )
      _bootstrap_debug(
        f"load_motion_chunks done count={int(motion_ids.numel())} "
        f"total_frames={total_frames} elapsed={time.perf_counter() - start:.3f}s "
        f"cuda_allocated={allocated} cuda_reserved={reserved}"
      )
    return loaded

  def _load_one_motion(self, motion_id: int) -> dict[str, torch.Tensor]:
    with np.load(self.motion_files[motion_id]) as data:
      joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
      joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
      body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
      body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
      body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
      body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

    if self._joint_reindex is not None:
      joint_pos = joint_pos[:, self._joint_reindex]
      joint_vel = joint_vel[:, self._joint_reindex]
    if self._body_reindex is not None:
      body_pos_w = body_pos_w[:, self._body_reindex, :]
      body_quat_w = body_quat_w[:, self._body_reindex, :]
      body_lin_vel_w = body_lin_vel_w[:, self._body_reindex, :]
      body_ang_vel_w = body_ang_vel_w[:, self._body_reindex, :]

    body_indices = self._body_indexes.numpy()
    body_pos_w = body_pos_w[:, body_indices, :]
    body_quat_w = body_quat_w[:, body_indices, :]
    body_lin_vel_w = body_lin_vel_w[:, body_indices, :]
    body_ang_vel_w = body_ang_vel_w[:, body_indices, :]

    return {
      "joint_pos": torch.as_tensor(joint_pos, dtype=torch.float32, device=self.device),
      "joint_vel": torch.as_tensor(joint_vel, dtype=torch.float32, device=self.device),
      "body_pos_w": torch.as_tensor(body_pos_w, dtype=torch.float32, device=self.device),
      "body_quat_w": torch.as_tensor(
        body_quat_w, dtype=torch.float32, device=self.device
      ),
      "body_lin_vel_w": torch.as_tensor(
        body_lin_vel_w, dtype=torch.float32, device=self.device
      ),
      "body_ang_vel_w": torch.as_tensor(
        body_ang_vel_w, dtype=torch.float32, device=self.device
      ),
    }


class GlobalAdaptiveBinPool:
  """Global full-dataset adaptive statistics with deferred distributed sync."""

  def __init__(
    self,
    file_lengths: torch.Tensor,
    *,
    bin_width_steps: int,
    init_num_failures: float,
    device: str | torch.device,
  ) -> None:
    self.device = torch.device(device)
    self.file_lengths = torch.as_tensor(
      file_lengths, dtype=torch.long, device=self.device
    )
    self.num_files = int(self.file_lengths.numel())
    self.bin_width_steps = max(int(bin_width_steps), 1)
    self.bin_count = int(self.file_lengths.max().item() // self.bin_width_steps) + 1

    self.motion_bin_counts = torch.clamp(
      torch.div(
        self.file_lengths + self.bin_width_steps - 1,
        self.bin_width_steps,
        rounding_mode="floor",
      ),
      min=1,
    )
    bin_indices = torch.arange(self.bin_count, device=self.device)
    self.bin_valid_mask = bin_indices.unsqueeze(0) < self.motion_bin_counts.unsqueeze(1)
    self.valid_motion_ids, self.valid_bin_ids = torch.where(self.bin_valid_mask)
    self.num_valid_motion_bins = max(int(self.valid_motion_ids.numel()), 1)
    bin_starts = bin_indices.unsqueeze(0) * self.bin_width_steps
    remaining_lengths = (self.file_lengths.unsqueeze(1) - bin_starts).clamp(min=0)
    self.bin_lengths = torch.minimum(
      remaining_lengths,
      torch.full_like(remaining_lengths, self.bin_width_steps),
    )
    self.bin_lengths.masked_fill_(~self.bin_valid_mask, 0)
    valid_bin_lengths = self.bin_lengths[self.bin_valid_mask].float()
    mean_bin_length = torch.clamp(valid_bin_lengths.mean(), min=1.0)
    self.base_bin_weights = self.bin_lengths.float() / mean_bin_length
    self.base_bin_weights.masked_fill_(~self.bin_valid_mask, 0.0)

    init_count = float(init_num_failures)
    self.bin_episode_count = torch.full(
      (self.num_files, self.bin_count),
      init_count,
      dtype=torch.float,
      device=self.device,
    )
    self.bin_failure_count = torch.full_like(self.bin_episode_count, init_count)
    self.bin_episode_count.masked_fill_(~self.bin_valid_mask, 0.0)
    self.bin_failure_count.masked_fill_(~self.bin_valid_mask, 0.0)
    self.pending_episode_delta = torch.zeros_like(self.bin_episode_count)
    self.pending_failure_delta = torch.zeros_like(self.bin_failure_count)
    self.last_episode_delta = torch.zeros_like(self.bin_episode_count)
    self.last_failure_delta = torch.zeros_like(self.bin_failure_count)

  def compute_motion_bin_indices(
    self, time_steps: torch.Tensor, motion_ids: torch.Tensor
  ) -> torch.Tensor:
    raw_bin_indices = torch.div(time_steps, self.bin_width_steps, rounding_mode="floor")
    max_bin_indices = self.motion_bin_counts[motion_ids] - 1
    return torch.minimum(raw_bin_indices, max_bin_indices)

  def compute_failure_rate(self) -> torch.Tensor:
    failure_rate = self.bin_failure_count / torch.clamp(
      self.bin_episode_count, min=1e-12
    )
    return failure_rate.masked_fill(~self.bin_valid_mask, 0.0)

  def accumulate(
    self,
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
    failure_mask: torch.Tensor | None,
  ) -> None:
    if motion_ids.numel() == 0:
      return
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
    time_steps = torch.as_tensor(time_steps, dtype=torch.long, device=self.device)
    current_bin_indices = self.compute_motion_bin_indices(time_steps, motion_ids)
    linear_indices = motion_ids * self.bin_count + current_bin_indices
    current_counts = torch.bincount(
      linear_indices, minlength=self.num_files * self.bin_count
    ).view(self.num_files, self.bin_count)
    episode_increments = current_counts.float() / torch.clamp(
      self.bin_lengths.float(), min=1.0
    )
    self.pending_episode_delta += episode_increments

    if failure_mask is None or not failure_mask.any():
      return
    failure_mask = torch.as_tensor(failure_mask, dtype=torch.bool, device=self.device)
    failed_linear_indices = linear_indices[failure_mask]
    failed_counts = torch.bincount(
      failed_linear_indices, minlength=self.num_files * self.bin_count
    ).view(self.num_files, self.bin_count)
    self.pending_failure_delta += failed_counts.float()

  def synchronize(self) -> float:
    start = time.perf_counter()
    episode_delta = self.pending_episode_delta.clone()
    failure_delta = self.pending_failure_delta.clone()
    if dist.is_available() and dist.is_initialized():
      dist.all_reduce(episode_delta, op=dist.ReduceOp.SUM)
      dist.all_reduce(failure_delta, op=dist.ReduceOp.SUM)
    self.bin_episode_count += episode_delta
    self.bin_failure_count += failure_delta
    self.last_episode_delta.copy_(episode_delta)
    self.last_failure_delta.copy_(failure_delta)
    self.pending_episode_delta.zero_()
    self.pending_failure_delta.zero_()
    return time.perf_counter() - start

  def compute_active_pair_sampling_probabilities(
    self,
    active_motion_ids: torch.Tensor,
    *,
    adaptive_uniform_ratio: float,
    adaptive_failure_rate_max_over_mean: float,
    adaptive_sequence_length_agnostic: bool,
    adaptive_max_prob_per_bin: float | Literal["auto"] | None = "auto",
    adaptive_max_prob_per_motion: float | Literal["auto"] | None = "auto",
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_motion_ids = torch.as_tensor(
      active_motion_ids, dtype=torch.long, device=self.device
    )
    active_bin_mask = self.bin_valid_mask[active_motion_ids]
    active_row_ids, valid_bin_ids = torch.where(active_bin_mask)
    valid_motion_ids = active_motion_ids[active_row_ids]
    if valid_motion_ids.numel() == 0:
      raise RuntimeError("Active subset contains no valid motion bins")

    probabilities, valid_failure_rate = self._compute_pair_probabilities(
      valid_motion_ids,
      valid_bin_ids,
      num_motions=int(active_motion_ids.numel()),
      adaptive_uniform_ratio=adaptive_uniform_ratio,
      adaptive_failure_rate_max_over_mean=adaptive_failure_rate_max_over_mean,
      adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
      adaptive_max_prob_per_bin=adaptive_max_prob_per_bin,
      adaptive_max_prob_per_motion=adaptive_max_prob_per_motion,
      auto_cap_over_mean=adaptive_failure_rate_max_over_mean,
    )
    return valid_motion_ids, valid_bin_ids, probabilities, valid_failure_rate

  def compute_motion_sampling_probabilities(
    self,
    candidate_motion_ids: torch.Tensor,
    *,
    adaptive_uniform_ratio: float,
    adaptive_failure_rate_max_over_mean: float,
    adaptive_sequence_length_agnostic: bool,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    motion_ids, _, pair_probabilities, _ = self.compute_active_pair_sampling_probabilities(
      candidate_motion_ids,
      adaptive_uniform_ratio=adaptive_uniform_ratio,
      adaptive_failure_rate_max_over_mean=adaptive_failure_rate_max_over_mean,
      adaptive_sequence_length_agnostic=adaptive_sequence_length_agnostic,
      adaptive_max_prob_per_bin=None,
      adaptive_max_prob_per_motion=None,
    )
    motion_probabilities = torch.zeros(
      self.num_files, dtype=pair_probabilities.dtype, device=self.device
    )
    motion_probabilities.scatter_add_(0, motion_ids, pair_probabilities)
    candidate_probabilities = motion_probabilities[candidate_motion_ids]
    candidate_probabilities = candidate_probabilities / torch.clamp(
      candidate_probabilities.sum(), min=1e-12
    )
    return candidate_motion_ids, candidate_probabilities

  def _compute_pair_probabilities(
    self,
    valid_motion_ids: torch.Tensor,
    valid_bin_ids: torch.Tensor,
    *,
    num_motions: int,
    adaptive_uniform_ratio: float,
    adaptive_failure_rate_max_over_mean: float,
    adaptive_sequence_length_agnostic: bool,
    adaptive_max_prob_per_bin: float | Literal["auto"] | None,
    adaptive_max_prob_per_motion: float | Literal["auto"] | None,
    auto_cap_over_mean: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    failure_rate = self.compute_failure_rate()
    valid_failure_rate = failure_rate[valid_motion_ids, valid_bin_ids]
    failure_rate_mean = valid_failure_rate.mean()
    failure_rate_upper_bound = failure_rate_mean * float(
      adaptive_failure_rate_max_over_mean
    )
    clipped_failure_rate = torch.clamp(
      valid_failure_rate, 0.0, failure_rate_upper_bound
    )
    clipped_sum = clipped_failure_rate.sum()
    if clipped_sum <= 0.0:
      failure_based_probabilities = torch.full(
        (len(valid_motion_ids),),
        1.0 / float(max(len(valid_motion_ids), 1)),
        dtype=torch.float,
        device=self.device,
      )
    else:
      failure_based_probabilities = clipped_failure_rate / clipped_sum

    uniform_probabilities = torch.full_like(
      failure_based_probabilities, 1.0 / float(max(len(valid_motion_ids), 1))
    )
    uniform_ratio = float(max(0.0, min(1.0, adaptive_uniform_ratio)))
    probabilities = (
      1.0 - uniform_ratio
    ) * failure_based_probabilities + uniform_ratio * uniform_probabilities
    bin_weights = self._compute_bin_weights(adaptive_sequence_length_agnostic)
    probabilities = probabilities * bin_weights[valid_motion_ids, valid_bin_ids]
    probabilities = probabilities / torch.clamp(probabilities.sum(), min=1e-12)
    probabilities = self._apply_max_probability_constraints(
      probabilities,
      valid_motion_ids,
      num_motions,
      adaptive_max_prob_per_bin,
      adaptive_max_prob_per_motion,
      auto_cap_over_mean,
    )
    return probabilities, valid_failure_rate

  def _compute_bin_weights(self, sequence_length_agnostic: bool) -> torch.Tensor:
    bin_weights = self.base_bin_weights
    if sequence_length_agnostic:
      bin_weights = bin_weights / self.motion_bin_counts.unsqueeze(1).float()
      bin_weights = bin_weights.masked_fill(~self.bin_valid_mask, 0.0)
    return bin_weights

  def _apply_max_probability_constraints(
    self,
    probabilities: torch.Tensor,
    valid_motion_ids: torch.Tensor,
    num_motions: int,
    max_prob_per_bin: float | Literal["auto"] | None,
    max_prob_per_motion: float | Literal["auto"] | None,
    auto_cap_over_mean: float,
  ) -> torch.Tensor:
    constrained = probabilities
    resolved_bin_cap = self._resolve_probability_cap(
      max_prob_per_bin, len(probabilities), auto_cap_over_mean
    )
    if resolved_bin_cap is not None and len(probabilities) > 1.0 / resolved_bin_cap:
      constrained = torch.clamp(constrained, max=resolved_bin_cap)
      constrained = constrained / torch.clamp(constrained.sum(), min=1e-12)

    resolved_motion_cap = self._resolve_probability_cap(
      max_prob_per_motion, num_motions, auto_cap_over_mean
    )
    if resolved_motion_cap is not None and num_motions > 1.0 / resolved_motion_cap:
      motion_probabilities = torch.zeros(
        self.num_files, dtype=constrained.dtype, device=self.device
      )
      motion_probabilities.scatter_add_(0, valid_motion_ids, constrained)
      motion_scale = torch.ones_like(motion_probabilities)
      oversized = motion_probabilities > resolved_motion_cap
      motion_scale[oversized] = resolved_motion_cap / torch.clamp(
        motion_probabilities[oversized], min=1e-12
      )
      constrained = constrained * motion_scale[valid_motion_ids]
      constrained = constrained / torch.clamp(constrained.sum(), min=1e-12)
    return constrained

  def _resolve_probability_cap(
    self, value: float | Literal["auto"] | None, count: int, auto_cap_over_mean: float
  ) -> float | None:
    if value is None:
      return None
    if value == "auto":
      if count <= 0:
        return 1.0
      return float(auto_cap_over_mean) / float(count)
    resolved = float(value)
    if resolved <= 0.0:
      return None
    return resolved


class LargeDatasetMultiMotionCommand(MultiMotionCommand):
  cfg: "LargeDatasetMultiMotionCommandCfg"

  def __init__(self, cfg: "LargeDatasetMultiMotionCommandCfg", env):
    _bootstrap_debug("LargeDatasetMultiMotionCommand init start")
    CommandTerm.__init__(self, cfg, env)

    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    _bootstrap_debug("before resolve motion files")
    motion_files = self._resolve_all_motion_files()
    _bootstrap_debug(f"after resolve motion files count={len(motion_files)}")
    store_start = time.perf_counter()
    self.motion_store = LargeDatasetMotionStore(
      motion_files,
      self.body_indexes,
      motion_type=self.cfg.motion_type,
      device=self.device,
    )
    _bootstrap_debug(
      f"after LargeDatasetMotionStore elapsed={time.perf_counter() - store_start:.3f}s"
    )
    subset_size = min(self.cfg.active_subset_size, self.motion_store.num_files)
    _bootstrap_debug(
      f"initial active subset sampling start subset_size={subset_size} "
      f"total_motion_count={self.motion_store.num_files}"
    )
    initial_motion_ids = self._sample_unique_motion_ids(
      torch.arange(self.motion_store.num_files, dtype=torch.long, device=self.device),
      subset_size,
      probabilities=None,
    )
    _bootstrap_debug(
      f"initial active subset sampled first_ids={initial_motion_ids[:5].detach().cpu().tolist()}"
    )
    self.active_subset = ActiveMotionSubset(
      total_motion_count=self.motion_store.num_files,
      subset_size=subset_size,
      min_resident_iterations=self.cfg.subset_min_resident_iterations,
      device=self.device,
    )
    self.active_subset.initialize(initial_motion_ids, iteration=0)
    load_start = time.perf_counter()
    _bootstrap_debug("before initial active subset load_slot_buffer")
    self.motion = self.motion_store.load_slot_buffer(self.active_subset.active_motion_ids)
    _bootstrap_debug(
      f"after initial active subset load_slot_buffer elapsed={time.perf_counter() - load_start:.3f}s"
    )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_length = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    if self.cfg.adaptive_bin_width_steps is not None:
      self.bin_width_steps = max(int(self.cfg.adaptive_bin_width_steps), 1)
    else:
      self.bin_width_steps = max(
        int(round(float(self.cfg.adaptive_bin_width_s) / env.step_dt)), 1
      )
    bin_pool_start = time.perf_counter()
    _bootstrap_debug("before GlobalAdaptiveBinPool")
    self.global_bin_pool = GlobalAdaptiveBinPool(
      self.motion_store.file_lengths,
      bin_width_steps=self.bin_width_steps,
      init_num_failures=self.cfg.adaptive_init_num_failures,
      device=self.device,
    )
    _bootstrap_debug(
      f"after GlobalAdaptiveBinPool elapsed={time.perf_counter() - bin_pool_start:.3f}s "
      f"bin_count={self.global_bin_pool.bin_count}"
    )
    self._bind_global_bin_pool_tensors()
    self._init_adaptive_sampling_window()
    self._adaptive_sampling_phase = "idle"
    self._skip_current_adaptive_episode_count = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    if self.cfg.if_log_metrics:
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
      self.metrics["sampling_uniform_prob"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_top1_prob"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_top1_ratio"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_failure_rate_mean"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_failure_rate_max"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_effective_num_bins"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_num_concentrated_bins"] = torch.zeros(
        self.num_envs, device=self.device
      )

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._extra_reference_ghost_model = None
    self._extra_reference_ghost_color = np.array((1.0, 0.45, 0.1, 0.45), dtype=np.float32)
    self.extra_reference_motion = (
      MotionLoader(
        self.cfg.extra_reference_motion_file,
        self.body_indexes,
        motion_type=self.cfg.motion_type,
        device=self.device,
      )
      if self.cfg.extra_reference_motion_file
      else None
    )
    self._last_global_bin_update_time = 0.0
    self._last_subset_update_time = 0.0
    _bootstrap_debug("LargeDatasetMultiMotionCommand init done")

  def _resolve_all_motion_files(self) -> list[str]:
    motion_path = os.fspath(self.cfg.motion_path)
    motion_file = os.fspath(self.cfg.motion_file)
    if motion_path and motion_file:
      raise ValueError(
        "Provide either motion_path for multi-motion input or motion_file for a "
        "single motion, but not both."
      )

    if motion_path:
      self._validate_motion_path(motion_path)
      manifest_file = self._resolve_motion_manifest_file(motion_path)
      if manifest_file:
        resolved_motion_files = self._resolve_motion_files_with_manifest(
          motion_path, manifest_file
        )
      else:
        resolved_motion_files = self._scan_motion_path(motion_path)
    elif motion_file:
      if not os.path.exists(motion_file):
        raise FileNotFoundError(f"Invalid motion file: {motion_file}")
      if not os.path.isfile(motion_file):
        raise ValueError(f"motion_file must point to a .npz file: {motion_file}")
      resolved_motion_files = [motion_file]
    else:
      resolved_motion_files = []

    if len(resolved_motion_files) == 0:
      raise ValueError(
        "No motion files found. Provide either:\n"
        "  - motion_path: path to a directory containing .npz files\n"
        "  - motion_file: path to a single motion file"
      )
    return resolved_motion_files

  def _validate_motion_path(self, motion_path: str) -> None:
    if not os.path.exists(motion_path):
      raise FileNotFoundError(f"Invalid motion path: {motion_path}")
    if not os.path.isdir(motion_path):
      raise ValueError(
        f"motion_path must point to a directory containing .npz files: {motion_path}"
      )

  def _resolve_motion_manifest_file(self, motion_path: str) -> str:
    configured_manifest = os.fspath(getattr(self.cfg, "motion_manifest_file", ""))
    if configured_manifest:
      return configured_manifest

    _, world_size = self._runtime_rank_context()
    debug_dir = os.environ.get("MJLAB_BOOTSTRAP_DEBUG_DIR", "")
    if world_size <= 1 or not debug_dir:
      return ""

    motion_path_key = hashlib.sha1(
      os.path.abspath(motion_path).encode("utf-8")
    ).hexdigest()[:12]
    return os.path.join(debug_dir, f"motion_manifest_{motion_path_key}.txt")

  def _resolve_motion_files_with_manifest(
    self, motion_path: str, manifest_file: str
  ) -> list[str]:
    rank, world_size = self._runtime_rank_context()
    _bootstrap_debug(
      "resolve motion files with manifest "
      f"path={motion_path} manifest={manifest_file} rank={rank} world_size={world_size}"
    )

    if os.path.exists(manifest_file):
      motion_files = self._read_motion_manifest(manifest_file)
      _bootstrap_debug(
        f"read existing motion manifest count={len(motion_files)} file={manifest_file}"
      )
      return motion_files

    if world_size <= 1 or rank == 0:
      motion_files = self._scan_motion_path(motion_path)
      self._write_motion_manifest(manifest_file, motion_files)
      _bootstrap_debug(
        f"wrote motion manifest count={len(motion_files)} file={manifest_file}"
      )
      return motion_files

    return self._wait_for_motion_manifest(manifest_file)

  def _runtime_rank_context(self) -> tuple[int, int]:
    try:
      rank = int(os.environ.get("RANK", "0"))
    except ValueError:
      rank = 0
    try:
      world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
      world_size = 1
    return rank, max(world_size, 1)

  def _scan_motion_path(self, motion_path: str) -> list[str]:
    start = time.perf_counter()
    last_log_time = start
    log_interval = float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0))
    resolved_motion_files: list[str] = []
    scanned_dirs = 0
    scanned_files = 0
    _bootstrap_debug(f"scan motion path start path={motion_path}")
    for root, _, files in os.walk(motion_path):
      scanned_dirs += 1
      scanned_files += len(files)
      for filename in files:
        if filename.lower().endswith(".npz"):
          resolved_motion_files.append(os.path.join(root, filename))

      now = time.perf_counter()
      if log_interval > 0.0 and now - last_log_time >= log_interval:
        _bootstrap_debug(
          "scan motion path progress "
          f"dirs={scanned_dirs} files={scanned_files} "
          f"motions={len(resolved_motion_files)} elapsed={now - start:.3f}s "
          f"root={root}"
        )
        last_log_time = now

    resolved_motion_files.sort()
    _bootstrap_debug(
      "scan motion path done "
      f"dirs={scanned_dirs} files={scanned_files} motions={len(resolved_motion_files)} "
      f"elapsed={time.perf_counter() - start:.3f}s"
    )
    return resolved_motion_files

  def _write_motion_manifest(self, manifest_file: str, motion_files: list[str]) -> None:
    manifest_dir = os.path.dirname(manifest_file)
    if manifest_dir:
      os.makedirs(manifest_dir, exist_ok=True)
    tmp_file = f"{manifest_file}.tmp.{os.getpid()}"
    with open(tmp_file, "w", encoding="utf-8") as f:
      for motion_file in motion_files:
        f.write(motion_file + "\n")
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp_file, manifest_file)

  def _read_motion_manifest(self, manifest_file: str) -> list[str]:
    with open(manifest_file, encoding="utf-8") as f:
      return [line.strip() for line in f if line.strip()]

  def _wait_for_motion_manifest(self, manifest_file: str) -> list[str]:
    timeout_s = float(getattr(self.cfg, "motion_manifest_wait_timeout_s", 600.0))
    poll_interval_s = max(
      float(getattr(self.cfg, "motion_manifest_poll_interval_s", 0.25)), 0.01
    )
    log_interval_s = max(
      float(getattr(self.cfg, "motion_scan_log_interval_s", 10.0)), 1.0
    )
    start = time.perf_counter()
    last_log_time = start
    _bootstrap_debug(
      f"waiting for motion manifest file={manifest_file} timeout={timeout_s:.1f}s"
    )
    while True:
      if os.path.exists(manifest_file):
        motion_files = self._read_motion_manifest(manifest_file)
        _bootstrap_debug(
          f"read motion manifest count={len(motion_files)} file={manifest_file}"
        )
        return motion_files

      now = time.perf_counter()
      elapsed = now - start
      if elapsed >= timeout_s:
        raise TimeoutError(
          f"Timed out after {timeout_s:.1f}s waiting for motion manifest: "
          f"{manifest_file}"
        )
      if now - last_log_time >= log_interval_s:
        _bootstrap_debug(
          f"still waiting for motion manifest elapsed={elapsed:.3f}s "
          f"file={manifest_file}"
        )
        last_log_time = now
      time.sleep(poll_interval_s)

  def _bind_global_bin_pool_tensors(self) -> None:
    self.bin_count = self.global_bin_pool.bin_count
    self.motion_bin_counts = self.global_bin_pool.motion_bin_counts
    self.bin_valid_mask = self.global_bin_pool.bin_valid_mask
    self.valid_motion_ids = self.global_bin_pool.valid_motion_ids
    self.valid_bin_ids = self.global_bin_pool.valid_bin_ids
    self.num_valid_motion_bins = self.global_bin_pool.num_valid_motion_bins
    self.bin_lengths = self.global_bin_pool.bin_lengths
    self.bin_weights = self.global_bin_pool._compute_bin_weights(
      self.cfg.adaptive_sequence_length_agnostic
    )
    self.bin_episode_count = self.global_bin_pool.bin_episode_count
    self.bin_failure_count = self.global_bin_pool.bin_failure_count

  def begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    if self.cfg.sampling_mode == "adaptive":
      self._last_global_bin_update_time = self.global_bin_pool.synchronize()
      if (
        self.global_bin_pool.last_episode_delta.any()
        or self.global_bin_pool.last_failure_delta.any()
      ):
        self._record_adaptive_sampling_window_increments(
          self.global_bin_pool.last_episode_delta,
          self.global_bin_pool.last_failure_delta,
        )
      super().begin_adaptive_sampling_iteration(iteration)
    else:
      self._last_global_bin_update_time = 0.0
    start = time.perf_counter()
    self._refresh_active_subset(iteration)
    self._last_subset_update_time = time.perf_counter() - start

  def get_large_dataset_timing_stats(self) -> dict[str, float]:
    return {
      "global_bin_update_time": float(self._last_global_bin_update_time),
      "subset_update_time": float(self._last_subset_update_time),
    }

  def _compute_failure_rate(self) -> torch.Tensor:
    return self.global_bin_pool.compute_failure_rate()

  def _compute_motion_bin_indices(
    self, time_steps: torch.Tensor, motion_indices: torch.Tensor
  ) -> torch.Tensor:
    return self.global_bin_pool.compute_motion_bin_indices(time_steps, motion_indices)

  def _accumulate_adaptive_sampling_stats(
    self,
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
    failure_mask: torch.Tensor | None,
  ) -> None:
    self.global_bin_pool.accumulate(motion_ids, time_steps, failure_mask)

  def _clamp_motion_time_steps(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    max_time_steps = self.motion_store.file_lengths[motion_ids] - 1
    if time_steps.ndim > 1:
      max_time_steps = max_time_steps.unsqueeze(-1)
    clamped_time_steps = torch.clamp_min(time_steps, 0)
    return torch.minimum(clamped_time_steps, max_time_steps)

  def _gather_motion_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    slot_ids = self.active_subset.motion_to_slot[motion_ids]
    if torch.any(slot_ids < 0):
      missing_motion_ids = motion_ids[slot_ids < 0]
      raise RuntimeError(
        "Requested motion ids are not resident in the active subset: "
        f"{missing_motion_ids.detach().cpu().tolist()}"
      )
    clamped_time_steps = self._clamp_motion_time_steps(motion_ids, time_steps)
    return self.motion.gather(field_name, slot_ids, clamped_time_steps)

  def _uniform_baseline_probabilities(
    self, motion_indices: torch.Tensor
  ) -> torch.Tensor:
    active_valid_bins = int(
      self.global_bin_pool.bin_valid_mask[self.active_subset.active_motion_ids]
      .sum()
      .item()
    )
    return torch.full(
      (len(motion_indices),),
      1.0 / float(max(active_valid_bins, 1)),
      dtype=torch.float,
      device=self.device,
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    valid_motion_ids, valid_bin_ids, sampling_probabilities, valid_failure_rate = (
      self.global_bin_pool.compute_active_pair_sampling_probabilities(
        self.active_subset.active_motion_ids,
        adaptive_uniform_ratio=self.cfg.adaptive_uniform_ratio,
        adaptive_failure_rate_max_over_mean=self.cfg.adaptive_failure_rate_max_over_mean,
        adaptive_sequence_length_agnostic=self.cfg.adaptive_sequence_length_agnostic,
        adaptive_max_prob_per_bin=self.cfg.adaptive_max_prob_per_bin,
        adaptive_max_prob_per_motion=self.cfg.adaptive_max_prob_per_motion,
      )
    )
    sampled_pair_indices = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    sampled_motion_indices = valid_motion_ids[sampled_pair_indices]
    sampled_bin_indices = valid_bin_ids[sampled_pair_indices]

    active_valid_bin_count = int(valid_motion_ids.numel())
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    denom = math.log(active_valid_bin_count) if active_valid_bin_count > 1 else 1.0
    H_norm = H / denom if active_valid_bin_count > 1 else 0.0
    pmax, _ = sampling_probabilities.max(dim=0)
    uniform_prob = 1.0 / float(max(active_valid_bin_count, 1))
    effective_num_bins = 1.0 / torch.clamp(
      (sampling_probabilities**2).sum(), min=1e-12
    )
    num_concentrated_bins = (sampling_probabilities > 10.0 * uniform_prob).sum().float()
    if self.cfg.if_log_metrics:
      self.metrics["sampling_entropy"][env_ids] = H_norm
      self.metrics["sampling_uniform_prob"][env_ids] = uniform_prob
      self.metrics["sampling_top1_prob"][env_ids] = pmax
      self.metrics["sampling_top1_ratio"][env_ids] = pmax / uniform_prob
      self.metrics["sampling_failure_rate_mean"][env_ids] = valid_failure_rate.mean()
      self.metrics["sampling_failure_rate_max"][env_ids] = valid_failure_rate.max()
      self.metrics["sampling_effective_num_bins"][env_ids] = effective_num_bins
      self.metrics["sampling_num_concentrated_bins"][env_ids] = num_concentrated_bins

    self.motion_idx[env_ids] = sampled_motion_indices
    self.motion_length[env_ids] = self.motion_store.file_lengths[sampled_motion_indices]

    bin_starts = sampled_bin_indices * self.bin_width_steps
    bin_ends = torch.minimum(
      bin_starts + self.bin_width_steps, self.motion_length[env_ids]
    )
    bin_lengths = torch.clamp(bin_ends - bin_starts, min=1)
    offsets = (
      sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      * bin_lengths.float()
    ).long()
    self.time_steps[env_ids] = torch.minimum(
      bin_starts + offsets, self.motion_length[env_ids] - 1
    )
    if self.cfg.adaptive_pre_failure_sample_window_steps > 0:
      pre_failure_offsets = torch.randint(
        self.cfg.adaptive_pre_failure_sample_window_steps,
        (len(env_ids),),
        device=self.device,
      )
      self.time_steps[env_ids] = (
        self.time_steps[env_ids] - pre_failure_offsets
      ).clamp_min(0)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = (
      sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      * self.motion_length[env_ids]
    ).long()
    if self.cfg.if_log_metrics:
      uniform_probabilities = self._uniform_baseline_probabilities(
        self.motion_idx[env_ids]
      )
      self.metrics["sampling_entropy"][env_ids] = 1.0
      self.metrics["sampling_uniform_prob"][env_ids] = uniform_probabilities[
        : len(env_ids)
      ]
      self.metrics["sampling_top1_prob"][env_ids] = uniform_probabilities[
        : len(env_ids)
      ]
      self.metrics["sampling_top1_ratio"][env_ids] = 1.0
      self.metrics["sampling_failure_rate_mean"][env_ids] = 0.0
      self.metrics["sampling_failure_rate_max"][env_ids] = 0.0
      self.metrics["sampling_effective_num_bins"][env_ids] = float(
        self.global_bin_pool.bin_valid_mask[self.active_subset.active_motion_ids]
        .sum()
        .item()
      )
      self.metrics["sampling_num_concentrated_bins"][env_ids] = 0.0

  def _resample_command(self, env_ids: torch.Tensor):
    if len(env_ids) == 0:
      return
    self._stage_pre_resample_adaptive_stats(env_ids)

    if self.cfg.sampling_mode == "start":
      motion_indices = self._sample_active_motion_ids(len(env_ids))
      self.motion_idx[env_ids] = motion_indices
      self.motion_length[env_ids] = self.motion_store.file_lengths[motion_indices]
      self.time_steps[env_ids] = 0
      print(
        " ************** [FOR DEBUG] WARNING: All envs time steps is set to start initialization ! ************** "
      )
    elif self.cfg.sampling_mode == "uniform":
      motion_indices = self._sample_active_motion_ids(len(env_ids))
      self.motion_idx[env_ids] = motion_indices
      self.motion_length[env_ids] = self.motion_store.file_lengths[motion_indices]
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()
    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids],
      soft_joint_pos_limits[:, :, 0],
      soft_joint_pos_limits[:, :, 1],
    )

    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(
      torch.cat(
        [
          root_pos[env_ids],
          root_ori[env_ids],
          root_lin_vel[env_ids],
          root_ang_vel[env_ids],
        ],
        dim=-1,
      ),
      env_ids=env_ids,
    )
    self.robot.clear_state(env_ids=env_ids)

  def _refresh_active_subset(self, iteration: int) -> None:
    refresh_count = int(self.cfg.subset_refresh_count)
    if refresh_count <= 0 or self.motion_store.num_files <= self.active_subset.subset_size:
      return
    self.active_subset.set_slot_ref_counts_from_motion_ids(self.motion_idx)
    replacement_ids = self._sample_subset_replacement_ids(refresh_count)
    if replacement_ids.numel() == 0:
      return
    refresh_result = self.active_subset.refresh(
      replacement_ids,
      iteration=iteration,
      max_replacements=refresh_count,
    )
    self.motion.replace_slots(
      refresh_result.replaced_slot_ids,
      refresh_result.new_motion_ids,
      self.motion_store,
    )

  def _sample_subset_replacement_ids(self, count: int) -> torch.Tensor:
    available_ids = self.active_subset.available_motion_ids()
    if available_ids.numel() == 0:
      return available_ids
    count = min(int(count), int(available_ids.numel()))
    adaptive_count = int(round(count * float(self.cfg.subset_adaptive_refresh_ratio)))
    adaptive_count = max(0, min(count, adaptive_count))
    sampled_parts: list[torch.Tensor] = []
    if adaptive_count > 0 and self.cfg.sampling_mode == "adaptive":
      candidate_ids, candidate_probabilities = (
        self.global_bin_pool.compute_motion_sampling_probabilities(
          available_ids,
          adaptive_uniform_ratio=self.cfg.adaptive_uniform_ratio,
          adaptive_failure_rate_max_over_mean=self.cfg.adaptive_failure_rate_max_over_mean,
          adaptive_sequence_length_agnostic=self.cfg.adaptive_sequence_length_agnostic,
        )
      )
      positive_probability = candidate_probabilities > 0.0
      positive_candidate_ids = candidate_ids[positive_probability]
      positive_probabilities = candidate_probabilities[positive_probability]
      adaptive_count = min(adaptive_count, int(positive_candidate_ids.numel()))
      if adaptive_count > 0:
        positive_probabilities = positive_probabilities / torch.clamp(
          positive_probabilities.sum(), min=1e-12
        )
        sampled_indices = torch.multinomial(
          positive_probabilities, adaptive_count, replacement=False
        )
        sampled_parts.append(positive_candidate_ids[sampled_indices])

    remaining_count = count - sum(part.numel() for part in sampled_parts)
    if remaining_count > 0:
      excluded = torch.zeros(
        self.motion_store.num_files, dtype=torch.bool, device=self.device
      )
      for part in sampled_parts:
        excluded[part] = True
      random_pool = available_ids[~excluded[available_ids]]
      if random_pool.numel() > 0:
        random_order = torch.randperm(random_pool.numel(), device=self.device)
        sampled_parts.append(random_pool[random_order[:remaining_count]])

    if not sampled_parts:
      return torch.empty(0, dtype=torch.long, device=self.device)
    return torch.cat(sampled_parts)[:count]

  def _sample_active_motion_ids(self, count: int) -> torch.Tensor:
    active_ids = self.active_subset.active_motion_ids
    random_indices = torch.randint(active_ids.numel(), (count,), device=self.device)
    return active_ids[random_indices]

  def _sample_unique_motion_ids(
    self,
    candidate_ids: torch.Tensor,
    count: int,
    probabilities: torch.Tensor | None,
  ) -> torch.Tensor:
    if probabilities is None:
      order = torch.randperm(candidate_ids.numel(), device=self.device)
      return candidate_ids[order[:count]]
    sampled_indices = torch.multinomial(probabilities, count, replacement=False)
    return candidate_ids[sampled_indices]


@dataclass(kw_only=True)
class LargeDatasetMultiMotionCommandCfg(MultiMotionCommandCfg):
  """Opt-in large-dataset motion command configuration."""

  active_subset_size: int = 20_000
  subset_refresh_count: int = 10
  subset_min_resident_iterations: int = 50
  subset_adaptive_refresh_ratio: float = 0.5
  motion_manifest_file: str = ""
  motion_manifest_wait_timeout_s: float = 600.0
  motion_manifest_poll_interval_s: float = 0.25
  motion_scan_log_interval_s: float = 10.0

  def build(self, env) -> LargeDatasetMultiMotionCommand:
    return LargeDatasetMultiMotionCommand(self, env)


MotionCommand = LargeDatasetMultiMotionCommand
MotionCommandCfg = LargeDatasetMultiMotionCommandCfg
