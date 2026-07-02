"""Tests for opt-in large-dataset multi-motion loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.mdp.multi_command_largedataset import (
  ActiveMotionSubset,
  GlobalAdaptiveBinPool,
  LargeDatasetMotionStore,
  LargeDatasetMultiMotionCommand,
  MotionCommand,
  MotionCommandCfg,
)


def _write_motion(path: Path, *, length: int, offset: float) -> None:
  joint_pos = np.arange(length * 2, dtype=np.float32).reshape(length, 2) + offset
  body_pos = np.zeros((length, 2, 3), dtype=np.float32)
  body_pos[..., 0] = offset
  body_quat = np.zeros((length, 2, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0
  np.savez(
    path,
    fps=np.array(30.0, dtype=np.float32),
    joint_pos=joint_pos,
    joint_vel=joint_pos + 0.5,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_pos + 1.0,
    body_ang_vel_w=body_pos + 2.0,
  )


def test_active_subset_tracks_unique_active_and_pending_motion_ids() -> None:
  subset = ActiveMotionSubset(
    total_motion_count=8,
    subset_size=4,
    min_resident_iterations=50,
    device="cpu",
  )

  subset.initialize(torch.tensor([0, 2, 4, 6], dtype=torch.long), iteration=0)
  subset.mark_pending(torch.tensor([1, 3], dtype=torch.long))

  assert subset.active_motion_ids.tolist() == [0, 2, 4, 6]
  assert subset.active_mask.tolist() == [True, False, True, False, True, False, True, False]
  assert subset.pending_mask.tolist() == [False, True, False, True, False, False, False, False]
  assert subset.available_motion_ids().tolist() == [5, 7]


def test_active_subset_refresh_respects_min_residence_and_ref_counts() -> None:
  subset = ActiveMotionSubset(
    total_motion_count=6,
    subset_size=3,
    min_resident_iterations=50,
    device="cpu",
  )
  subset.initialize(torch.tensor([0, 1, 2], dtype=torch.long), iteration=0)

  early = subset.refresh(
    torch.tensor([3, 4], dtype=torch.long),
    iteration=49,
    max_replacements=2,
  )

  assert early.num_replaced == 0
  assert subset.active_motion_ids.tolist() == [0, 1, 2]

  subset.slot_ref_count[:] = torch.tensor([1, 0, 1], dtype=torch.long)
  updated = subset.refresh(
    torch.tensor([3, 4], dtype=torch.long),
    iteration=50,
    max_replacements=2,
  )

  assert updated.num_replaced == 1
  assert subset.active_motion_ids[0].item() == 0
  assert subset.active_motion_ids[2].item() == 2
  assert len(set(subset.active_motion_ids.tolist())) == 3
  assert set(subset.active_motion_ids.tolist()) <= {0, 2, 3, 4}


def test_motion_store_loads_metadata_and_selected_motions_only(tmp_path: Path) -> None:
  files = []
  for index, length in enumerate([3, 5, 4]):
    path = tmp_path / f"motion_{index}.npz"
    _write_motion(path, length=length, offset=float(index * 10))
    files.append(str(path))

  store = LargeDatasetMotionStore(
    files,
    body_indexes=torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert store.num_files == 3
  assert store.file_lengths.tolist() == [3, 5, 4]
  assert not hasattr(store, "joint_pos")

  buffer = store.load_motion_ids(torch.tensor([2, 0], dtype=torch.long))

  assert buffer.global_motion_ids.tolist() == [2, 0]
  assert buffer.file_lengths.tolist() == [4, 3]
  assert buffer.joint_pos.shape == (7, 2)
  torch.testing.assert_close(buffer.joint_pos[0], torch.tensor([20.0, 21.0]))
  torch.testing.assert_close(buffer.body_pos_w[0, 0], torch.tensor([20.0, 0.0, 0.0]))


def test_global_bin_pool_syncs_local_deltas_without_distributed() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 6], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )

  pool.accumulate(
    torch.tensor([0, 1], dtype=torch.long),
    torch.tensor([7, 0], dtype=torch.long),
    torch.tensor([True, False], dtype=torch.bool),
  )
  elapsed = pool.synchronize()

  assert elapsed >= 0.0
  assert pool.pending_episode_delta.sum().item() == pytest.approx(0.0)
  assert pool.pending_failure_delta.sum().item() == pytest.approx(0.0)
  assert pool.bin_episode_count[0, 1].item() == pytest.approx(1.2)
  assert pool.bin_failure_count[0, 1].item() == pytest.approx(2.0)
  assert pool.bin_episode_count[1, 0].item() == pytest.approx(1.2)
  assert pool.bin_failure_count[1, 0].item() == pytest.approx(1.0)


def test_global_bin_pool_probabilities_are_limited_to_active_subset() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 10, 10], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )
  pool.bin_episode_count[:] = 10.0
  pool.bin_failure_count[:] = 1.0
  pool.bin_failure_count[2, 0] = 9.0

  motion_ids, bin_ids, probabilities, failure_rate = (
    pool.compute_active_pair_sampling_probabilities(
      torch.tensor([0, 2], dtype=torch.long),
      adaptive_uniform_ratio=0.0,
      adaptive_failure_rate_max_over_mean=200.0,
      adaptive_sequence_length_agnostic=False,
    )
  )

  assert set(motion_ids.tolist()) == {0, 2}
  assert probabilities.shape == motion_ids.shape == bin_ids.shape
  assert probabilities.sum().item() == pytest.approx(1.0)
  hard_pair = (motion_ids == 2) & (bin_ids == 0)
  easy_pair = (motion_ids == 0) & (bin_ids == 0)
  assert probabilities[hard_pair].item() > probabilities[easy_pair].item()
  assert failure_rate[hard_pair].item() == pytest.approx(0.9)


def test_global_bin_pool_auto_probability_cap_uses_configured_over_mean() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 10, 10], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )

  assert pool._resolve_probability_cap("auto", 4, 12.0) == pytest.approx(3.0)


def _make_large_dataset_command_shell() -> LargeDatasetMultiMotionCommand:
  command = object.__new__(LargeDatasetMultiMotionCommand)
  command._env = SimpleNamespace(device="cpu", num_envs=64)
  command.cfg = SimpleNamespace(
    sampling_mode="adaptive",
    if_log_metrics=False,
    adaptive_uniform_ratio=0.0,
    adaptive_failure_rate_max_over_mean=200.0,
    adaptive_sequence_length_agnostic=False,
    adaptive_max_prob_per_bin=None,
    adaptive_max_prob_per_motion=None,
    adaptive_pre_failure_sample_window_steps=0,
    adaptive_failure_rate_window_iterations=None,
    adaptive_failure_rate_window_chunks=40,
    subset_refresh_count=0,
  )
  command.global_bin_pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 10, 10, 10], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )
  command._bind_global_bin_pool_tensors()
  command.bin_width_steps = command.global_bin_pool.bin_width_steps
  command.active_subset = ActiveMotionSubset(
    total_motion_count=4,
    subset_size=2,
    min_resident_iterations=50,
    device="cpu",
  )
  command.active_subset.initialize(torch.tensor([1, 3], dtype=torch.long), iteration=0)
  command.motion_store = SimpleNamespace(file_lengths=command.global_bin_pool.file_lengths)
  command.motion_idx = torch.zeros(command.num_envs, dtype=torch.long)
  command.motion_length = torch.zeros(command.num_envs, dtype=torch.long)
  command.time_steps = torch.zeros(command.num_envs, dtype=torch.long)
  command.metrics = {}
  return command


def test_large_dataset_command_adaptive_sampling_uses_active_subset_only() -> None:
  command = _make_large_dataset_command_shell()
  command.global_bin_pool.bin_episode_count[:] = 10.0
  command.global_bin_pool.bin_failure_count[:] = 1.0
  command.global_bin_pool.bin_failure_count[3, 0] = 9.0

  env_ids = torch.arange(command.num_envs, dtype=torch.long)
  command._adaptive_sampling(env_ids)

  sampled_motion_ids = set(command.motion_idx.tolist())
  assert sampled_motion_ids <= {1, 3}
  assert 3 in sampled_motion_ids
  assert torch.all(command.motion_length == 10)


def test_large_dataset_command_gathers_by_global_motion_id_through_active_slots() -> None:
  command = _make_large_dataset_command_shell()

  class _FakeSlotBuffer:
    def gather(self, field_name, slot_ids, time_steps):
      assert field_name == "joint_pos"
      return (slot_ids.float() * 10.0 + time_steps.float()).unsqueeze(-1)

  command.motion = _FakeSlotBuffer()

  gathered = command._gather_motion_field(
    "joint_pos",
    torch.tensor([3, 1], dtype=torch.long),
    torch.tensor([1, 2], dtype=torch.long),
  )

  torch.testing.assert_close(gathered.squeeze(-1), torch.tensor([11.0, 2.0]))


def test_large_dataset_command_records_synced_delta_before_advancing_window() -> None:
  command = _make_large_dataset_command_shell()
  command.cfg.adaptive_failure_rate_window_iterations = 2
  command.cfg.adaptive_failure_rate_window_chunks = 2
  command._init_adaptive_sampling_window()
  command.begin_adaptive_sampling_iteration(0)
  chunk_zero_before = command._adaptive_window_episode_chunks[0].clone()
  command.global_bin_pool.pending_episode_delta[1, 0] = 0.2

  command.begin_adaptive_sampling_iteration(1)

  assert command._adaptive_window_current_chunk == 1
  assert command._adaptive_window_episode_chunks[0, 1, 0].item() == pytest.approx(
    chunk_zero_before[1, 0].item() + 0.2
  )
  assert command._adaptive_window_episode_chunks[1, 1, 0].item() == pytest.approx(0.0)


def test_large_dataset_subset_refresh_falls_back_when_adaptive_probabilities_are_sparse(
  monkeypatch,
) -> None:
  command = _make_large_dataset_command_shell()
  command.cfg.subset_adaptive_refresh_ratio = 1.0
  command.active_subset = ActiveMotionSubset(
    total_motion_count=5,
    subset_size=2,
    min_resident_iterations=50,
    device="cpu",
  )
  command.active_subset.initialize(torch.tensor([0, 1], dtype=torch.long), iteration=0)
  command.motion_store = SimpleNamespace(num_files=5)

  def sparse_probabilities(candidate_ids, **kwargs):
    return candidate_ids, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

  monkeypatch.setattr(
    command.global_bin_pool,
    "compute_motion_sampling_probabilities",
    sparse_probabilities,
  )

  sampled = command._sample_subset_replacement_ids(3)

  assert sampled.numel() == 3
  assert len(set(sampled.tolist())) == 3
  assert set(sampled.tolist()) == {2, 3, 4}


def test_large_dataset_command_exports_opt_in_aliases() -> None:
  assert MotionCommand is LargeDatasetMultiMotionCommand
  assert MotionCommandCfg.__name__ == "LargeDatasetMultiMotionCommandCfg"
