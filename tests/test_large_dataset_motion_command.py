"""Tests for opt-in large-dataset multi-motion loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mjlab.tasks.tracking.mdp.multi_command_largedataset as large_dataset_module
from mjlab.tasks.tracking.mdp.multi_command_largedataset import (
  ActiveMotionSubset,
  GlobalAdaptiveBinPool,
  LargeDatasetMotionSlotBuffer,
  LargeDatasetMotionStore,
  LargeDatasetMultiMotionCommand,
  MotionCommand,
  MotionCommandCfg,
)


def _write_motion(
  path: Path,
  *,
  length: int,
  offset: float,
  fps: float | np.ndarray = 30.0,
) -> None:
  joint_pos = np.arange(length * 2, dtype=np.float32).reshape(length, 2) + offset
  body_pos = np.zeros((length, 2, 3), dtype=np.float32)
  body_pos[..., 0] = offset
  body_quat = np.zeros((length, 2, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0
  np.savez(
    path,
    fps=np.asarray(fps, dtype=np.float32),
    joint_pos=joint_pos,
    joint_vel=joint_pos + 0.5,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_pos + 1.0,
    body_ang_vel_w=body_pos + 2.0,
  )


def _make_motion_resolver_shell(
  *,
  motion_path: Path,
  manifest_file: Path,
) -> LargeDatasetMultiMotionCommand:
  command = object.__new__(LargeDatasetMultiMotionCommand)
  command.cfg = SimpleNamespace(
    motion_path=str(motion_path),
    motion_file="",
    motion_manifest_file=str(manifest_file),
    motion_manifest_wait_timeout_s=0.2,
    motion_manifest_poll_interval_s=0.01,
    motion_scan_log_interval_s=60.0,
  )
  return command


def test_large_dataset_rank_zero_writes_motion_manifest(
  tmp_path: Path, monkeypatch
) -> None:
  motion_root = tmp_path / "motions"
  nested = motion_root / "nested"
  nested.mkdir(parents=True)
  _write_motion(motion_root / "b.npz", length=3, offset=0.0)
  _write_motion(nested / "a.npz", length=3, offset=1.0)
  (motion_root / "ignore.txt").write_text("not a motion", encoding="utf-8")
  manifest_file = tmp_path / "manifest.txt"
  command = _make_motion_resolver_shell(
    motion_path=motion_root,
    manifest_file=manifest_file,
  )
  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "0")

  resolved = command._resolve_all_motion_files()

  expected = sorted([str(nested / "a.npz"), str(motion_root / "b.npz")])
  assert resolved == expected
  assert manifest_file.read_text(encoding="utf-8").splitlines() == expected


def test_large_dataset_nonzero_rank_reads_manifest_without_scanning(
  tmp_path: Path, monkeypatch
) -> None:
  motion_root = tmp_path / "motions"
  motion_root.mkdir()
  manifest_file = tmp_path / "manifest.txt"
  motion_file = motion_root / "motion.npz"
  _write_motion(motion_file, length=3, offset=0.0)
  manifest_file.write_text(str(motion_file) + "\n", encoding="utf-8")
  command = _make_motion_resolver_shell(
    motion_path=motion_root,
    manifest_file=manifest_file,
  )
  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")

  def fail_if_scanned(*args, **kwargs):
    raise AssertionError("nonzero ranks should read the manifest instead of scanning")

  monkeypatch.setattr(large_dataset_module.os, "walk", fail_if_scanned)

  assert command._resolve_all_motion_files() == [str(motion_file)]


def test_large_dataset_metadata_cache_path_defaults_to_manifest_sidecar(
  tmp_path: Path,
) -> None:
  manifest_file = tmp_path / "manifest.txt"
  command = _make_motion_resolver_shell(
    motion_path=tmp_path,
    manifest_file=manifest_file,
  )
  command.cfg.motion_metadata_cache_file = ""

  assert command._resolve_motion_metadata_cache_file() == (
    str(manifest_file) + ".metadata.npz"
  )

  command.cfg.motion_metadata_cache_file = str(tmp_path / "explicit_metadata.npz")

  assert command._resolve_motion_metadata_cache_file() == str(
    tmp_path / "explicit_metadata.npz"
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


def test_slot_buffer_gather_uses_bucketed_slots() -> None:
  chunks = {}
  for field_name in LargeDatasetMotionSlotBuffer._FIELD_NAMES:
    chunks[field_name] = [
      torch.tensor([[0.0], [1.0]], dtype=torch.float32),
      torch.tensor(
        [[10.0], [11.0], [12.0], [13.0], [14.0]], dtype=torch.float32
      ),
      torch.tensor([[20.0], [21.0], [22.0]], dtype=torch.float32),
    ]
  buffer = LargeDatasetMotionSlotBuffer(
    global_motion_ids=torch.tensor([0, 1, 2], dtype=torch.long),
    chunks=chunks,
    file_lengths=torch.tensor([2, 5, 3], dtype=torch.long),
    fps=30.0,
  )

  gathered = buffer.gather(
    "joint_pos",
    torch.tensor([2, 0, 1, 2], dtype=torch.long),
    torch.tensor([0, 1, 4, 2], dtype=torch.long),
  )

  torch.testing.assert_close(
    gathered.squeeze(-1), torch.tensor([20.0, 1.0, 14.0, 22.0])
  )
  torch.testing.assert_close(
    buffer.joint_pos.squeeze(-1),
    torch.tensor([0.0, 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0]),
  )
  assert sorted(buffer._bucket_capacities) == [2, 4, 8]


def test_slot_buffer_replace_slots_does_not_rebuild_flat_cache(monkeypatch) -> None:
  chunks = {}
  for field_name in LargeDatasetMotionSlotBuffer._FIELD_NAMES:
    chunks[field_name] = [
      torch.tensor([[0.0], [1.0]], dtype=torch.float32),
      torch.tensor([[10.0], [11.0], [12.0]], dtype=torch.float32),
    ]
  buffer = LargeDatasetMotionSlotBuffer(
    global_motion_ids=torch.tensor([0, 1], dtype=torch.long),
    chunks=chunks,
    file_lengths=torch.tensor([2, 3], dtype=torch.long),
    fps=30.0,
  )

  class _FakeStore:
    def load_motion_chunks(self, motion_ids):
      assert motion_ids.tolist() == [9]
      loaded = {
        "global_motion_ids": torch.tensor([9], dtype=torch.long),
        "file_lengths": torch.tensor([5], dtype=torch.long),
      }
      for field_name in LargeDatasetMotionSlotBuffer._FIELD_NAMES:
        loaded[field_name] = [torch.arange(5, dtype=torch.float32).unsqueeze(-1) + 90.0]
      return loaded

  def fail_cat(*args, **kwargs):
    raise AssertionError("replace_slots should not rebuild a full flat cache")

  monkeypatch.setattr(large_dataset_module.torch, "cat", fail_cat)

  buffer.replace_slots(
    torch.tensor([0], dtype=torch.long),
    torch.tensor([9], dtype=torch.long),
    _FakeStore(),
  )

  assert buffer.global_motion_ids.tolist() == [9, 1]
  assert buffer.file_lengths.tolist() == [5, 3]
  gathered = buffer.gather(
    "joint_pos",
    torch.tensor([0, 1], dtype=torch.long),
    torch.tensor([4, 2], dtype=torch.long),
  )
  torch.testing.assert_close(gathered.squeeze(-1), torch.tensor([94.0, 12.0]))


def test_motion_store_accepts_non_scalar_fps_arrays(tmp_path: Path) -> None:
  path = tmp_path / "motion_with_array_fps.npz"
  _write_motion(
    path,
    length=3,
    offset=0.0,
    fps=np.array([30.0, 30.0], dtype=np.float32),
  )

  store = LargeDatasetMotionStore(
    [str(path)],
    body_indexes=torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert store.fps == pytest.approx(30.0)
  assert store.fps_list == [pytest.approx(30.0)]
  assert store.non_scalar_fps_count == 1
  assert store.empty_fps_count == 0


def test_motion_store_uses_default_fps_for_empty_fps_arrays(tmp_path: Path) -> None:
  path = tmp_path / "motion_with_empty_fps.npz"
  _write_motion(
    path,
    length=3,
    offset=0.0,
    fps=np.array([], dtype=np.float32),
  )

  store = LargeDatasetMotionStore(
    [str(path)],
    body_indexes=torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert store.fps == pytest.approx(30.0)
  assert store.fps_list == [pytest.approx(30.0)]
  assert store.non_scalar_fps_count == 0
  assert store.empty_fps_count == 1


def test_motion_store_reuses_metadata_cache_without_opening_motion_files(
  tmp_path: Path, monkeypatch
) -> None:
  motion_files = []
  for index, length in enumerate([3, 5]):
    path = tmp_path / f"motion_{index}.npz"
    _write_motion(path, length=length, offset=float(index))
    motion_files.append(str(path))
  metadata_cache_file = tmp_path / "metadata_cache.npz"

  first_store = LargeDatasetMotionStore(
    motion_files,
    body_indexes=torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
    metadata_cache_file=str(metadata_cache_file),
  )
  assert first_store.file_lengths.tolist() == [3, 5]
  assert metadata_cache_file.exists()

  original_np_load = large_dataset_module.np.load

  def fail_motion_np_load(path, *args, **kwargs):
    if Path(path) == metadata_cache_file:
      return original_np_load(path, *args, **kwargs)
    raise AssertionError("metadata cache hit should not open motion npz files")

  monkeypatch.setattr(large_dataset_module.np, "load", fail_motion_np_load)

  second_store = LargeDatasetMotionStore(
    motion_files,
    body_indexes=torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
    metadata_cache_file=str(metadata_cache_file),
  )

  assert second_store.file_lengths.tolist() == [3, 5]
  assert second_store.fps_list == [pytest.approx(30.0), pytest.approx(30.0)]


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


def test_global_bin_pool_accumulate_updates_only_touched_bins(monkeypatch) -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 6], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )

  def fail_bincount(*args, **kwargs):
    raise AssertionError("accumulate should use sparse touched-bin updates")

  monkeypatch.setattr(large_dataset_module.torch, "bincount", fail_bincount)

  pool.accumulate(
    torch.tensor([0, 0, 0, 1], dtype=torch.long),
    torch.tensor([0, 3, 7, 5], dtype=torch.long),
    torch.tensor([True, False, True, True], dtype=torch.bool),
  )

  torch.testing.assert_close(
    pool.pending_episode_delta,
    torch.tensor(
      [
        [0.4, 0.2, 0.0],
        [0.0, 1.0, 0.0],
      ],
      dtype=torch.float32,
    ),
  )
  torch.testing.assert_close(
    pool.pending_failure_delta,
    torch.tensor(
      [
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
      ],
      dtype=torch.float32,
    ),
  )


def test_sharded_global_bin_pool_keeps_owner_shard_and_active_cache() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 10, 10, 10], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
    rank=1,
    world_size=2,
  )

  assert pool.owned_motion_ids.tolist() == [1, 3]
  assert pool.bin_episode_count.shape == (2, 3)

  pool.set_active_motion_ids(torch.tensor([0, 3], dtype=torch.long))

  assert pool.active_motion_ids.tolist() == [0, 3]
  assert pool.active_episode_count.shape == (2, 3)
  assert pool.active_motion_to_slot[0].item() == 0
  assert pool.active_motion_to_slot[3].item() == 1


def test_sharded_global_bin_pool_updates_shard_and_active_cache() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 10, 10, 10], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
    rank=1,
    world_size=2,
  )
  pool.set_active_motion_ids(torch.tensor([1, 2], dtype=torch.long))

  pool.accumulate(
    torch.tensor([1, 2], dtype=torch.long),
    torch.tensor([7, 0], dtype=torch.long),
    torch.tensor([True, True], dtype=torch.bool),
  )
  pool.synchronize()

  # Motion 1 is owned by rank 1, so the sharded full pool is updated.
  torch.testing.assert_close(pool.bin_episode_count[0], torch.tensor([1.0, 1.2, 0.0]))
  torch.testing.assert_close(pool.bin_failure_count[0], torch.tensor([1.0, 2.0, 0.0]))
  # Motion 2 is not owned by rank 1, but it is in the local active subset, so
  # the active cache still receives the gathered sparse update.
  torch.testing.assert_close(
    pool.active_episode_count[1], torch.tensor([1.2, 1.0, 0.0])
  )
  torch.testing.assert_close(
    pool.active_failure_count[1], torch.tensor([2.0, 1.0, 0.0])
  )


def test_sharded_global_bin_pool_reports_sparse_update_timing() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([10, 10, 10, 10], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
    rank=1,
    world_size=2,
  )

  pool.accumulate(
    torch.tensor([1, 2], dtype=torch.long),
    torch.tensor([7, 0], dtype=torch.long),
    torch.tensor([True, False], dtype=torch.bool),
  )
  elapsed = pool.synchronize()
  stats = pool.get_timing_stats()

  assert stats["global_bin_update_time"] == pytest.approx(elapsed)
  assert stats["global_bin_update_pack_time"] >= 0.0
  assert stats["global_bin_update_gather_time"] >= 0.0
  assert stats["global_bin_update_apply_time"] >= 0.0
  assert stats["global_bin_update_episode_key_count"] == pytest.approx(2.0)
  assert stats["global_bin_update_failure_key_count"] == pytest.approx(1.0)


def test_global_bin_pool_resets_counts_on_configured_interval() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([20, 20], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
    rank=0,
    world_size=2,
  )
  pool.set_active_motion_ids(torch.tensor([0, 1], dtype=torch.long))
  pool.bin_episode_count[0] = torch.tensor([10.0, 12.0, 14.0, 16.0, 0.0])
  pool.bin_failure_count[0] = torch.tensor([0.0, 10.0, 2.0, 8.0, 0.0])
  pool.active_episode_count[0] = pool.bin_episode_count[0]
  pool.active_failure_count[0] = pool.bin_failure_count[0]

  skipped_time = pool.reset_counts_if_due(
    iteration=4999,
    interval_iterations=5000,
  )
  assert skipped_time == pytest.approx(0.0)
  torch.testing.assert_close(
    pool.bin_episode_count[0],
    torch.tensor([10.0, 12.0, 14.0, 16.0, 0.0]),
  )

  reset_time = pool.reset_counts_if_due(
    iteration=5000,
    interval_iterations=5000,
  )

  assert reset_time >= 0.0
  assert pool.get_timing_stats()["adaptive_bin_pool_reset_applied"] == pytest.approx(1.0)
  torch.testing.assert_close(
    pool.bin_episode_count[0],
    torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
  )
  torch.testing.assert_close(
    pool.bin_failure_count[0],
    torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
  )
  torch.testing.assert_close(pool.active_failure_count[0], pool.bin_failure_count[0])


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
    subset_adaptive_candidate_pool_size=10_000,
    adaptive_bin_pool_reset_interval_iterations=5000,
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
  command._last_global_bin_update_time = 0.0
  command._last_subset_update_time = 0.0
  command._motion_gather_time_accum = 0.0
  command._motion_gather_call_count = 0
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
  assert command._motion_gather_call_count == 1
  assert command._motion_gather_time_accum >= 0.0


def test_large_dataset_timing_stats_report_and_reset_gather_accumulators() -> None:
  command = _make_large_dataset_command_shell()
  command._last_global_bin_update_time = 0.25
  command._last_subset_update_time = 0.5
  command._motion_gather_time_accum = 0.75
  command._motion_gather_call_count = 3
  command.global_bin_pool._last_timing_stats.update(
    {
      "global_bin_update_time": 0.25,
      "global_bin_update_pack_time": 0.01,
      "global_bin_update_gather_time": 0.02,
      "global_bin_update_apply_time": 0.03,
      "adaptive_bin_pool_reset_time": 0.04,
      "adaptive_bin_pool_reset_applied": 1.0,
      "global_bin_update_episode_key_count": 5.0,
      "global_bin_update_failure_key_count": 2.0,
    }
  )

  stats = command.get_large_dataset_timing_stats(reset=True)

  assert stats["global_bin_update_time"] == pytest.approx(0.25)
  assert stats["global_bin_update_pack_time"] == pytest.approx(0.01)
  assert stats["global_bin_update_gather_time"] == pytest.approx(0.02)
  assert stats["global_bin_update_apply_time"] == pytest.approx(0.03)
  assert stats["adaptive_bin_pool_reset_time"] == pytest.approx(0.04)
  assert stats["adaptive_bin_pool_reset_applied"] == pytest.approx(1.0)
  assert stats["global_bin_update_episode_key_count"] == pytest.approx(5.0)
  assert stats["global_bin_update_failure_key_count"] == pytest.approx(2.0)
  assert stats["subset_update_time"] == pytest.approx(0.5)
  assert stats["motion_gather_time"] == pytest.approx(0.75)
  assert stats["motion_gather_call_count"] == pytest.approx(3.0)
  assert command._motion_gather_time_accum == pytest.approx(0.0)
  assert command._motion_gather_call_count == 0


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


def test_large_dataset_command_reset_clears_adaptive_window_chunks() -> None:
  command = _make_large_dataset_command_shell()
  command.cfg.adaptive_bin_pool_reset_interval_iterations = 1
  command.cfg.adaptive_failure_rate_window_iterations = 4
  command.cfg.adaptive_failure_rate_window_chunks = 2
  command._init_adaptive_sampling_window()
  command.begin_adaptive_sampling_iteration(0)
  command.global_bin_pool.pending_episode_delta[1, 0] = 0.2
  command.global_bin_pool.pending_failure_delta[1, 0] = 1.0

  command.begin_adaptive_sampling_iteration(1)

  torch.testing.assert_close(
    command.global_bin_pool.bin_episode_count,
    torch.tensor(
      [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
      ]
    ),
  )
  torch.testing.assert_close(
    command.global_bin_pool.bin_failure_count,
    torch.tensor(
      [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
      ]
    ),
  )
  assert command._adaptive_window_episode_chunks.sum().item() == pytest.approx(
    command.global_bin_pool.bin_episode_count.sum().item()
  )
  assert command._adaptive_window_failure_chunks.sum().item() == pytest.approx(
    command.global_bin_pool.bin_failure_count.sum().item()
  )
  assert (
    command.get_large_dataset_timing_stats()["adaptive_bin_pool_reset_applied"]
    == pytest.approx(1.0)
  )


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


def test_large_dataset_subset_refresh_limits_adaptive_candidate_pool(monkeypatch) -> None:
  command = _make_large_dataset_command_shell()
  command.cfg.subset_adaptive_refresh_ratio = 1.0
  command.cfg.subset_adaptive_candidate_pool_size = 3
  command.active_subset = ActiveMotionSubset(
    total_motion_count=10,
    subset_size=2,
    min_resident_iterations=50,
    device="cpu",
  )
  command.active_subset.initialize(torch.tensor([0, 1], dtype=torch.long), iteration=0)
  command.motion_store = SimpleNamespace(num_files=10)

  def candidate_probabilities(candidate_ids, **kwargs):
    assert candidate_ids.numel() == 3
    return candidate_ids, torch.full((3,), 1.0 / 3.0)

  monkeypatch.setattr(
    command.global_bin_pool,
    "compute_motion_sampling_probabilities",
    candidate_probabilities,
  )

  sampled = command._sample_subset_replacement_ids(2)

  assert sampled.numel() == 2


def test_large_dataset_command_exports_opt_in_aliases() -> None:
  assert MotionCommand is LargeDatasetMultiMotionCommand
  assert MotionCommandCfg.__name__ == "LargeDatasetMultiMotionCommandCfg"
