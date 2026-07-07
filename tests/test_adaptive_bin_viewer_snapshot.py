from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.multi_command_largedataset import GlobalAdaptiveBinPool
from mjlab.tasks.tracking.viewer.snapshot import (
  AdaptiveBinPoolSnapshotWriter,
  build_compact_bin_pool_snapshot,
)


def test_compact_snapshot_covers_full_dataset_and_subtracts_prior() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([3, 8, 11, 6], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )
  pool.bin_episode_count[:] = torch.tensor(
    [
      [3.0, 0.0, 0.0],
      [6.0, 4.0, 0.0],
      [2.0, 5.0, 8.0],
      [1.0, 9.0, 0.0],
    ]
  )
  pool.bin_failure_count[:] = torch.tensor(
    [
      [2.0, 0.0, 0.0],
      [1.5, 3.0, 0.0],
      [1.0, 2.0, 4.0],
      [1.0, 6.0, 0.0],
    ]
  )

  snapshot = build_compact_bin_pool_snapshot(pool, num_buckets=2)

  np.testing.assert_allclose(
    snapshot.access_sum,
    np.array([[7.0, 3.0, 0.0], [1.0, 12.0, 7.0]], dtype=np.float32),
  )
  np.testing.assert_allclose(
    snapshot.failure_sum,
    np.array([[1.5, 2.0, 0.0], [0.0, 6.0, 3.0]], dtype=np.float32),
  )
  np.testing.assert_array_equal(
    snapshot.valid_count,
    np.array([[2, 1, 0], [2, 2, 1]], dtype=np.int32),
  )
  np.testing.assert_array_equal(snapshot.bucket_start_motion_ids, [0, 2])
  np.testing.assert_array_equal(snapshot.bucket_end_motion_ids, [2, 4])


def test_compact_snapshot_buckets_by_global_motion_id_for_shards() -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([3, 8, 11, 6], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
    rank=1,
    world_size=2,
  )
  assert pool.owned_motion_ids.tolist() == [1, 3]
  pool.bin_episode_count[:] = torch.tensor(
    [
      [6.0, 4.0, 0.0],
      [1.0, 9.0, 0.0],
    ]
  )
  pool.bin_failure_count[:] = torch.tensor(
    [
      [1.5, 3.0, 0.0],
      [1.0, 6.0, 0.0],
    ]
  )

  snapshot = build_compact_bin_pool_snapshot(pool, num_buckets=2)

  np.testing.assert_allclose(
    snapshot.access_sum,
    np.array([[5.0, 3.0, 0.0], [0.0, 8.0, 0.0]], dtype=np.float32),
  )
  np.testing.assert_allclose(
    snapshot.failure_sum,
    np.array([[0.5, 2.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32),
  )
  np.testing.assert_array_equal(
    snapshot.valid_count,
    np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int32),
  )


def test_snapshot_writer_writes_atomic_binary_files_and_metadata(tmp_path: Path) -> None:
  pool = GlobalAdaptiveBinPool(
    torch.tensor([3, 8, 11, 6], dtype=torch.long),
    bin_width_steps=5,
    init_num_failures=1.0,
    device="cpu",
  )
  pool.bin_episode_count += 2.0
  pool.bin_failure_count += 1.0
  motion_files = [f"/data/motion_{idx:04d}.npz" for idx in range(4)]
  writer = AdaptiveBinPoolSnapshotWriter(
    snapshot_dir=tmp_path,
    num_buckets=2,
    motion_files=motion_files,
    manifest_file="/data/manifest.txt",
  )

  writer.write(pool, iteration=12)

  metadata = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
  assert metadata["iteration"] == 12
  assert metadata["num_files"] == 4
  assert metadata["bucket_count"] == 2
  assert metadata["bin_count"] == 3
  assert metadata["bin_width_steps"] == 5
  assert metadata["manifest_file"] == "/data/manifest.txt"
  assert metadata["bucket_first_paths"] == [
    "/data/motion_0000.npz",
    "/data/motion_0002.npz",
  ]
  assert metadata["bucket_last_paths"] == [
    "/data/motion_0001.npz",
    "/data/motion_0003.npz",
  ]

  access = np.fromfile(tmp_path / "access_sum.f32", dtype=np.float32).reshape(2, 3)
  failure = np.fromfile(tmp_path / "failure_sum.f32", dtype=np.float32).reshape(2, 3)
  valid = np.fromfile(tmp_path / "valid_count.i32", dtype=np.int32).reshape(2, 3)
  assert access.shape == (2, 3)
  assert failure.shape == (2, 3)
  assert valid.shape == (2, 3)
  assert np.all(access[valid > 0] >= 0.0)
