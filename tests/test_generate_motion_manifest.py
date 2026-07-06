from __future__ import annotations

from pathlib import Path

import numpy as np

from mjlab.scripts.generate_motion_manifest import (
  write_motion_manifest,
  write_motion_metadata_cache,
)


def _write_motion(path: Path, *, length: int, fps: float = 30.0) -> None:
  np.savez(
    path,
    fps=np.asarray(fps, dtype=np.float32),
    joint_pos=np.zeros((length, 2), dtype=np.float32),
  )


def test_write_motion_manifest_collects_sorted_absolute_npz_paths(tmp_path: Path) -> None:
  motion_root = tmp_path / "motions"
  nested = motion_root / "nested"
  nested.mkdir(parents=True)
  first_motion = nested / "b.npz"
  second_motion = motion_root / "a.NPZ"
  ignored = motion_root / "ignore.txt"
  first_motion.write_text("motion", encoding="utf-8")
  second_motion.write_text("motion", encoding="utf-8")
  ignored.write_text("ignore", encoding="utf-8")
  manifest_file = tmp_path / "manifest.txt"

  result = write_motion_manifest(
    motion_root,
    manifest_file,
    backend="python",
  )

  assert result.count == 2
  assert manifest_file.read_text(encoding="utf-8").splitlines() == sorted(
    [
      str(second_motion.resolve()),
      str(first_motion.resolve()),
    ]
  )


def test_write_motion_manifest_can_write_relative_paths(tmp_path: Path) -> None:
  motion_root = tmp_path / "motions"
  nested = motion_root / "nested"
  nested.mkdir(parents=True)
  motion_file = nested / "motion.npz"
  motion_file.write_text("motion", encoding="utf-8")
  manifest_file = tmp_path / "manifest.txt"

  write_motion_manifest(
    motion_root,
    manifest_file,
    backend="python",
    relative_to=motion_root,
  )

  assert manifest_file.read_text(encoding="utf-8").splitlines() == [
    "nested/motion.npz"
  ]


def test_write_motion_manifest_reports_scan_progress(tmp_path: Path) -> None:
  motion_root = tmp_path / "motions"
  motion_root.mkdir()
  for index in range(3):
    (motion_root / f"motion_{index}.npz").write_text("motion", encoding="utf-8")
  progress_counts: list[int] = []

  write_motion_manifest(
    motion_root,
    tmp_path / "manifest.txt",
    backend="python",
    progress_callback=progress_counts.append,
  )

  assert progress_counts == [1, 2, 3]


def test_write_motion_metadata_cache_matches_loader_cache_format(
  tmp_path: Path,
) -> None:
  motion_root = tmp_path / "motions"
  motion_root.mkdir()
  first_motion = motion_root / "first.npz"
  second_motion = motion_root / "second.npz"
  _write_motion(first_motion, length=3, fps=30.0)
  _write_motion(second_motion, length=5, fps=60.0)
  manifest_file = tmp_path / "manifest.txt"
  manifest_file.write_text(
    "\n".join([str(first_motion.resolve()), str(second_motion.resolve())]) + "\n",
    encoding="utf-8",
  )
  metadata_cache_file = tmp_path / "manifest.txt.metadata.npz"
  progress_counts: list[int] = []

  result = write_motion_metadata_cache(
    manifest_file,
    metadata_cache_file,
    workers=1,
    progress_callback=progress_counts.append,
  )

  assert result.count == 2
  assert progress_counts == [1, 2]
  with np.load(metadata_cache_file) as data:
    assert int(data["version"].item()) == 1
    assert int(data["num_files"].item()) == 2
    assert np.asarray(data["file_lengths"], dtype=np.int64).tolist() == [3, 5]
    assert np.asarray(data["fps_values"], dtype=np.float32).tolist() == [
      30.0,
      60.0,
    ]
    assert int(data["non_scalar_fps_count"].item()) == 0
    assert int(data["empty_fps_count"].item()) == 0


def test_reuse_existing_manifest_skips_scan_when_building_metadata(
  tmp_path: Path, monkeypatch
) -> None:
  from mjlab.scripts import generate_motion_manifest

  motion_root = tmp_path / "motions"
  motion_root.mkdir()
  motion_file = motion_root / "motion.npz"
  _write_motion(motion_file, length=3)
  manifest_file = tmp_path / "manifest.txt"
  manifest_file.write_text(str(motion_file.resolve()) + "\n", encoding="utf-8")

  def fail_scan(*args, **kwargs):
    raise AssertionError("existing manifest should be reused without scanning")

  monkeypatch.setattr(generate_motion_manifest, "_collect_motion_files", fail_scan)

  result = generate_motion_manifest.reuse_or_write_motion_manifest(
    motion_root,
    manifest_file,
    backend="python",
    reuse_existing=True,
  )

  assert result.count == 1
