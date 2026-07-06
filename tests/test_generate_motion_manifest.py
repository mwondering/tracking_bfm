from __future__ import annotations

from pathlib import Path

from mjlab.scripts.generate_motion_manifest import write_motion_manifest


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
