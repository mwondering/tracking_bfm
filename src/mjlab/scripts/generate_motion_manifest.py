"""Generate a stable manifest for large motion datasets."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NamedTuple

from tqdm import tqdm


Backend = Literal["auto", "find", "python"]
ProgressCallback = Callable[[int], None]


class ProgressReporter(NamedTuple):
  callback: ProgressCallback | None
  close: Callable[[], None]


@dataclass(frozen=True)
class ManifestResult:
  count: int
  elapsed_s: float
  manifest_file: Path


def _notify_progress(
  progress_callback: ProgressCallback | None,
  count: int,
) -> None:
  if progress_callback is not None:
    progress_callback(count)


def _collect_with_find(
  motion_root: Path,
  *,
  progress_callback: ProgressCallback | None = None,
) -> list[Path]:
  find_bin = shutil.which("find")
  if find_bin is None:
    raise RuntimeError("The `find` executable is not available")
  process = subprocess.Popen(
    [
      find_bin,
      str(motion_root),
      "-type",
      "f",
      "(",
      "-iname",
      "*.npz",
      ")",
      "-print0",
    ],
    stdout=subprocess.PIPE,
  )
  assert process.stdout is not None
  motion_files: list[Path] = []
  pending = b""
  count = 0
  while True:
    chunk = process.stdout.read(1024 * 1024)
    if not chunk:
      break
    pending += chunk
    parts = pending.split(b"\0")
    pending = parts.pop()
    for raw_path in parts:
      if not raw_path:
        continue
      motion_files.append(Path(raw_path.decode()))
      count += 1
      _notify_progress(progress_callback, count)
  if pending:
    motion_files.append(Path(pending.decode()))
    count += 1
    _notify_progress(progress_callback, count)
  return_code = process.wait()
  if return_code != 0:
    raise subprocess.CalledProcessError(return_code, process.args)
  return motion_files


def _collect_with_python(
  motion_root: Path,
  *,
  progress_callback: ProgressCallback | None = None,
) -> list[Path]:
  motion_files: list[Path] = []
  stack = [motion_root]
  count = 0
  while stack:
    root = stack.pop()
    with os.scandir(root) as entries:
      for entry in entries:
        if entry.is_dir(follow_symlinks=False):
          stack.append(Path(entry.path))
        elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(
          ".npz"
        ):
          motion_files.append(Path(entry.path))
          count += 1
          _notify_progress(progress_callback, count)
  return motion_files


def _collect_motion_files(
  motion_root: Path,
  backend: Backend,
  *,
  progress_callback: ProgressCallback | None = None,
) -> list[Path]:
  if backend == "find":
    return _collect_with_find(motion_root, progress_callback=progress_callback)
  if backend == "python":
    return _collect_with_python(motion_root, progress_callback=progress_callback)
  try:
    return _collect_with_find(motion_root, progress_callback=progress_callback)
  except Exception as exc:
    print(f"[WARN] find backend failed, falling back to Python scanner: {exc}")
    return _collect_with_python(motion_root, progress_callback=progress_callback)


def _make_tqdm_progress_callback(enabled: bool) -> ProgressReporter:
  if not enabled:
    return ProgressReporter(callback=None, close=lambda: None)
  progress_bar = tqdm(
    desc="Scanning motion files",
    unit="files",
    dynamic_ncols=True,
    mininterval=0.5,
  )

  def update(count: int) -> None:
    progress_bar.update(count - progress_bar.n)

  return ProgressReporter(callback=update, close=progress_bar.close)


def write_motion_manifest(
  motion_root: str | Path,
  manifest_file: str | Path,
  *,
  backend: Backend = "auto",
  relative_to: str | Path | None = None,
  progress_callback: ProgressCallback | None = None,
  show_progress: bool = False,
) -> ManifestResult:
  start = time.perf_counter()
  motion_root = Path(motion_root).expanduser().resolve()
  manifest_file = Path(manifest_file).expanduser()
  if not motion_root.is_dir():
    raise ValueError(f"motion_root must be a directory: {motion_root}")

  tqdm_progress = _make_tqdm_progress_callback(show_progress)

  def report_progress(count: int) -> None:
    if tqdm_progress.callback is not None:
      tqdm_progress.callback(count)
    if progress_callback is not None:
      progress_callback(count)

  try:
    motion_files = _collect_motion_files(
      motion_root,
      backend,
      progress_callback=report_progress,
    )
  finally:
    tqdm_progress.close()
  if relative_to is None:
    manifest_paths = [str(path.resolve()) for path in motion_files]
  else:
    relative_root = Path(relative_to).expanduser().resolve()
    manifest_paths = [
      path.resolve().relative_to(relative_root).as_posix() for path in motion_files
    ]
  manifest_paths.sort()

  manifest_file.parent.mkdir(parents=True, exist_ok=True)
  tmp_file = manifest_file.with_name(f"{manifest_file.name}.tmp.{os.getpid()}")
  with tmp_file.open("w", encoding="utf-8") as f:
    for motion_file in manifest_paths:
      f.write(motion_file + "\n")
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp_file, manifest_file)
  return ManifestResult(
    count=len(manifest_paths),
    elapsed_s=time.perf_counter() - start,
    manifest_file=manifest_file,
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Generate a sorted .npz motion manifest for LargeDatasetMultiMotionCommand. "
      "Use the generated file with --env.commands.motion.motion-manifest-file."
    )
  )
  parser.add_argument("motion_root", type=Path, help="Root directory to scan.")
  parser.add_argument("manifest_file", type=Path, help="Output manifest text file.")
  parser.add_argument(
    "--backend",
    choices=("auto", "find", "python"),
    default="auto",
    help="Scanner backend. `auto` uses GNU find when available.",
  )
  parser.add_argument(
    "--relative",
    action="store_true",
    help=(
      "Write paths relative to motion_root. Absolute paths are safer when training "
      "can start from different working directories."
    ),
  )
  parser.add_argument(
    "--no-progress",
    action="store_true",
    help="Disable the scanning progress bar.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  result = write_motion_manifest(
    args.motion_root,
    args.manifest_file,
    backend=args.backend,
    relative_to=args.motion_root if args.relative else None,
    show_progress=not args.no_progress,
  )
  print(
    "Wrote motion manifest: "
    f"count={result.count} file={result.manifest_file} "
    f"elapsed={result.elapsed_s:.3f}s"
  )
  metadata_cache_file = str(result.manifest_file) + ".metadata.npz"
  print(f"Recommended metadata cache file: {metadata_cache_file}")


if __name__ == "__main__":
  main()
