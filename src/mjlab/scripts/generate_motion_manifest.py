"""Generate a stable manifest for large motion datasets."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import numpy as np
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


@dataclass(frozen=True)
class MetadataCacheResult:
  count: int
  elapsed_s: float
  metadata_cache_file: Path


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


def _make_tqdm_progress_callback(enabled: bool, *, desc: str) -> ProgressReporter:
  if not enabled:
    return ProgressReporter(callback=None, close=lambda: None)
  progress_bar = tqdm(
    desc=desc,
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

  tqdm_progress = _make_tqdm_progress_callback(
    show_progress, desc="Scanning motion files"
  )

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


def reuse_or_write_motion_manifest(
  motion_root: str | Path,
  manifest_file: str | Path,
  *,
  backend: Backend = "auto",
  relative_to: str | Path | None = None,
  progress_callback: ProgressCallback | None = None,
  show_progress: bool = False,
  reuse_existing: bool = False,
) -> ManifestResult:
  manifest_file = Path(manifest_file).expanduser()
  if reuse_existing and manifest_file.exists():
    start = time.perf_counter()
    return ManifestResult(
      count=len(_read_manifest_file(manifest_file)),
      elapsed_s=time.perf_counter() - start,
      manifest_file=manifest_file,
    )
  return write_motion_manifest(
    motion_root,
    manifest_file,
    backend=backend,
    relative_to=relative_to,
    progress_callback=progress_callback,
    show_progress=show_progress,
  )


def _read_manifest_file(manifest_file: Path) -> list[str]:
  with manifest_file.open(encoding="utf-8") as f:
    return [line.strip() for line in f if line.strip()]


def _motion_files_hash(motion_files: list[str]) -> str:
  digest = hashlib.sha1()
  for motion_file in motion_files:
    digest.update(os.path.abspath(motion_file).encode("utf-8"))
    digest.update(b"\0")
  return digest.hexdigest()


def _extract_fps_value(fps_data: np.ndarray) -> tuple[float, bool, bool]:
  fps_array = np.asarray(fps_data).reshape(-1)
  if fps_array.size == 0:
    return 30.0, False, True
  return float(fps_array[0]), fps_array.size != 1, False


def _read_motion_metadata_job(job: tuple[int, str]) -> tuple[int, int, float, bool, bool]:
  index, motion_file = job
  if not os.path.isfile(motion_file):
    raise FileNotFoundError(f"Invalid motion file path: {motion_file}")
  with np.load(motion_file) as data:
    file_length = int(data["joint_pos"].shape[0])
    fps_value, is_non_scalar_fps, is_empty_fps = _extract_fps_value(data["fps"])
  return index, file_length, fps_value, is_non_scalar_fps, is_empty_fps


def _iter_motion_metadata(
  motion_files: list[str],
  *,
  workers: int,
  chunksize: int,
):
  jobs = enumerate(motion_files)
  if workers <= 1:
    for job in jobs:
      yield _read_motion_metadata_job(job)
    return
  with mp.Pool(processes=workers) as pool:
    yield from pool.imap_unordered(
      _read_motion_metadata_job,
      jobs,
      chunksize=max(int(chunksize), 1),
    )


def write_motion_metadata_cache(
  manifest_file: str | Path,
  metadata_cache_file: str | Path | None = None,
  *,
  workers: int | None = None,
  chunksize: int = 64,
  progress_callback: ProgressCallback | None = None,
  show_progress: bool = False,
) -> MetadataCacheResult:
  start = time.perf_counter()
  manifest_file = Path(manifest_file).expanduser()
  if metadata_cache_file is None:
    metadata_cache_file = Path(str(manifest_file) + ".metadata.npz")
  else:
    metadata_cache_file = Path(metadata_cache_file).expanduser()
  motion_files = _read_manifest_file(manifest_file)
  num_files = len(motion_files)
  if workers is None:
    workers = min(os.cpu_count() or 1, 8)
  workers = max(int(workers), 1)
  file_lengths = np.empty(num_files, dtype=np.int64)
  fps_values = np.empty(num_files, dtype=np.float32)
  non_scalar_fps_count = 0
  empty_fps_count = 0
  tqdm_progress = _make_tqdm_progress_callback(
    show_progress, desc="Reading motion metadata"
  )
  completed_count = 0

  try:
    for (
      index,
      file_length,
      fps_value,
      is_non_scalar_fps,
      is_empty_fps,
    ) in _iter_motion_metadata(
      motion_files,
      workers=workers,
      chunksize=chunksize,
    ):
      file_lengths[index] = file_length
      fps_values[index] = fps_value
      if is_non_scalar_fps:
        non_scalar_fps_count += 1
      if is_empty_fps:
        empty_fps_count += 1
      completed_count += 1
      if tqdm_progress.callback is not None:
        tqdm_progress.callback(completed_count)
      _notify_progress(progress_callback, completed_count)
  finally:
    tqdm_progress.close()

  metadata_cache_file.parent.mkdir(parents=True, exist_ok=True)
  tmp_file = metadata_cache_file.with_name(
    f"{metadata_cache_file.name}.tmp.{os.getpid()}.npz"
  )
  np.savez(
    tmp_file,
    version=np.array(1, dtype=np.int64),
    num_files=np.array(num_files, dtype=np.int64),
    motion_files_hash=np.array(_motion_files_hash(motion_files)),
    file_lengths=file_lengths,
    fps_values=fps_values,
    non_scalar_fps_count=np.array(non_scalar_fps_count, dtype=np.int64),
    empty_fps_count=np.array(empty_fps_count, dtype=np.int64),
  )
  os.replace(tmp_file, metadata_cache_file)
  return MetadataCacheResult(
    count=num_files,
    elapsed_s=time.perf_counter() - start,
    metadata_cache_file=metadata_cache_file,
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
  parser.add_argument(
    "--reuse-existing-manifest",
    action="store_true",
    help="If manifest_file exists, reuse it instead of scanning motion_root again.",
  )
  parser.add_argument(
    "--build-metadata-cache",
    action="store_true",
    help="Also build the .metadata.npz cache used by LargeDatasetMotionStore.",
  )
  parser.add_argument(
    "--metadata-cache-file",
    type=Path,
    default=None,
    help="Metadata cache output path. Defaults to <manifest_file>.metadata.npz.",
  )
  parser.add_argument(
    "--metadata-workers",
    type=int,
    default=None,
    help="Parallel workers for metadata cache generation. Defaults to min(cpu_count, 8).",
  )
  parser.add_argument(
    "--metadata-chunksize",
    type=int,
    default=64,
    help="Task chunk size for metadata worker processes.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  reused_existing_manifest = args.reuse_existing_manifest and args.manifest_file.exists()
  result = reuse_or_write_motion_manifest(
    args.motion_root,
    args.manifest_file,
    backend=args.backend,
    relative_to=args.motion_root if args.relative else None,
    show_progress=not args.no_progress,
    reuse_existing=args.reuse_existing_manifest,
  )
  action = "Reused existing motion manifest" if reused_existing_manifest else "Wrote motion manifest"
  print(
    f"{action}: count={result.count} file={result.manifest_file} "
    f"elapsed={result.elapsed_s:.3f}s"
  )
  metadata_cache_file = str(result.manifest_file) + ".metadata.npz"
  print(f"Recommended metadata cache file: {metadata_cache_file}")
  if args.build_metadata_cache:
    metadata_result = write_motion_metadata_cache(
      result.manifest_file,
      args.metadata_cache_file,
      workers=args.metadata_workers,
      chunksize=args.metadata_chunksize,
      show_progress=not args.no_progress,
    )
    print(
      "Wrote motion metadata cache: "
      f"count={metadata_result.count} file={metadata_result.metadata_cache_file} "
      f"elapsed={metadata_result.elapsed_s:.3f}s"
    )


if __name__ == "__main__":
  main()
