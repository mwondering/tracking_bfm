"""Tests for latent distillation training helper scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_train_latentdistillation_h100_defaults_to_bfmzero_sphere(
  tmp_path: Path,
) -> None:
  script = Path("scripts/train_latentdistillation_h100.sh")
  uv_stub = tmp_path / "uv"
  args_file = tmp_path / "uv-args.txt"

  uv_stub.write_text(
    "#!/usr/bin/env bash\n"
    "printf '%s\\n' \"$@\" > \"$UV_ARGS_FILE\"\n"
  )
  uv_stub.chmod(0o755)

  env = os.environ.copy()
  env["PATH"] = f"{tmp_path}:{env['PATH']}"
  env["UV_ARGS_FILE"] = str(args_file)

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  args = args_file.read_text().splitlines()

  assert args[:3] == [
    "run",
    "train",
    "Mjlab-LatentDistillation-Flat-Unitree-G1",
  ]
  assert args[args.index("--agent.latent_regularization") + 1] == "bfmzero_sphere"
  assert "--agent.sphere_orthonormal_weight" in args
  assert "--agent.sphere_knn_smooth_weight" in args
  assert "--agent.sphere_knn_k" in args
  assert "--agent.sphere_knn_max_samples" in args


def test_train_latentdistillation_defaults_to_bfmzero_sphere(
  tmp_path: Path,
) -> None:
  script = Path("scripts/train_latentdistillation.sh")
  uv_stub = tmp_path / "uv"
  args_file = tmp_path / "uv-args.txt"

  uv_stub.write_text(
    "#!/usr/bin/env bash\n"
    "printf '%s\\n' \"$@\" > \"$UV_ARGS_FILE\"\n"
  )
  uv_stub.chmod(0o755)

  env = os.environ.copy()
  env["PATH"] = f"{tmp_path}:{env['PATH']}"
  env["UV_ARGS_FILE"] = str(args_file)

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  args = args_file.read_text().splitlines()

  assert args[:3] == [
    "run",
    "train",
    "Mjlab-LatentDistillation-Flat-Unitree-G1",
  ]
  assert args[args.index("--agent.latent_regularization") + 1] == "bfmzero_sphere"
  assert "--agent.sphere_orthonormal_weight" in args
  assert "--agent.sphere_knn_smooth_weight" in args
  assert "--agent.sphere_knn_k" in args
  assert "--agent.sphere_knn_max_samples" in args
