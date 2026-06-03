"""Tests for wbteleop training helper script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_train_wbteleop_script_defaults_to_wbteleop_task(tmp_path: Path) -> None:
  script = Path("scripts/train_wbteleop.sh")
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
    "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop",
  ]
  assert "--agent.algorithm.teacher_checkpoint_path" in args
  assert "--agent.algorithm.bc_weight_start" in args
  assert "--agent.algorithm.bc_weight_end" in args
  assert "--agent.algorithm.bc_decay_steps" in args
  assert "--env.commands.motion.history_steps" in args
  assert "--env.commands.motion.future_steps" in args
  assert "--env.observations.actor.terms.projected_gravity.history_length" in args
  assert "--env.observations.actor.terms.actions.history_length" in args


def test_train_wbteleop_script_dry_run_quotes_paths(tmp_path: Path) -> None:
  script = Path("scripts/train_wbteleop.sh")
  motion_path = tmp_path / "motion clips"
  teacher_ckpt = tmp_path / "teacher model.pt"
  motion_path.mkdir()
  teacher_ckpt.write_text("dummy")

  result = subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env={
      **os.environ,
      "MOTION_PATH": str(motion_path),
      "TEACHER_CKPT": str(teacher_ckpt),
      "DRY_RUN": "true",
    },
  )

  stdout = result.stdout
  assert "[DRY RUN]" in stdout
  assert "motion\\ clips" in stdout
  assert "teacher\\ model.pt" in stdout
