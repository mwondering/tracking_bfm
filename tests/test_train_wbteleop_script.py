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
  for key in ("RESUME", "LOAD_RUN", "LOAD_CHECKPOINT"):
    env.pop(key, None)

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
  assert "--agent.algorithm.pure_bc_enabled" in args
  assert "--agent.algorithm.pure_bc_weight" in args
  assert "--agent.algorithm.pure_bc_rollout" in args
  assert "--agent.algorithm.bc_actor_checkpoint_path" in args
  assert "--agent.algorithm.init_critic_from_teacher" in args
  assert "--agent.resume" in args
  assert args[args.index("--agent.algorithm.pure_bc_enabled") + 1] == "True"
  assert args[args.index("--agent.resume") + 1] == "False"
  assert "--agent.load_run" not in args
  assert "--agent.load_checkpoint" not in args


def test_train_wbteleop_script_can_disable_resume(tmp_path: Path) -> None:
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
  for key in ("RESUME", "LOAD_RUN", "LOAD_CHECKPOINT"):
    env.pop(key, None)
  env["RESUME"] = "False"

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  args = args_file.read_text().splitlines()
  assert "--agent.resume" in args
  assert args[args.index("--agent.resume") + 1] == "False"
  assert "--agent.load_run" not in args
  assert "--agent.load_checkpoint" not in args


def test_train_wbteleop_h100_script_defaults_to_scratch_with_bc_flags(
  tmp_path: Path,
) -> None:
  script = Path("scripts/train_wbteleop_h100.sh")
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
  for key in ("RESUME", "LOAD_RUN", "LOAD_CHECKPOINT"):
    env.pop(key, None)

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
  assert "--agent.algorithm.pure_bc_enabled" in args
  assert "--agent.algorithm.pure_bc_weight" in args
  assert "--agent.algorithm.pure_bc_rollout" in args
  assert "--agent.algorithm.bc_actor_checkpoint_path" in args
  assert "--agent.algorithm.init_critic_from_teacher" in args
  assert "--agent.algorithm.strict_init" in args
  assert "--agent.resume" in args
  assert args[args.index("--agent.resume") + 1] == "False"
  assert "--agent.load_run" not in args
  assert "--agent.load_checkpoint" not in args


def test_train_wbteleop_h100_script_can_resume(tmp_path: Path) -> None:
  script = Path("scripts/train_wbteleop_h100.sh")
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
  env["RESUME"] = "True"
  env["LOAD_RUN"] = "h100_run"
  env["LOAD_CHECKPOINT"] = "model_2000.pt"

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  args = args_file.read_text().splitlines()
  assert args[args.index("--agent.resume") + 1] == "True"
  assert args[args.index("--agent.load_run") + 1] == "h100_run"
  assert args[args.index("--agent.load_checkpoint") + 1] == "model_2000.pt"


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
