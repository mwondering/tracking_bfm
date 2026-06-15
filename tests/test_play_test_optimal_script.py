from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_play_test_optimal_script_defaults_to_no_reg_no_dr_task() -> None:
  script = Path("scripts/play_test_optimal.sh")
  assert script.exists()

  subprocess.run(
    ["bash", "-n", str(script)],
    check=True,
    capture_output=True,
    text=True,
  )

  text = script.read_text()
  assert "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR" in text
  assert 'MOTION_FILE="${MOTION_FILE:-}"' in text
  assert 'CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"' in text
  assert "--motion-file" in text
  assert "--checkpoint-file" in text


def test_play_test_optimal_script_dry_run_quotes_paths(tmp_path: Path) -> None:
  script = Path("scripts/play_test_optimal.sh")
  motion_file = tmp_path / "recorded motion.npz"
  checkpoint_file = tmp_path / "test optimal policy.pt"
  motion_file.write_text("dummy")
  checkpoint_file.write_text("dummy")

  result = subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env={
      **os.environ,
      "MOTION_FILE": str(motion_file),
      "CHECKPOINT_FILE": str(checkpoint_file),
      "DRY_RUN": "true",
    },
  )

  stdout = result.stdout
  assert "[DRY RUN]" in stdout
  assert "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR" in stdout
  assert "recorded\\ motion.npz" in stdout
  assert "test\\ optimal\\ policy.pt" in stdout
  assert "--env.commands.motion.history_steps 0" in stdout
  assert "--env.commands.motion.future_steps 1" in stdout
  assert "--viewer viser" in stdout
  assert "--num-envs 1" in stdout
