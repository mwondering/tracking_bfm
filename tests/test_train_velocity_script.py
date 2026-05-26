"""Tests for velocity tracking training script."""

from pathlib import Path
import subprocess


def test_train_velocity_script_defaults() -> None:
  script = Path("scripts/train_velocity.sh")
  assert script.exists()

  result = subprocess.run(
    ["bash", "-n", str(script)],
    check=True,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0
  text = script.read_text()
  assert "Mjlab-Velocity-Flat-Unitree-G1" in text
  assert "--env.scene.num-envs" in text
  assert "--agent.experiment_name" in text
  assert "--agent.max_iterations" in text
