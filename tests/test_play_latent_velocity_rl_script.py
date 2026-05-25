"""Tests for latent velocity RL play script."""

import subprocess
from pathlib import Path


def test_play_latent_velocity_rl_script_defaults() -> None:
  script = Path("scripts/play_latent_velocity_rl.sh")
  assert script.exists()

  result = subprocess.run(
    ["bash", "-n", str(script)],
    check=True,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0
  text = script.read_text()
  assert "Mjlab-LatentRL-Flat-Unitree-G1" in text
  assert "--rl.latent-decoder-checkpoint-path" in text
  assert "--checkpoint-file" in text
  assert "--stochastic-policy" in text
