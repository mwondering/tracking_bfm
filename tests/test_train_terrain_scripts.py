"""Tests for terrain training helper scripts."""

import subprocess
from pathlib import Path


def test_train_latent_terrain_script_defaults() -> None:
  script = Path("scripts/train_latent_terrain.sh")
  assert script.exists()

  result = subprocess.run(
    ["bash", "-n", str(script)],
    check=True,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0
  text = script.read_text()
  assert "Mjlab-LatentRL-Rough-Unitree-G1" in text
  assert "--agent.latent_decoder_checkpoint_path" in text
  assert "--agent.latent_action_clip" in text
  assert "latent_rl_rough_g1" in text


def test_train_scratch_terrain_script_defaults() -> None:
  script = Path("scripts/train_scratch_terrain.sh")
  assert script.exists()

  result = subprocess.run(
    ["bash", "-n", str(script)],
    check=True,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0
  text = script.read_text()
  assert "Mjlab-Velocity-Rough-Unitree-G1" in text
  assert "--agent.latent_decoder_checkpoint_path" not in text
  assert "velocity_rough_g1" in text
