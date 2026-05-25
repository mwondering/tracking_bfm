"""Tests for latent velocity RL training script."""

from pathlib import Path
import subprocess


def test_train_latent_velocity_rl_script_defaults() -> None:
  script = Path("scripts/train_latent_velocity_rl.sh")
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
  assert "--agent.latent_decoder_checkpoint_path" in text
  assert "--agent.latent_action_clip 6.0" in text
  assert "--agent.clip_actions None" not in text
