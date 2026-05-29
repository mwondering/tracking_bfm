"""Tests for latent tracking play helper script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_play_latent_tracking_script_defaults() -> None:
  script = Path("scripts/play_latent_tracking_1stage.sh")
  assert script.exists()

  result = subprocess.run(
    ["bash", "-n", str(script)],
    check=True,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0
  text = script.read_text()
  assert "Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage" in text
  assert 'MOTION_FILE="${MOTION_FILE:-}"' in text
  assert 'POLICY_CKPT="${POLICY_CKPT:-}"' in text
  assert 'LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-}"' in text
  assert "--motion-file" in text
  assert "--checkpoint-file" in text
  assert "--rl.latent-decoder-checkpoint-path" in text


def test_play_latent_tracking_script_dry_run_quotes_paths(
  tmp_path: Path,
) -> None:
  script = Path("scripts/play_latent_tracking_1stage.sh")
  motion_file = tmp_path / "motion clip.npz"
  policy = tmp_path / "latent policy.pt"
  decoder = tmp_path / "latent decoder.pt"
  for path in (motion_file, policy, decoder):
    path.write_text("dummy")

  result = subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env={
      **os.environ,
      "MOTION_FILE": str(motion_file),
      "POLICY_CKPT": str(policy),
      "LATENT_DECODER_CKPT": str(decoder),
      "DRY_RUN": "true",
    },
  )

  stdout = result.stdout
  assert "[DRY RUN]" in stdout
  assert "motion\\ clip.npz" in stdout
  assert "latent\\ policy.pt" in stdout
  assert "latent\\ decoder.pt" in stdout
  assert "--motion-file" in stdout
  assert "--checkpoint-file" in stdout
  assert "--rl.latent-decoder-checkpoint-path" in stdout
