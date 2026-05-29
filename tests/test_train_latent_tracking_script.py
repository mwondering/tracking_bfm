"""Tests for latent tracking training helper script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_train_latent_tracking_script_defaults() -> None:
  script = Path("scripts/train_latent_tracking_1stage.sh")
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
  assert 'MOTION_PATH="${MOTION_PATH:-}"' in text
  assert 'LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-}"' in text
  assert "--agent.latent_decoder_checkpoint_path" in text
  assert "--agent.latent_action_clip" in text
  assert "--agent.resume" in text
  assert "--agent.load_run" in text
  assert "--agent.load_checkpoint" in text


def test_train_latent_tracking_script_dry_run_quotes_paths(
  tmp_path: Path,
) -> None:
  script = Path("scripts/train_latent_tracking_1stage.sh")
  motion_path = tmp_path / "motion clips"
  decoder = tmp_path / "latent decoder.pt"
  motion_path.mkdir()
  decoder.write_text("dummy")

  result = subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env={
      **os.environ,
      "MOTION_PATH": str(motion_path),
      "LATENT_DECODER_CKPT": str(decoder),
      "RESUME": "true",
      "LOAD_RUN": "2026-05-29 latent run",
      "LOAD_CHECKPOINT": "model 1000.pt",
      "DRY_RUN": "true",
    },
  )

  stdout = result.stdout
  assert "[DRY RUN]" in stdout
  assert "motion\\ clips" in stdout
  assert "latent\\ decoder.pt" in stdout
  assert "--agent.resume True" in stdout
  assert "2026-05-29\\ latent\\ run" in stdout
  assert "model\\ 1000.pt" in stdout
