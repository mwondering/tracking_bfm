"""Tests for rough terrain velocity play helper scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_play_latent_terrain_script_defaults() -> None:
  script = Path("scripts/play_latent_terrain.sh")
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
  assert 'POLICY_CKPT="${POLICY_CKPT:-}"' in text
  assert 'LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-}"' in text
  assert "--rl.latent-decoder-checkpoint-path" in text
  assert "--checkpoint-file" in text


def test_play_baseline_terrain_script_defaults() -> None:
  script = Path("scripts/play_baseline_terrain.sh")
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
  assert 'POLICY_CKPT="${POLICY_CKPT:-}"' in text
  assert "--rl.latent-decoder-checkpoint-path" not in text
  assert "--checkpoint-file" in text


def test_play_terrain_scripts_dry_run_quote_checkpoint_paths(
  tmp_path: Path,
) -> None:
  latent_policy = tmp_path / "latent policy.pt"
  decoder = tmp_path / "latent decoder.pt"
  baseline_policy = tmp_path / "baseline policy.pt"
  for path in (latent_policy, decoder, baseline_policy):
    path.write_text("dummy")

  latent_result = subprocess.run(
    [
      "bash",
      "scripts/play_latent_terrain.sh",
    ],
    check=True,
    capture_output=True,
    text=True,
    env={
      **os.environ,
      "POLICY_CKPT": str(latent_policy),
      "LATENT_DECODER_CKPT": str(decoder),
      "DRY_RUN": "true",
    },
  )
  baseline_result = subprocess.run(
    [
      "bash",
      "scripts/play_baseline_terrain.sh",
    ],
    check=True,
    capture_output=True,
    text=True,
    env={
      **os.environ,
      "POLICY_CKPT": str(baseline_policy),
      "DRY_RUN": "true",
    },
  )

  assert "[DRY RUN]" in latent_result.stdout
  assert "--rl.latent-decoder-checkpoint-path" in latent_result.stdout
  assert "latent\\ policy.pt" in latent_result.stdout
  assert "latent\\ decoder.pt" in latent_result.stdout
  assert "[DRY RUN]" in baseline_result.stdout
  assert "--rl.latent-decoder-checkpoint-path" not in baseline_result.stdout
  assert "baseline\\ policy.pt" in baseline_result.stdout
