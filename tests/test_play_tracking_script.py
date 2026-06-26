from __future__ import annotations

import os
import subprocess
from pathlib import Path

import tyro

import mjlab
from mjlab.scripts.play import (
  PlayCliConfig,
  _apply_extra_reference_motion_file,
  _apply_play_domain_randomization_override,
)


def _make_play_tracking_files(tmp_path: Path) -> tuple[Path, Path]:
  motion_file = tmp_path / "motion clip.npz"
  checkpoint_dir = tmp_path / "teacher amass lafan noiton sonic"
  checkpoint_dir.mkdir()
  checkpoint_file = checkpoint_dir / "model 8500.pt"

  motion_file.write_text("dummy")
  checkpoint_file.write_text("dummy")

  return motion_file, checkpoint_file


def _make_extra_reference_motion_file(tmp_path: Path) -> Path:
  extra_motion_file = tmp_path / "extra ref motion.npz"
  extra_motion_file.write_text("dummy")
  return extra_motion_file


def test_play_tracking_script_handles_paths_with_spaces(tmp_path: Path) -> None:
  script = Path("scripts/play_tracking.sh")
  motion_file, checkpoint_file = _make_play_tracking_files(tmp_path)

  result = subprocess.run(
    [
      "bash",
      str(script),
      "--task",
      "Mjlab-Trackingbfm-Flat-Unitree-G1",
      "--motion-file",
      str(motion_file),
      "--checkpoint-file",
      str(checkpoint_file),
      "--num-envs",
      "1",
      "--viewer",
      "viser",
      "--dry-run",
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  stdout = result.stdout
  assert "[DRY RUN]" in stdout
  assert "--motion-file" in stdout
  assert "--checkpoint-file" in stdout
  assert "motion\\ clip.npz" in stdout
  assert "teacher\\ amass\\ lafan\\ noiton\\ sonic/model\\ 8500.pt" in stdout


def test_play_tracking_script_keeps_domain_randomization_by_default(
  tmp_path: Path,
) -> None:
  script = Path("scripts/play_tracking.sh")
  motion_file, checkpoint_file = _make_play_tracking_files(tmp_path)

  result = subprocess.run(
    [
      "bash",
      str(script),
      "--motion-file",
      str(motion_file),
      "--checkpoint-file",
      str(checkpoint_file),
      "--dry-run",
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  assert "--env.events" not in result.stdout


def test_play_tracking_script_can_disable_domain_randomization(
  tmp_path: Path,
) -> None:
  script = Path("scripts/play_tracking.sh")
  motion_file, checkpoint_file = _make_play_tracking_files(tmp_path)

  result = subprocess.run(
    [
      "bash",
      str(script),
      "--motion-file",
      str(motion_file),
      "--checkpoint-file",
      str(checkpoint_file),
      "--domain-randomization",
      "false",
      "--dry-run",
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  stdout = result.stdout
  assert "--domain-randomization" in stdout
  assert "False" in stdout


def test_play_tracking_script_can_disable_domain_randomization_from_env(
  tmp_path: Path,
) -> None:
  script = Path("scripts/play_tracking.sh")
  motion_file, checkpoint_file = _make_play_tracking_files(tmp_path)

  result = subprocess.run(
    [
      "bash",
      str(script),
      "--motion-file",
      str(motion_file),
      "--checkpoint-file",
      str(checkpoint_file),
      "--dry-run",
    ],
    check=True,
    capture_output=True,
    text=True,
    env={**os.environ, "DOMAIN_RANDOMIZATION": "false"},
  )

  stdout = result.stdout
  assert "--domain-randomization" in stdout
  assert "False" in stdout


def test_play_tracking_script_forwards_extra_reference_motion_file(
  tmp_path: Path,
) -> None:
  script = Path("scripts/play_tracking.sh")
  motion_file, checkpoint_file = _make_play_tracking_files(tmp_path)
  extra_motion_file = _make_extra_reference_motion_file(tmp_path)

  result = subprocess.run(
    [
      "bash",
      str(script),
      "--motion-file",
      str(motion_file),
      "--checkpoint-file",
      str(checkpoint_file),
      "--extra-reference-motion-file",
      str(extra_motion_file),
      "--dry-run",
    ],
    check=True,
    capture_output=True,
    text=True,
  )

  stdout = result.stdout
  assert "--extra-reference-motion-file" in stdout
  assert "extra\\ ref\\ motion.npz" in stdout


def test_play_cli_config_can_disable_domain_randomization() -> None:
  args = tyro.cli(
    PlayCliConfig,
    args=["--domain-randomization", "False"],
    default=PlayCliConfig.from_task("Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop"),
    config=mjlab.TYRO_FLAGS,
  )

  assert not args.domain_randomization
  assert args.env.events

  _apply_play_domain_randomization_override(args.env, args.domain_randomization)

  assert args.env.events == {}


def test_play_cli_config_can_apply_extra_reference_motion_file(tmp_path: Path) -> None:
  extra_motion_file = _make_extra_reference_motion_file(tmp_path)
  args = tyro.cli(
    PlayCliConfig,
    args=["--extra-reference-motion-file", str(extra_motion_file)],
    default=PlayCliConfig.from_task("Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop"),
    config=mjlab.TYRO_FLAGS,
  )

  assert args.extra_reference_motion_file == str(extra_motion_file)
  assert args.env.commands["motion"].extra_reference_motion_file == ""

  _apply_extra_reference_motion_file(args.env, args.extra_reference_motion_file)

  assert args.env.commands["motion"].extra_reference_motion_file == str(
    extra_motion_file
  )
