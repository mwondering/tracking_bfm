from __future__ import annotations

import subprocess
from pathlib import Path


def test_play_tracking_script_handles_paths_with_spaces(tmp_path: Path) -> None:
  script = Path("scripts/play_tracking.sh")
  motion_file = tmp_path / "motion clip.npz"
  checkpoint_dir = tmp_path / "teacher amass lafan noiton sonic"
  checkpoint_dir.mkdir()
  checkpoint_file = checkpoint_dir / "model 8500.pt"

  motion_file.write_text("dummy")
  checkpoint_file.write_text("dummy")

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
