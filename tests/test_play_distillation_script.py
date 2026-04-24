from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_play_distillation_script_forwards_reference_motion_toggle(
  tmp_path: Path,
) -> None:
  script = Path("scripts/play_distillation.sh")
  motion_file = tmp_path / "motion.npz"
  checkpoint_file = tmp_path / "model.pt"

  motion_file.write_text("dummy")
  checkpoint_file.write_text("dummy")

  env = os.environ.copy()
  env.update(
    {
      "CHECKPOINT_FILE": str(checkpoint_file),
      "MOTION_FILE": str(motion_file),
      "SHOW_REFERENCE_MOTION": "false",
      "DRY_RUN": "true",
    }
  )

  result = subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  stdout = result.stdout
  assert "[DRY RUN]" in stdout
  assert "--show-reference-motion" in stdout
  assert "False" in stdout
