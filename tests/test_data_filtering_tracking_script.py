from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_data_filtering_tracking_script_supports_generate_dataset(
  tmp_path: Path,
) -> None:
  """The shell wrapper should expose the generate-dataset data-filtering mode."""
  script = Path("scripts/data_filtering_tracking.sh")
  uv_stub = tmp_path / "uv"
  args_file = tmp_path / "uv-args.txt"

  uv_stub.write_text(
    '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$UV_ARGS_FILE"\n',
    encoding="utf-8",
  )
  uv_stub.chmod(0o755)

  env = os.environ.copy()
  env["PATH"] = f"{tmp_path}:{env['PATH']}"
  env["UV_ARGS_FILE"] = str(args_file)
  env["MODE"] = "generate-dataset"
  env["MOTION_PATH"] = str(tmp_path / "motions")
  env["OUTPUT_MOTION_PATH"] = str(tmp_path / "generated")
  env["OUTPUT_FILE"] = str(tmp_path / "generated" / "report.json")
  env["CHECKPOINT_FILE"] = str(tmp_path / "teacher.pt")
  env["NUM_ENVS"] = "4"

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  args = args_file.read_text(encoding="utf-8").splitlines()
  assert args[:4] == [
    "run",
    "data-filtering",
    "generate-dataset",
    "Mjlab-Trackingbfm-Flat-Unitree-G1",
  ]
  assert "--output-motion-path" in args
  assert str(tmp_path / "generated") in args
  assert "--completion-threshold" in args
  assert "0.95" in args
