from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_train_1stage_script_defaults_to_sparse_tracking_task(tmp_path: Path) -> None:
  script = Path("scripts/train_1stage_tracking_adaptive_sampling_h100.sh")
  uv_stub = tmp_path / "uv"
  args_file = tmp_path / "uv-args.txt"

  uv_stub.write_text(
    "#!/usr/bin/env bash\n"
    "printf '%s\\n' \"$@\" > \"$UV_ARGS_FILE\"\n"
  )
  uv_stub.chmod(0o755)

  env = os.environ.copy()
  env["PATH"] = f"{tmp_path}:{env['PATH']}"
  env["UV_ARGS_FILE"] = str(args_file)

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  args = args_file.read_text().splitlines()
  assert args[:3] == [
    "run",
    "train",
    "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage",
  ]
  assert "--agent.resume" not in args
