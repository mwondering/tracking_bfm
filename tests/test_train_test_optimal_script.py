from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run_script_with_uv_stub(tmp_path: Path, **env_overrides: str) -> list[str]:
  script = Path("scripts/train_test_optimal.sh")
  uv_stub = tmp_path / "uv"
  args_file = tmp_path / "uv-args.txt"

  uv_stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$UV_ARGS_FILE"\n')
  uv_stub.chmod(0o755)

  env = os.environ.copy()
  env["PATH"] = f"{tmp_path}:{env['PATH']}"
  env["UV_ARGS_FILE"] = str(args_file)
  env.update(env_overrides)

  subprocess.run(
    ["bash", str(script)],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )

  return args_file.read_text().splitlines()


def test_train_test_optimal_script_defaults_to_no_reg_no_dr_task(
  tmp_path: Path,
) -> None:
  args = _run_script_with_uv_stub(tmp_path)

  assert args[:3] == [
    "run",
    "train",
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR",
  ]
  assert "--agent.resume" not in args
  assert args[args.index("--env.commands.motion.sampling-mode") + 1] == "adaptive"
  assert args[args.index("--env.commands.motion.history_steps") + 1] == "0"
  assert args[args.index("--env.commands.motion.future_steps") + 1] == "1"
  assert args[args.index("--gpu_ids") + 1] == "[5,6]"


def test_train_test_optimal_script_can_select_regularized_control_task(
  tmp_path: Path,
) -> None:
  args = _run_script_with_uv_stub(
    tmp_path,
    DISABLE_REG_AND_DR="False",
    ATTENTION_VARIANT="mlp",
  )

  assert args[:3] == [
    "run",
    "train",
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal",
  ]
  assert args[args.index("--agent.run_name") + 1] == (
    "test_optimal_global_body_full_obs_with_reg_dr"
  )


def test_train_test_optimal_script_can_select_sparsetrack_attention_task(
  tmp_path: Path,
) -> None:
  args = _run_script_with_uv_stub(tmp_path, ATTENTION_VARIANT="sparsetrack_full_ref")

  assert args[:3] == [
    "run",
    "train",
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR",
  ]
  assert args[args.index("--agent.run_name") + 1] == (
    "test_optimal_sparsetrack_full_ref_attn_no_reg_no_dr"
  )


def test_train_test_optimal_script_can_select_hist_proprio_actor_critic_task(
  tmp_path: Path,
) -> None:
  args = _run_script_with_uv_stub(
    tmp_path,
    ATTENTION_VARIANT="hist_proprio_cross_actor_critic",
  )

  assert args[:3] == [
    "run",
    "train",
    (
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-"
      "HistProprioCrossAttnActorCritic-NoRegNoDR"
    ),
  ]
  assert args[args.index("--agent.run_name") + 1] == (
    "test_optimal_hist_proprio_cross_attn_actor_critic_no_reg_no_dr"
  )
