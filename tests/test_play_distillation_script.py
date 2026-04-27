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


def test_play_distillation_script_forwards_student_history_overrides(
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
      "STUDENT_HISTORY_STEPS": "3",
      "STUDENT_FUTURE_STEPS": "4",
      "STUDENT_ROBOT_HISTORY_STEPS": "5",
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
  assert "--env.observations.student_actor.terms.ee_pose.params.history_steps" in stdout
  assert "--env.observations.student_actor.terms.ee_pose.params.future_steps" in stdout
  assert "--env.observations.student_actor.terms.base_lin_vel_w.params.history_steps" in stdout
  assert "--env.observations.student_actor.terms.base_lin_vel_w.params.future_steps" in stdout
  assert "--env.observations.student_actor.terms.base_ang_vel_w.params.history_steps" in stdout
  assert "--env.observations.student_actor.terms.base_ang_vel_w.params.future_steps" in stdout
  assert "--env.observations.student_actor.terms.anchor_height_w.params.history_steps" in stdout
  assert "--env.observations.student_actor.terms.anchor_height_w.params.future_steps" in stdout
  assert "--env.observations.student_actor.terms.projected_gravity.history_length" in stdout
  assert "--env.observations.student_actor.terms.base_ang_vel.history_length" in stdout
  assert "--env.observations.student_actor.terms.joint_pos.history_length" in stdout
  assert "--env.observations.student_actor.terms.joint_vel.history_length" in stdout
  assert "--env.observations.student_actor.terms.actions.history_length" in stdout
  assert "3" in stdout
  assert "4" in stdout
  assert "5" in stdout
