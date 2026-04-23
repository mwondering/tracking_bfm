"""Tests for distillation teacher mixing schedules."""

from mjlab.tasks.distillation.rl.schedules import LinearTeacherMixSchedule


def test_linear_teacher_mix_schedule_decays() -> None:
  schedule = LinearTeacherMixSchedule(beta_start=1.0, beta_end=0.0, decay_steps=100)

  assert schedule(0) == 1.0
  assert schedule(100) == 0.0
  assert 0.0 < schedule(50) < 1.0


def test_linear_teacher_mix_schedule_clamps_to_end() -> None:
  schedule = LinearTeacherMixSchedule(beta_start=0.8, beta_end=0.1, decay_steps=10)

  assert schedule(1000) == 0.1
