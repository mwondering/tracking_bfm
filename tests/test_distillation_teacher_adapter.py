"""Tests for distillation teacher adapter behavior."""

import torch

from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter


def test_teacher_adapter_uses_deterministic_mean_action() -> None:
  called = {"count": 0}

  def _policy(obs: torch.Tensor) -> torch.Tensor:
    called["count"] += 1
    return obs + 1.0

  adapter = TeacherPolicyAdapter(_policy)
  obs = torch.zeros(2, 3)
  out = adapter.act_mean(obs)

  assert adapter.uses_deterministic_mean_action is True
  assert called["count"] == 1
  assert torch.allclose(out, torch.ones(2, 3))
