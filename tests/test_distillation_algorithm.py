"""Tests for distillation student models and action distillation updates."""

import torch
from tensordict import TensorDict

from mjlab.tasks.distillation.rl.algorithm import ActionDistillationAlgorithm
from mjlab.tasks.distillation.rl.models import build_student_model


def test_build_student_model_matches_action_dim() -> None:
  obs = TensorDict({"student_actor": torch.zeros(2, 6)}, batch_size=[2])
  model = build_student_model(
    obs=obs,
    student_obs_group="student_actor",
    action_dim=4,
    hidden_dims=(32, 16),
    activation="elu",
  )

  out = model(obs)

  assert out.shape == (2, 4)


def test_action_distillation_algorithm_updates_student() -> None:
  obs = TensorDict({"student_actor": torch.randn(32, 5)}, batch_size=[32])
  teacher_actions = obs["student_actor"][:, :3] * 0.5
  model = build_student_model(
    obs=obs,
    student_obs_group="student_actor",
    action_dim=3,
    hidden_dims=(32, 32),
    activation="elu",
  )
  algorithm = ActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
  )

  with torch.no_grad():
    before = torch.nn.functional.mse_loss(model(obs), teacher_actions).item()

  metrics = algorithm.update(
    student_obs=obs,
    teacher_actions=teacher_actions,
    num_learning_epochs=4,
    num_mini_batches=4,
  )

  with torch.no_grad():
    after = torch.nn.functional.mse_loss(model(obs), teacher_actions).item()

  assert "action_mse" in metrics
  assert "action_l1" in metrics
  assert "grad_norm" in metrics
  assert after < before
