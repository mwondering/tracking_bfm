"""Tests for distillation student models and action distillation updates."""

import torch
from tensordict import TensorDict
from unittest.mock import patch

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


def test_action_distillation_algorithm_broadcast_parameters_noops_without_multi_gpu() -> None:
  obs = TensorDict({"student_actor": torch.randn(8, 5)}, batch_size=[8])
  model = build_student_model(
    obs=obs,
    student_obs_group="student_actor",
    action_dim=3,
    hidden_dims=(16, 16),
    activation="elu",
  )
  algorithm = ActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-3,
    max_grad_norm=1.0,
  )

  with patch("torch.distributed.broadcast_object_list") as broadcast:
    algorithm.broadcast_parameters()

  broadcast.assert_not_called()


def test_action_distillation_algorithm_multi_gpu_syncs_parameters_and_gradients() -> None:
  obs = TensorDict({"student_actor": torch.randn(16, 5)}, batch_size=[16])
  teacher_actions = obs["student_actor"][:, :3] * 0.25
  model = build_student_model(
    obs=obs,
    student_obs_group="student_actor",
    action_dim=3,
    hidden_dims=(16, 16),
    activation="elu",
  )
  algorithm = ActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-3,
    max_grad_norm=1.0,
    multi_gpu_cfg={"global_rank": 1, "local_rank": 1, "world_size": 2},
  )

  reduce_calls: list[torch.Tensor] = []

  def _record_all_reduce(tensor: torch.Tensor, op=None):
    reduce_calls.append(tensor.detach().clone())
    return None

  with (
    patch("torch.distributed.broadcast_object_list") as broadcast,
    patch("torch.distributed.all_reduce", side_effect=_record_all_reduce) as all_reduce,
  ):
    algorithm.broadcast_parameters()
    algorithm.update(
      student_obs=obs,
      teacher_actions=teacher_actions,
      num_learning_epochs=1,
      num_mini_batches=2,
    )

  broadcast.assert_called_once()
  assert all_reduce.call_count > 0
  assert reduce_calls
