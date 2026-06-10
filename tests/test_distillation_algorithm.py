"""Tests for distillation student models and action distillation updates."""

from unittest.mock import patch

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


def test_latent_distillation_model_outputs_actions_and_latent_stats() -> None:
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(5, 7),
      "proprio_actor": torch.randn(5, 4),
    },
    batch_size=[5],
  )
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=6,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    activation="elu",
  )

  actions, latent = model(obs, deterministic=True)

  assert actions.shape == (5, 3)
  assert latent["mu"].shape == (5, 6)
  assert latent["log_std"].shape == (5, 6)
  assert latent["z"].shape == (5, 6)


def test_latent_distillation_model_projects_spherical_latent_and_slerp() -> None:
  from mjlab.tasks.distillation.rl.models import (
    LatentDistillationModel,
    build_latent_student_model,
  )

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(5, 7),
      "proprio_actor": torch.randn(5, 4),
    },
    batch_size=[5],
  )
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=6,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    activation="elu",
    latent_mode="bfmzero_sphere",
  )

  _, latent = model(obs, deterministic=True)
  expected_radius = torch.sqrt(torch.tensor(6.0))

  assert latent["z_raw"].shape == (5, 6)
  assert latent["z_sphere"].shape == (5, 6)
  assert torch.allclose(
    latent["z"].norm(dim=-1),
    torch.full((5,), expected_radius),
    atol=1.0e-5,
  )

  z0 = LatentDistillationModel.spherical_project(torch.randn(3, 6), radius=-1.0)
  z1 = LatentDistillationModel.spherical_project(torch.randn(3, 6), radius=-1.0)
  interpolated = LatentDistillationModel.slerp(z0, z1, 0.5, radius=-1.0)
  zero_projected = LatentDistillationModel.spherical_project(torch.zeros(2, 6))

  assert torch.allclose(
    interpolated.norm(dim=-1),
    torch.full((3,), expected_radius),
    atol=1.0e-5,
  )
  assert torch.allclose(
    zero_projected.norm(dim=-1),
    torch.full((2,), expected_radius),
    atol=1.0e-5,
  )


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


def test_latent_action_distillation_algorithm_updates_student() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(32, 7),
      "proprio_actor": torch.randn(32, 4),
    },
    batch_size=[32],
  )
  teacher_actions = obs["teacher_actor"][:, :3] * 0.25
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=5,
    encoder_hidden_dims=(32, 32),
    decoder_hidden_dims=(32, 32),
    activation="elu",
  )
  algorithm = LatentActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
    kl_weight=1.0e-3,
    kl_warmup_iterations=10,
    free_nats_per_dim=0.01,
    latent_smooth_weight=1.0e-3,
  )

  before = {name: param.detach().clone() for name, param in model.named_parameters()}
  metrics = algorithm.update(
    obs=obs,
    teacher_actions=teacher_actions,
    num_learning_epochs=2,
    num_mini_batches=4,
    iteration=5,
  )

  assert "action_mse" in metrics
  assert "action_l1" in metrics
  assert "kl_loss" in metrics
  assert "kl_per_dim" in metrics
  assert "kl_weight" in metrics
  assert "latent_mu_norm" in metrics
  assert "latent_std_mean" in metrics
  assert "latent_smooth_loss" in metrics
  assert "total_loss" in metrics
  assert "grad_norm" in metrics
  assert any(
    not torch.allclose(before[name], param.detach())
    for name, param in model.named_parameters()
  )


def test_latent_action_distillation_algorithm_supports_wae_mmd_regularization() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(24, 7),
      "proprio_actor": torch.randn(24, 4),
    },
    batch_size=[24],
  )
  teacher_actions = obs["teacher_actor"][:, :3] * 0.25
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=5,
    encoder_hidden_dims=(32, 32),
    decoder_hidden_dims=(32, 32),
    activation="elu",
  )
  algorithm = LatentActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
    latent_regularization="wae_mmd",
    mmd_weight=1.0e-2,
    mmd_kernel_scales=(0.5, 1.0, 2.0),
    kl_weight=1.0e-3,
    kl_warmup_iterations=10,
  )

  metrics = algorithm.update(
    obs=obs,
    teacher_actions=teacher_actions,
    num_learning_epochs=1,
    num_mini_batches=3,
    iteration=5,
  )

  assert metrics["kl_weight"] == 0.0
  assert metrics["mmd_weight"] == 1.0e-2
  assert metrics["mmd_loss"] >= 0.0
  assert "aggregate_mean_norm" in metrics
  assert "aggregate_std_mean" in metrics
  assert "total_loss" in metrics


def test_latent_kl_regularization_does_not_compute_mmd() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(16, 7),
      "proprio_actor": torch.randn(16, 4),
    },
    batch_size=[16],
  )
  teacher_actions = obs["teacher_actor"][:, :3] * 0.25
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    activation="elu",
  )
  algorithm = LatentActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
    latent_regularization="kl",
    mmd_weight=1.0,
  )

  with patch.object(
    LatentActionDistillationAlgorithm,
    "_mmd_rbf",
    side_effect=AssertionError("MMD should not be computed in KL mode"),
  ):
    metrics = algorithm.update(
      obs=obs,
      teacher_actions=teacher_actions,
      num_learning_epochs=1,
      num_mini_batches=2,
      iteration=1,
    )

  assert metrics["mmd_weight"] == 0.0
  assert metrics["mmd_loss"] == 0.0


def test_wae_mmd_subsamples_pairwise_kernel_inputs() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(48, 7),
      "proprio_actor": torch.randn(48, 4),
    },
    batch_size=[48],
  )
  teacher_actions = obs["teacher_actor"][:, :3] * 0.25
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    activation="elu",
  )
  algorithm = LatentActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
    latent_regularization="wae_mmd",
    mmd_weight=1.0e-2,
    mmd_max_samples=7,
  )
  cdist_rows: list[int] = []
  original_cdist = torch.cdist

  def _record_cdist(x1: torch.Tensor, x2: torch.Tensor):
    cdist_rows.extend([x1.shape[0], x2.shape[0]])
    return original_cdist(x1, x2)

  with patch("torch.cdist", side_effect=_record_cdist):
    algorithm.update(
      obs=obs,
      teacher_actions=teacher_actions,
      num_learning_epochs=1,
      num_mini_batches=1,
      iteration=1,
    )

  assert cdist_rows
  assert max(cdist_rows) <= 7


def test_latent_action_distillation_algorithm_supports_bfmzero_sphere_regularization() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(32, 7),
      "proprio_actor": torch.randn(32, 4),
    },
    batch_size=[32],
  )
  teacher_actions = obs["teacher_actor"][:, :3] * 0.25
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=5,
    encoder_hidden_dims=(32, 32),
    decoder_hidden_dims=(32, 32),
    activation="elu",
    latent_mode="bfmzero_sphere",
  )
  algorithm = LatentActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
    latent_regularization="bfmzero_sphere",
    kl_weight=1.0,
    mmd_weight=1.0,
    sphere_orthonormal_weight=1.0e-2,
    sphere_knn_smooth_weight=1.0e-2,
    sphere_knn_k=3,
    sphere_knn_max_samples=11,
  )

  metrics = algorithm.update(
    obs=obs,
    teacher_actions=teacher_actions,
    num_learning_epochs=1,
    num_mini_batches=2,
    iteration=5,
  )

  assert metrics["kl_weight"] == 0.0
  assert metrics["mmd_weight"] == 0.0
  assert metrics["sphere_orthonormal_weight"] == 1.0e-2
  assert metrics["sphere_knn_smooth_weight"] == 1.0e-2
  assert metrics["sphere_orthonormal_loss"] >= 0.0
  assert metrics["sphere_knn_smooth_loss"] >= 0.0
  assert abs(metrics["sphere_radius_mean"] - torch.sqrt(torch.tensor(5.0)).item()) < 1.0e-5
  assert "total_loss" in metrics


def test_bfmzero_sphere_knn_smooth_subsamples_distance_inputs() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm
  from mjlab.tasks.distillation.rl.models import build_latent_student_model

  obs = TensorDict(
    {
      "teacher_actor": torch.randn(48, 7),
      "proprio_actor": torch.randn(48, 4),
    },
    batch_size=[48],
  )
  teacher_actions = obs["teacher_actor"][:, :3] * 0.25
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=3,
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    activation="elu",
    latent_mode="bfmzero_sphere",
  )
  algorithm = LatentActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-2,
    max_grad_norm=1.0,
    latent_regularization="bfmzero_sphere",
    sphere_knn_smooth_weight=1.0e-2,
    sphere_knn_max_samples=9,
  )
  cdist_rows: list[int] = []
  original_cdist = torch.cdist

  def _record_cdist(x1: torch.Tensor, x2: torch.Tensor):
    cdist_rows.extend([x1.shape[0], x2.shape[0]])
    return original_cdist(x1, x2)

  with patch("torch.cdist", side_effect=_record_cdist):
    algorithm.update(
      obs=obs,
      teacher_actions=teacher_actions,
      num_learning_epochs=1,
      num_mini_batches=1,
      iteration=1,
    )

  assert cdist_rows
  assert max(cdist_rows) <= 9


def test_latent_smoothness_uses_temporal_pairs_and_done_mask() -> None:
  from mjlab.tasks.distillation.rl.algorithm import LatentActionDistillationAlgorithm

  mu = torch.tensor(
    [
      [0.0],
      [10.0],
      [1.0],
      [20.0],
      [2.0],
      [30.0],
    ]
  )
  dones = torch.tensor(
    [
      [False, True],
      [False, False],
      [False, False],
    ]
  )

  smoothness = LatentActionDistillationAlgorithm._trajectory_latent_smoothness(
    mu=mu,
    rollout_shape=(3, 2),
    dones=dones,
  )

  assert smoothness.item() == torch.tensor((1.0 + 1.0 + 100.0) / 3.0).item()


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
