"""Pure action-distillation algorithm."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict


class ActionDistillationAlgorithm:
  """Update a student policy to match teacher mean actions."""

  def __init__(
    self,
    policy: torch.nn.Module,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    multi_gpu_cfg: dict | None = None,
  ):
    self.policy = policy
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
    self.is_multi_gpu = multi_gpu_cfg is not None
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = int(multi_gpu_cfg["global_rank"])
      self.gpu_world_size = int(multi_gpu_cfg["world_size"])
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

  def broadcast_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    model_params = [self.policy.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)
    self.policy.load_state_dict(model_params[0])

  def reduce_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    for param in self.policy.parameters():
      if param.grad is None:
        continue
      torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
      param.grad.div_(self.gpu_world_size)

  def update(
    self,
    student_obs: TensorDict,
    teacher_actions: torch.Tensor,
    num_learning_epochs: int,
    num_mini_batches: int,
  ) -> dict[str, float]:
    batch_size = teacher_actions.shape[0]
    if batch_size == 0:
      raise ValueError("teacher_actions must contain at least one sample")

    num_mini_batches = max(1, min(num_mini_batches, batch_size))
    mini_batch_size = math.ceil(batch_size / num_mini_batches)

    mse_total = 0.0
    l1_total = 0.0
    grad_norm_total = 0.0
    updates = 0

    self.policy.train()
    for _ in range(num_learning_epochs):
      permutation = torch.randperm(batch_size, device=teacher_actions.device)
      for start in range(0, batch_size, mini_batch_size):
        batch_idx = permutation[start : start + mini_batch_size]
        batch_obs = student_obs[batch_idx]
        batch_teacher = teacher_actions[batch_idx]

        if hasattr(self.policy, "update_normalization"):
          self.policy.update_normalization(batch_obs)

        pred_actions = self.policy(batch_obs)
        mse_loss = F.mse_loss(pred_actions, batch_teacher)
        l1_loss = F.l1_loss(pred_actions, batch_teacher)

        self.optimizer.zero_grad(set_to_none=True)
        mse_loss.backward()
        self.reduce_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(
          self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        mse_total += float(mse_loss.item())
        l1_total += float(l1_loss.item())
        grad_norm_total += float(grad_norm.item())
        updates += 1

    return {
      "action_mse": mse_total / updates,
      "action_l1": l1_total / updates,
      "grad_norm": grad_norm_total / updates,
    }

  def save(self) -> dict:
    return {
      "policy_state_dict": self.policy.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "learning_rate": self.learning_rate,
    }

  def load(self, checkpoint: dict) -> None:
    self.policy.load_state_dict(checkpoint["policy_state_dict"])
    if "optimizer_state_dict" in checkpoint:
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class LatentActionDistillationAlgorithm:
  """Update a latent encoder/decoder student to match teacher actions."""

  def __init__(
    self,
    policy: torch.nn.Module,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    kl_weight: float = 1.0e-4,
    kl_warmup_iterations: int = 2_000,
    free_nats_per_dim: float = 0.02,
    latent_regularization: str = "kl",
    mmd_weight: float = 0.0,
    mmd_kernel_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    mmd_max_samples: int = 1024,
    latent_smooth_weight: float = 1.0e-3,
    latent_smooth_max_pairs: int = 2048,
    multi_gpu_cfg: dict | None = None,
  ):
    self.policy = policy
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.kl_weight = float(kl_weight)
    self.kl_warmup_iterations = int(kl_warmup_iterations)
    self.free_nats_per_dim = float(free_nats_per_dim)
    if latent_regularization not in {"kl", "wae_mmd"}:
      raise ValueError(
        "latent_regularization must be one of {'kl', 'wae_mmd'}, "
        f"got {latent_regularization!r}"
      )
    self.latent_regularization = latent_regularization
    self.mmd_weight = float(mmd_weight)
    self.mmd_kernel_scales = tuple(float(scale) for scale in mmd_kernel_scales)
    self.mmd_max_samples = int(mmd_max_samples)
    self.latent_smooth_weight = float(latent_smooth_weight)
    self.latent_smooth_max_pairs = int(latent_smooth_max_pairs)
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
    self.is_multi_gpu = multi_gpu_cfg is not None
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = int(multi_gpu_cfg["global_rank"])
      self.gpu_world_size = int(multi_gpu_cfg["world_size"])
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

  def broadcast_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    model_params = [self.policy.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)
    self.policy.load_state_dict(model_params[0])

  def reduce_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    for param in self.policy.parameters():
      if param.grad is None:
        continue
      torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
      param.grad.div_(self.gpu_world_size)

  def update(
    self,
    obs: TensorDict,
    teacher_actions: torch.Tensor,
    num_learning_epochs: int,
    num_mini_batches: int,
    iteration: int = 0,
    rollout_shape: tuple[int, int] | None = None,
    dones: torch.Tensor | None = None,
  ) -> dict[str, float]:
    batch_size = teacher_actions.shape[0]
    if batch_size == 0:
      raise ValueError("teacher_actions must contain at least one sample")

    num_mini_batches = max(1, min(num_mini_batches, batch_size))
    mini_batch_size = math.ceil(batch_size / num_mini_batches)
    effective_kl_weight = self._effective_kl_weight(iteration)
    effective_mmd_weight = self.mmd_weight if self.latent_regularization == "wae_mmd" else 0.0

    totals = {
      "action_mse": 0.0,
      "action_l1": 0.0,
      "kl_loss": 0.0,
      "kl_per_dim": 0.0,
      "mmd_loss": 0.0,
      "aggregate_mean_norm": 0.0,
      "aggregate_std_mean": 0.0,
      "latent_mu_norm": 0.0,
      "latent_std_mean": 0.0,
      "latent_smooth_loss": 0.0,
      "total_loss": 0.0,
      "grad_norm": 0.0,
    }
    updates = 0

    self.policy.train()
    for _ in range(num_learning_epochs):
      permutation = torch.randperm(batch_size, device=teacher_actions.device)
      for start in range(0, batch_size, mini_batch_size):
        batch_idx = permutation[start : start + mini_batch_size]
        batch_obs = obs[batch_idx]
        batch_teacher = teacher_actions[batch_idx]

        if hasattr(self.policy, "update_normalization"):
          self.policy.update_normalization(batch_obs)

        pred_actions, latent = self.policy(batch_obs, deterministic=False)
        mse_loss = F.mse_loss(pred_actions, batch_teacher)
        l1_loss = F.l1_loss(pred_actions, batch_teacher)
        raw_kl = self._standard_normal_kl(latent["mu"], latent["log_std"])
        kl_loss = torch.clamp(raw_kl, min=self.free_nats_per_dim).sum(dim=-1).mean()
        if effective_mmd_weight > 0.0:
          mmd_z = self._subsample_rows(latent["z"], self.mmd_max_samples)
          prior_z = torch.randn_like(mmd_z)
          mmd_loss = self._mmd_rbf(mmd_z, prior_z, self.mmd_kernel_scales)
        else:
          mmd_loss = latent["z"].new_zeros(())
        smooth_loss = self._sample_trajectory_latent_smoothness(
          obs=obs,
          rollout_shape=rollout_shape,
          dones=dones,
          batch_size=mini_batch_size,
        )
        total_loss = (
          mse_loss
          + effective_kl_weight * kl_loss
          + effective_mmd_weight * mmd_loss
          + self.latent_smooth_weight * smooth_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.reduce_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(
          self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        totals["action_mse"] += float(mse_loss.item())
        totals["action_l1"] += float(l1_loss.item())
        totals["kl_loss"] += float(kl_loss.item())
        totals["kl_per_dim"] += float(raw_kl.mean().item())
        totals["mmd_loss"] += float(mmd_loss.item())
        totals["aggregate_mean_norm"] += float(latent["z"].mean(dim=0).norm().item())
        totals["aggregate_std_mean"] += float(latent["z"].std(dim=0, unbiased=False).mean().item())
        totals["latent_mu_norm"] += float(latent["mu"].norm(dim=-1).mean().item())
        totals["latent_std_mean"] += float(torch.exp(latent["log_std"]).mean().item())
        totals["latent_smooth_loss"] += float(smooth_loss.item())
        totals["total_loss"] += float(total_loss.item())
        totals["grad_norm"] += float(grad_norm.item())
        updates += 1

    metrics = {key: value / updates for key, value in totals.items()}
    metrics["kl_weight"] = effective_kl_weight
    metrics["mmd_weight"] = effective_mmd_weight
    return metrics

  def save(self) -> dict:
    saved = {
      "model_type": "latent",
      "policy_state_dict": self.policy.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "learning_rate": self.learning_rate,
      "regularization_cfg": {
        "latent_regularization": self.latent_regularization,
        "kl_weight": self.kl_weight,
        "kl_warmup_iterations": self.kl_warmup_iterations,
        "free_nats_per_dim": self.free_nats_per_dim,
        "mmd_weight": self.mmd_weight,
        "mmd_kernel_scales": self.mmd_kernel_scales,
        "mmd_max_samples": self.mmd_max_samples,
        "latent_smooth_weight": self.latent_smooth_weight,
        "latent_smooth_max_pairs": self.latent_smooth_max_pairs,
      },
      "latent_cfg": self.policy.latent_cfg()
      if hasattr(self.policy, "latent_cfg")
      else {},
    }
    if hasattr(self.policy, "encoder"):
      saved["encoder_state_dict"] = self.policy.encoder.state_dict()
    if hasattr(self.policy, "decoder"):
      saved["decoder_state_dict"] = self.policy.decoder.state_dict()
    return saved

  def load(self, checkpoint: dict) -> None:
    self.policy.load_state_dict(checkpoint["policy_state_dict"])
    if "optimizer_state_dict" in checkpoint:
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

  def _effective_kl_weight(self, iteration: int) -> float:
    if self.latent_regularization != "kl":
      return 0.0
    if self.kl_warmup_iterations <= 0:
      return self.kl_weight
    warmup = min(max(float(iteration) / float(self.kl_warmup_iterations), 0.0), 1.0)
    return self.kl_weight * warmup

  @staticmethod
  def _standard_normal_kl(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    log_var = 2.0 * log_std
    return 0.5 * (mu.square() + torch.exp(log_var) - 1.0 - log_var)

  def _sample_trajectory_latent_smoothness(
    self,
    obs: TensorDict,
    rollout_shape: tuple[int, int] | None,
    dones: torch.Tensor | None,
    batch_size: int,
  ) -> torch.Tensor:
    sample_tensor = next(self.policy.parameters())
    if self.latent_smooth_weight <= 0.0 or rollout_shape is None:
      return sample_tensor.new_zeros(())

    pair_idx, next_pair_idx = self._valid_temporal_pair_indices(
      rollout_shape=rollout_shape,
      dones=dones,
      device=sample_tensor.device,
    )
    if pair_idx.numel() == 0:
      return sample_tensor.new_zeros(())

    sample_size = min(
      int(batch_size),
      int(pair_idx.numel()),
      max(self.latent_smooth_max_pairs, 0),
    )
    if sample_size <= 0:
      return sample_tensor.new_zeros(())
    sampled = torch.randint(pair_idx.numel(), (sample_size,), device=pair_idx.device)
    current_obs = obs[pair_idx[sampled]]
    next_obs = obs[next_pair_idx[sampled]]
    current_mu, _ = self.policy.encode(current_obs)
    next_mu, _ = self.policy.encode(next_obs)
    return (next_mu - current_mu).square().mean()

  @staticmethod
  def _trajectory_latent_smoothness(
    mu: torch.Tensor,
    rollout_shape: tuple[int, int],
    dones: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_steps, num_envs = rollout_shape
    if mu.shape[0] != num_steps * num_envs:
      raise ValueError(
        f"Expected {num_steps * num_envs} latent rows for rollout_shape={rollout_shape}, "
        f"got {mu.shape[0]}"
      )
    if num_steps < 2:
      return mu.new_zeros(())

    mu_by_time = mu.reshape(num_steps, num_envs, -1)
    diff = mu_by_time[1:] - mu_by_time[:-1]
    if dones is None:
      valid = torch.ones(num_steps - 1, num_envs, dtype=torch.bool, device=mu.device)
    else:
      dones = dones.reshape(num_steps, num_envs).to(device=mu.device, dtype=torch.bool)
      valid = ~dones[:-1]
    if not torch.any(valid):
      return mu.new_zeros(())
    return diff[valid].square().mean()

  @staticmethod
  def _valid_temporal_pair_indices(
    rollout_shape: tuple[int, int],
    dones: torch.Tensor | None,
    device: torch.device,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    num_steps, num_envs = rollout_shape
    if num_steps < 2:
      empty = torch.empty(0, dtype=torch.long, device=device)
      return empty, empty

    starts = torch.arange((num_steps - 1) * num_envs, device=device, dtype=torch.long)
    next_starts = starts + num_envs
    if dones is None:
      return starts, next_starts

    dones = dones.reshape(num_steps, num_envs).to(device=device, dtype=torch.bool)
    valid = (~dones[:-1]).reshape(-1)
    return starts[valid], next_starts[valid]

  @staticmethod
  def _mmd_rbf(
    samples: torch.Tensor,
    prior_samples: torch.Tensor,
    kernel_scales: tuple[float, ...],
  ) -> torch.Tensor:
    if samples.shape != prior_samples.shape:
      raise ValueError(
        f"MMD sample shapes must match, got {samples.shape} and {prior_samples.shape}"
      )
    sq_xx = torch.cdist(samples, samples).square()
    sq_yy = torch.cdist(prior_samples, prior_samples).square()
    sq_xy = torch.cdist(samples, prior_samples).square()
    latent_dim = max(samples.shape[-1], 1)

    mmd = samples.new_zeros(())
    for scale in kernel_scales:
      bandwidth = max(float(scale), 1.0e-6) ** 2 * latent_dim
      gamma = 0.5 / bandwidth
      k_xx = torch.exp(-gamma * sq_xx).mean()
      k_yy = torch.exp(-gamma * sq_yy).mean()
      k_xy = torch.exp(-gamma * sq_xy).mean()
      mmd = mmd + k_xx + k_yy - 2.0 * k_xy
    return torch.clamp(mmd / max(len(kernel_scales), 1), min=0.0)

  @staticmethod
  def _subsample_rows(samples: torch.Tensor, max_samples: int) -> torch.Tensor:
    max_samples = int(max_samples)
    if max_samples <= 0 or samples.shape[0] <= max_samples:
      return samples
    sample_idx = torch.randperm(samples.shape[0], device=samples.device)[:max_samples]
    return samples[sample_idx]
