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
    latent_smooth_weight: float = 1.0e-3,
    multi_gpu_cfg: dict | None = None,
  ):
    self.policy = policy
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.kl_weight = float(kl_weight)
    self.kl_warmup_iterations = int(kl_warmup_iterations)
    self.free_nats_per_dim = float(free_nats_per_dim)
    self.latent_smooth_weight = float(latent_smooth_weight)
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
  ) -> dict[str, float]:
    batch_size = teacher_actions.shape[0]
    if batch_size == 0:
      raise ValueError("teacher_actions must contain at least one sample")

    num_mini_batches = max(1, min(num_mini_batches, batch_size))
    mini_batch_size = math.ceil(batch_size / num_mini_batches)
    effective_kl_weight = self._effective_kl_weight(iteration)

    totals = {
      "action_mse": 0.0,
      "action_l1": 0.0,
      "kl_loss": 0.0,
      "kl_per_dim": 0.0,
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
        smooth_loss = self._latent_smoothness(latent["mu"])
        total_loss = (
          mse_loss
          + effective_kl_weight * kl_loss
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
        totals["latent_mu_norm"] += float(latent["mu"].norm(dim=-1).mean().item())
        totals["latent_std_mean"] += float(torch.exp(latent["log_std"]).mean().item())
        totals["latent_smooth_loss"] += float(smooth_loss.item())
        totals["total_loss"] += float(total_loss.item())
        totals["grad_norm"] += float(grad_norm.item())
        updates += 1

    metrics = {key: value / updates for key, value in totals.items()}
    metrics["kl_weight"] = effective_kl_weight
    return metrics

  def save(self) -> dict:
    saved = {
      "model_type": "latent",
      "policy_state_dict": self.policy.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "learning_rate": self.learning_rate,
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
    if self.kl_warmup_iterations <= 0:
      return self.kl_weight
    warmup = min(max(float(iteration) / float(self.kl_warmup_iterations), 0.0), 1.0)
    return self.kl_weight * warmup

  @staticmethod
  def _standard_normal_kl(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    log_var = 2.0 * log_std
    return 0.5 * (mu.square() + torch.exp(log_var) - 1.0 - log_var)

  @staticmethod
  def _latent_smoothness(mu: torch.Tensor) -> torch.Tensor:
    if mu.shape[0] < 2:
      return mu.new_zeros(())
    return (mu[1:] - mu[:-1]).square().mean()
