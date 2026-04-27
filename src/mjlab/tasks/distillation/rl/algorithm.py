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
