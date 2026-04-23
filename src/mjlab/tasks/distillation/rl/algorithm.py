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
  ):
    self.policy = policy
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

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
