"""Student model builders for distillation."""

from __future__ import annotations

from rsl_rl.models import MLPModel
from tensordict import TensorDict


def build_student_model(
  obs: TensorDict,
  student_obs_group: str,
  action_dim: int,
  hidden_dims: tuple[int, ...],
  activation: str,
  obs_normalization: bool = True,
) -> MLPModel:
  """Build the default MLP student policy."""
  return MLPModel(
    obs=obs,
    obs_groups={"actor": [student_obs_group]},
    obs_set="actor",
    output_dim=action_dim,
    hidden_dims=hidden_dims,
    activation=activation,
    obs_normalization=obs_normalization,
  )
