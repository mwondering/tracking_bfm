"""Tracking-only attention actor configuration for G1 BFM ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg

AttentionVariant = Literal[
  "full_obs_causal",
  "proprio_ref_cross",
  "hist_proprio_cross",
]

FRAME_DIM = 286
COMMAND_DIM = 58
NUM_DOFS = 29
ACTOR_HISTORY_LENGTH = 11

FULL_OBS_CAUSAL_CLASS = (
  "mjlab.tasks.tracking.rl.attention_models:FullObsCausalAttentionActor"
)
PROPRIO_REF_CROSS_CLASS = (
  "mjlab.tasks.tracking.rl.attention_models:ProprioRefCrossAttentionActor"
)
HIST_PROPRIO_CROSS_CLASS = (
  "mjlab.tasks.tracking.rl.attention_models:HistProprioCrossAttentionActor"
)


@dataclass
class TrackingAttentionModelCfg(RslRlModelCfg):
  """RSL-RL model config for tracking-specific attention actors."""

  history_length: int = ACTOR_HISTORY_LENGTH
  frame_dim: int = FRAME_DIM
  command_dim: int = COMMAND_DIM
  num_dofs: int = NUM_DOFS
  d_model: int = 384
  num_heads: int = 6
  ffn_dim: int = 1536
  history_layers: int = 0
  cross_layers: int = 0
  head_hidden_dims: tuple[int, ...] = field(
    default_factory=lambda: (1536, 1024, 512, 256)
  )
  dropout: float = 0.0
  attention_activation: str = "gelu"


def tracking_attention_actor_cfg(
  variant: AttentionVariant,
) -> TrackingAttentionModelCfg:
  """Build an actor config for one tracking attention ablation."""
  if variant == "full_obs_causal":
    return _tracking_attention_actor_cfg(FULL_OBS_CAUSAL_CLASS, 3, 0)
  if variant == "proprio_ref_cross":
    return _tracking_attention_actor_cfg(PROPRIO_REF_CROSS_CLASS, 0, 3)
  if variant == "hist_proprio_cross":
    return _tracking_attention_actor_cfg(HIST_PROPRIO_CROSS_CLASS, 2, 1)
  raise ValueError(f"Unknown attention variant: {variant}")


def _tracking_attention_actor_cfg(
  class_name: str,
  history_layers: int,
  cross_layers: int,
) -> TrackingAttentionModelCfg:
  return TrackingAttentionModelCfg(
    class_name=class_name,
    hidden_dims=(1536, 1024, 512, 256),
    head_hidden_dims=(1536, 1024, 512, 256),
    activation="gelu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    history_layers=history_layers,
    cross_layers=cross_layers,
  )
