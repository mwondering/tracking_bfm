from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.tracking.config.g1.attention_cfg import (
  ACTOR_HISTORY_LENGTH,
  FRAME_DIM,
  NUM_DOFS,
  AttentionVariant,
  tracking_attention_actor_cfg,
)
from mjlab.tasks.tracking.rl.attention_models import (
  PROPRIO_TERMS,
  TERM_DIMS,
  FullObsCausalAttentionActor,
  HistProprioCrossAttentionActor,
  ProprioRefCrossAttentionActor,
)

AttentionActorClass = type[
  FullObsCausalAttentionActor
  | HistProprioCrossAttentionActor
  | ProprioRefCrossAttentionActor
]

ATTENTION_ACTORS: tuple[tuple[AttentionVariant, AttentionActorClass], ...] = (
  ("full_obs_causal", FullObsCausalAttentionActor),
  ("proprio_ref_cross", ProprioRefCrossAttentionActor),
  ("hist_proprio_cross", HistProprioCrossAttentionActor),
)


def _dummy_obs(batch_size: int = 4, obs_dim: int | None = None) -> TensorDict:
  obs_dim = obs_dim or ACTOR_HISTORY_LENGTH * FRAME_DIM
  return TensorDict(
    {"actor": torch.randn(batch_size, obs_dim)},
    batch_size=[batch_size],
  )


def _flat_term_major_obs(term_histories: dict[str, torch.Tensor]) -> torch.Tensor:
  return torch.cat(
    [
      term_histories[name].reshape(term_histories[name].shape[0], -1)
      for name in TERM_DIMS
    ],
    dim=-1,
  )


def _make_actor(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> (
  FullObsCausalAttentionActor
  | HistProprioCrossAttentionActor
  | ProprioRefCrossAttentionActor
):
  cfg = tracking_attention_actor_cfg(variant)
  cfg_dict = cfg.__dict__.copy()
  cfg_dict.pop("class_name")
  return cls(
    _dummy_obs(),
    {"actor": ["actor"], "critic": ["critic"]},
    "actor",
    NUM_DOFS,
    **cfg_dict,
  )


@pytest.mark.parametrize(("variant", "cls"), ATTENTION_ACTORS)
def test_tracking_attention_actor_forward_shape(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> None:
  actor = _make_actor(variant, cls)
  obs = _dummy_obs(batch_size=3)

  actions = actor(obs)

  assert actions.shape == (3, NUM_DOFS)


@pytest.mark.parametrize(("variant", "cls"), ATTENTION_ACTORS)
def test_tracking_attention_actor_stochastic_distribution_api(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> None:
  actor = _make_actor(variant, cls)
  obs = _dummy_obs(batch_size=3)

  sampled_actions = actor(obs, stochastic_output=True)
  log_prob = actor.get_output_log_prob(sampled_actions)

  assert sampled_actions.shape == (3, NUM_DOFS)
  assert log_prob.shape == (3,)
  assert actor.output_mean.shape == (3, NUM_DOFS)
  assert actor.output_std.shape[-1] == NUM_DOFS
  assert actor.output_entropy.shape == (3,)


@pytest.mark.parametrize(("variant", "cls"), ATTENTION_ACTORS)
def test_tracking_attention_actor_parameter_budget(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> None:
  actor = _make_actor(variant, cls)

  num_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)

  assert 8_000_000 <= num_params <= 10_500_000, (variant, num_params)


@pytest.mark.parametrize(("variant", "cls"), ATTENTION_ACTORS)
def test_tracking_attention_actor_rejects_wrong_flat_observation_dim(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> None:
  cfg = tracking_attention_actor_cfg(variant)
  cfg_dict = cfg.__dict__.copy()
  cfg_dict.pop("class_name")

  with pytest.raises(ValueError, match="expected flat observation dim"):
    cls(
      _dummy_obs(obs_dim=123),
      {"actor": ["actor"], "critic": ["critic"]},
      "actor",
      NUM_DOFS,
      **cfg_dict,
    )


@pytest.mark.parametrize(("variant", "cls"), ATTENTION_ACTORS)
def test_tracking_attention_actor_onnx_wrapper_shape(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> None:
  actor = _make_actor(variant, cls)
  wrapper = actor.as_onnx(verbose=False)

  out = wrapper(torch.randn(2, ACTOR_HISTORY_LENGTH * FRAME_DIM))

  assert out.shape == (2, NUM_DOFS)


def test_proprio_ref_cross_attention_preserves_history_order() -> None:
  actor = _make_actor("proprio_ref_cross", ProprioRefCrossAttentionActor)
  actor.eval()

  term_histories = {
    name: torch.zeros(1, ACTOR_HISTORY_LENGTH, dim) for name, dim in TERM_DIMS.items()
  }
  for name in PROPRIO_TERMS:
    dim = TERM_DIMS[name]
    term_histories[name] = torch.arange(
      ACTOR_HISTORY_LENGTH * dim,
      dtype=torch.float32,
    ).reshape(1, ACTOR_HISTORY_LENGTH, dim)

  reversed_histories = {
    name: history.flip(dims=(1,)) if name in PROPRIO_TERMS else history
    for name, history in term_histories.items()
  }

  forward_obs = TensorDict(
    {"actor": _flat_term_major_obs(term_histories)},
    batch_size=[1],
  )
  reversed_obs = TensorDict(
    {"actor": _flat_term_major_obs(reversed_histories)},
    batch_size=[1],
  )

  with torch.no_grad():
    forward_action = actor(forward_obs)
    reversed_action = actor(reversed_obs)

  assert not torch.allclose(forward_action, reversed_action)


def test_hist_proprio_cross_attention_features_are_zero_impact_at_init() -> None:
  actor = _make_actor("hist_proprio_cross", HistProprioCrossAttentionActor)
  first_layer = actor.mlp[0]
  assert isinstance(first_layer, torch.nn.Linear)

  attention_feature_weights = first_layer.weight[:, FRAME_DIM:]

  assert torch.count_nonzero(attention_feature_weights) == 0


def test_full_obs_causal_uses_current_frame_residual_at_init() -> None:
  actor = _make_actor("full_obs_causal", FullObsCausalAttentionActor)
  first_layer = actor.mlp[0]
  assert isinstance(first_layer, torch.nn.Linear)

  transformer_feature_weights = first_layer.weight[:, FRAME_DIM:]

  assert first_layer.in_features == FRAME_DIM + actor.d_model
  assert torch.count_nonzero(transformer_feature_weights) == 0
