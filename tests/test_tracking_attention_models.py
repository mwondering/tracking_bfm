from __future__ import annotations

import pytest
import torch
from rsl_rl.storage import RolloutStorage
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
  SparseTrackFullRefAttentionActor,
  SparseTrackFullRefAttentionCritic,
)
from mjlab.tasks.tracking.rl.ppo import SparseTrackSplitLrPPO

AttentionActorClass = type[
  FullObsCausalAttentionActor
  | HistProprioCrossAttentionActor
  | ProprioRefCrossAttentionActor
  | SparseTrackFullRefAttentionActor
]

ATTENTION_ACTORS: tuple[tuple[AttentionVariant, AttentionActorClass], ...] = (
  ("full_obs_causal", FullObsCausalAttentionActor),
  ("proprio_ref_cross", ProprioRefCrossAttentionActor),
  ("hist_proprio_cross", HistProprioCrossAttentionActor),
  ("sparsetrack_full_ref", SparseTrackFullRefAttentionActor),
)

PARAMETER_BUDGETS: dict[AttentionVariant, tuple[int, int]] = {
  "full_obs_causal": (8_000_000, 10_500_000),
  "proprio_ref_cross": (8_000_000, 10_500_000),
  "hist_proprio_cross": (8_000_000, 10_500_000),
  "sparsetrack_full_ref": (2_900_000, 3_200_000),
}


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
  | SparseTrackFullRefAttentionActor
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


def _make_sparsetrack_critic() -> SparseTrackFullRefAttentionCritic:
  cfg = tracking_attention_actor_cfg("sparsetrack_full_ref")
  cfg_dict = cfg.__dict__.copy()
  cfg_dict.pop("class_name")
  cfg_dict["distribution_cfg"] = None
  return SparseTrackFullRefAttentionCritic(
    TensorDict(
      {"critic": torch.randn(4, ACTOR_HISTORY_LENGTH * FRAME_DIM)},
      batch_size=[4],
    ),
    {"actor": ["actor"], "critic": ["critic"]},
    "critic",
    1,
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
  low, high = PARAMETER_BUDGETS[variant]

  assert low <= num_params <= high, (variant, num_params)


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


def test_sparsetrack_full_ref_attention_zero_initial_action_mean() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  obs = TensorDict(
    {"actor": torch.zeros(3, ACTOR_HISTORY_LENGTH * FRAME_DIM)},
    batch_size=[3],
  )

  with torch.no_grad():
    action_mean = actor(obs, stochastic_output=False)

  assert torch.allclose(action_mean, torch.zeros_like(action_mean), atol=1e-6)


def test_sparsetrack_full_ref_attention_task_tokens_affect_latent() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  actor.eval()

  term_histories = {
    name: torch.zeros(1, ACTOR_HISTORY_LENGTH, dim) for name, dim in TERM_DIMS.items()
  }
  changed_histories = {name: value.clone() for name, value in term_histories.items()}
  changed_histories["command"][:, -1, 0] = 1.0
  changed_histories["motion_anchor_pos_b"][:, -1, 1] = -0.5

  flat_obs = _flat_term_major_obs(term_histories)
  changed_flat_obs = _flat_term_major_obs(changed_histories)

  with torch.no_grad():
    latent = actor._attention_latent_from_flat(flat_obs)
    changed_latent = actor._attention_latent_from_flat(changed_flat_obs)

  assert not torch.allclose(latent, changed_latent)


def test_sparsetrack_full_ref_attention_uses_task_embedder_module() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)

  assert hasattr(actor, "task_embedder")
  assert not hasattr(actor, "task_projection")
  task_tokens = actor.task_embedder(
    torch.zeros(2, ACTOR_HISTORY_LENGTH, actor.task_obs_dim)
  )

  assert task_tokens.shape == (2, ACTOR_HISTORY_LENGTH, actor.d_model)


def test_sparsetrack_task_embedder_linear_init_matches_reference_scale() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  task_projection = actor.task_embedder.task_projection

  assert isinstance(task_projection, torch.nn.Linear)
  expected_std = 1.0 / (actor.task_obs_dim**0.5)
  actual_std = task_projection.weight.std().item()

  assert actual_std == pytest.approx(expected_std, rel=0.2)
  assert torch.allclose(task_projection.bias, torch.zeros_like(task_projection.bias))


def test_sparsetrack_full_ref_attention_initializes_residual_projections_smaller() -> (
  None
):
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  block = actor.transformer_blocks[0]

  in_proj_std = block.self_attention.in_proj_weight.std().item()
  out_proj_std = block.self_attention.out_proj.weight.std().item()

  assert out_proj_std < in_proj_std * 0.5


def test_sparsetrack_full_ref_attention_uses_linear_projection_head() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  linear_layers = [
    module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
  ]

  assert len(linear_layers) == 1
  assert linear_layers[0].in_features == actor.d_model
  assert linear_layers[0].out_features == NUM_DOFS


def test_sparsetrack_full_ref_attention_critic_forward_shape() -> None:
  critic = _make_sparsetrack_critic()
  obs = TensorDict(
    {"critic": torch.randn(3, ACTOR_HISTORY_LENGTH * FRAME_DIM)},
    batch_size=[3],
  )

  values = critic(obs)

  assert values.shape == (3, 1)


def test_sparsetrack_split_lr_ppo_uses_actor_critic_param_groups() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  critic = _make_sparsetrack_critic()
  obs = TensorDict(
    {
      "actor": torch.randn(4, ACTOR_HISTORY_LENGTH * FRAME_DIM),
      "critic": torch.randn(4, ACTOR_HISTORY_LENGTH * FRAME_DIM),
    },
    batch_size=[4],
  )
  storage = RolloutStorage("rl", 4, 2, obs, [NUM_DOFS], "cpu")

  alg = SparseTrackSplitLrPPO(
    actor,
    critic,
    storage,
    actor_learning_rate=2.0e-5,
    critic_learning_rate=1.0e-3,
  )

  assert [group["lr"] for group in alg.optimizer.param_groups] == [2.0e-5, 1.0e-3]
  assert alg.learning_rate == 2.0e-5


@pytest.mark.parametrize(
  ("variant", "cls"),
  (
    ("full_obs_causal", FullObsCausalAttentionActor),
    ("hist_proprio_cross", HistProprioCrossAttentionActor),
  ),
)
def test_attention_branch_receives_initial_gradient(
  variant: AttentionVariant,
  cls: AttentionActorClass,
) -> None:
  actor = _make_actor(variant, cls)
  obs = _dummy_obs(batch_size=4)

  actor(obs).pow(2).mean().backward()

  grad = actor.history_encoder.layers[0].self_attn.in_proj_weight.grad

  assert grad is not None
  assert torch.count_nonzero(grad) > 0
