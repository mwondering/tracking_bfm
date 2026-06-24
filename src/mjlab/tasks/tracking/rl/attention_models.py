"""Attention actor models for G1 BFM tracking ablations."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import ClassVar, cast

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

TERM_DIMS: OrderedDict[str, int] = OrderedDict(
  (
    ("command", 58),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("body_pos", 42),
    ("body_ori", 84),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("actions", 29),
  )
)
PROPRIO_TERMS = ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions")


def _make_causal_mask(length: int) -> torch.Tensor:
  return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)


def _make_encoder_layer(
  d_model: int,
  num_heads: int,
  ffn_dim: int,
  dropout: float,
  activation: str,
) -> nn.TransformerEncoderLayer:
  return nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=num_heads,
    dim_feedforward=ffn_dim,
    dropout=dropout,
    activation=activation,
    batch_first=True,
    norm_first=True,
  )


class _CrossAttentionBlock(nn.Module):
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    activation: str,
  ) -> None:
    super().__init__()
    self.q_norm = nn.LayerNorm(d_model)
    self.kv_norm = nn.LayerNorm(d_model)
    self.attn = nn.MultiheadAttention(
      d_model,
      num_heads,
      dropout=dropout,
      batch_first=True,
    )
    self.ffn_norm = nn.LayerNorm(d_model)
    activation_mod: nn.Module
    if activation == "gelu":
      activation_mod = nn.GELU()
    elif activation == "elu":
      activation_mod = nn.ELU()
    else:
      raise ValueError(f"Unsupported cross-attention activation: {activation}")
    self.ffn = nn.Sequential(
      nn.Linear(d_model, ffn_dim),
      activation_mod,
      nn.Dropout(dropout),
      nn.Linear(ffn_dim, d_model),
      nn.Dropout(dropout),
    )

  def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
    norm_kv = self.kv_norm(key_value)
    attn_out, _ = self.attn(
      self.q_norm(query),
      norm_kv,
      norm_kv,
      need_weights=False,
    )
    query = query + attn_out
    return query + self.ffn(self.ffn_norm(query))


class _BaseTrackingAttentionActor(MLPModel):
  """RSL-RL compatible base class for flattened-history tracking actors."""

  is_recurrent: ClassVar[bool] = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (1536, 1024, 512, 256),
    activation: str = "gelu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict | None = None,
    rnn_type: str | None = None,
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    history_length: int = 11,
    frame_dim: int = 286,
    command_dim: int = 58,
    num_dofs: int = 29,
    d_model: int = 384,
    num_heads: int = 6,
    ffn_dim: int = 1536,
    history_layers: int = 0,
    cross_layers: int = 0,
    head_hidden_dims: tuple[int, ...] | list[int] = (1536, 1024, 512, 256),
    dropout: float = 0.0,
    attention_activation: str = "gelu",
  ) -> None:
    self.history_length = int(history_length)
    self.frame_dim = int(frame_dim)
    self.command_dim = int(command_dim)
    self.num_dofs = int(num_dofs)
    self.d_model = int(d_model)
    self.num_heads = int(num_heads)
    self.ffn_dim = int(ffn_dim)
    self.history_layers = int(history_layers)
    self.cross_layers = int(cross_layers)
    self.dropout = float(dropout)
    self.attention_activation = attention_activation
    self._latent_dim = self._attention_latent_dim()
    if cnn_cfg is not None:
      raise ValueError("tracking attention actors do not support cnn_cfg")
    if rnn_type is not None:
      raise ValueError("tracking attention actors do not support rnn_type")
    _ = (rnn_hidden_dim, rnn_num_layers)

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=tuple(head_hidden_dims or hidden_dims),
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self._validate_config(output_dim)

  def _attention_latent_dim(self) -> int:
    raise NotImplementedError

  def _build_history_encoder(self, num_layers: int) -> nn.TransformerEncoder:
    encoder_layer = _make_encoder_layer(
      self.d_model,
      self.num_heads,
      self.ffn_dim,
      self.dropout,
      self.attention_activation,
    )
    return nn.TransformerEncoder(
      encoder_layer,
      num_layers=num_layers,
      norm=nn.LayerNorm(self.d_model),
      enable_nested_tensor=False,
    )

  def _build_cross_blocks(self, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList(
      [
        _CrossAttentionBlock(
          self.d_model,
          self.num_heads,
          self.ffn_dim,
          self.dropout,
          self.attention_activation,
        )
        for _ in range(num_layers)
      ]
    )

  def _validate_config(self, output_dim: int) -> None:
    expected_flat_dim = self.history_length * self.frame_dim
    if self.obs_dim != expected_flat_dim:
      raise ValueError(
        f"expected flat observation dim {expected_flat_dim}, got {self.obs_dim}"
      )
    expected_frame_dim = sum(TERM_DIMS.values())
    if self.frame_dim != expected_frame_dim:
      raise ValueError(f"expected frame_dim {expected_frame_dim}, got {self.frame_dim}")
    if self.command_dim != 2 * self.num_dofs:
      raise ValueError(
        f"expected command_dim {2 * self.num_dofs}, got {self.command_dim}"
      )
    if output_dim != self.num_dofs:
      raise ValueError(f"expected output_dim {self.num_dofs}, got {output_dim}")
    if self.d_model % self.num_heads != 0:
      raise ValueError("d_model must be divisible by num_heads")

  def _get_latent_dim(self) -> int:
    return self._latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    obs_list = [cast(torch.Tensor, obs[obs_group]) for obs_group in self.obs_groups]
    flat_obs = torch.cat(obs_list, dim=-1)
    flat_obs = self.obs_normalizer(flat_obs)
    return self._attention_latent_from_flat(flat_obs)

  def _term_history(self, flat_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    terms: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, dim in TERM_DIMS.items():
      next_cursor = cursor + self.history_length * dim
      terms[name] = flat_obs[:, cursor:next_cursor].reshape(
        flat_obs.shape[0],
        self.history_length,
        dim,
      )
      cursor = next_cursor
    return terms

  def _frame_history(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    return self._frame_history_from_terms(terms)

  def _proprio_history(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    return self._proprio_history_from_terms(terms)

  def _current_command_tokens(self, flat_obs: torch.Tensor) -> torch.Tensor:
    command = self._term_history(flat_obs)["command"][:, -1]
    return self._command_tokens(command)

  def _frame_history_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    return torch.cat([terms[name] for name in TERM_DIMS], dim=-1)

  def _proprio_history_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    return torch.cat([terms[name] for name in PROPRIO_TERMS], dim=-1)

  def _current_command_tokens_from_terms(
    self,
    terms: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    command = terms["command"][:, -1]
    return self._command_tokens(command)

  def _command_tokens(self, command: torch.Tensor) -> torch.Tensor:
    q_ref = command[:, : self.num_dofs]
    qd_ref = command[:, self.num_dofs :]
    return torch.stack((q_ref, qd_ref), dim=-1)

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxTrackingAttentionActor(self, verbose)


class _OnnxTrackingAttentionActor(nn.Module):
  """ONNX wrapper that preserves custom attention preprocessing."""

  is_recurrent: bool = False

  def __init__(self, model: _BaseTrackingAttentionActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.model = copy.deepcopy(model)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.model.obs_normalizer(x)
    latent = self.model._attention_latent_from_flat(x)
    out = self.model.mlp(latent)
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class FullObsCausalAttentionActor(_BaseTrackingAttentionActor):
  """Humanoid-GPT-style causal attention over full observation frames."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.frame_proj = nn.Linear(self.frame_dim, self.d_model)
    self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_length, self.d_model))
    self.history_encoder = self._build_history_encoder(self.history_layers)
    self.register_buffer(
      "_causal_mask",
      _make_causal_mask(self.history_length),
      persistent=False,
    )

  def _attention_latent_dim(self) -> int:
    return self.d_model

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    frames = self._frame_history_from_terms(terms)
    tokens = self.frame_proj(frames) + self.pos_embedding
    encoded = self.history_encoder(tokens, mask=self._causal_mask)
    return encoded[:, -1]


class ProprioRefCrossAttentionActor(_BaseTrackingAttentionActor):
  """Cross-attention from proprioceptive history summary to current ref tokens."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.proprio_proj = nn.Linear(93, self.d_model)
    self.proprio_pos_embedding = nn.Parameter(
      torch.zeros(1, self.history_length, self.d_model)
    )
    self.command_token_proj = nn.Linear(2, self.d_model)
    self.joint_embedding = nn.Parameter(torch.zeros(1, self.num_dofs, self.d_model))
    self.history_pool = nn.Linear(self.history_length * self.d_model, self.d_model)
    self._init_current_token_history_pool()
    self.query_norm = nn.LayerNorm(self.d_model)
    self.cross_blocks = self._build_cross_blocks(self.cross_layers)

  def _attention_latent_dim(self) -> int:
    return self.d_model

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    proprio = self._proprio_history_from_terms(terms)
    proprio_tokens = self.proprio_proj(proprio) + self.proprio_pos_embedding
    query = self.history_pool(proprio_tokens.flatten(start_dim=1)).unsqueeze(1)
    query = self.query_norm(query)

    ref_tokens = self.command_token_proj(self._current_command_tokens_from_terms(terms))
    ref_tokens = ref_tokens + self.joint_embedding

    for block in self.cross_blocks:
      query = block(query, ref_tokens)
    return query.squeeze(1)

  def _init_current_token_history_pool(self) -> None:
    nn.init.zeros_(self.history_pool.weight)
    nn.init.zeros_(self.history_pool.bias)
    current_offset = (self.history_length - 1) * self.d_model
    with torch.no_grad():
      for dim_idx in range(self.d_model):
        self.history_pool.weight[dim_idx, current_offset + dim_idx] = 1.0


class HistProprioCrossAttentionActor(_BaseTrackingAttentionActor):
  """RoHM-style causal proprio history encoder plus command cross-attention."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.proprio_proj = nn.Linear(93, self.d_model)
    self.proprio_pos_embedding = nn.Parameter(
      torch.zeros(1, self.history_length, self.d_model)
    )
    self.history_encoder = self._build_history_encoder(self.history_layers)
    self.command_token_proj = nn.Linear(2, self.d_model)
    self.joint_embedding = nn.Parameter(torch.zeros(1, self.num_dofs, self.d_model))
    self.cross_blocks = self._build_cross_blocks(self.cross_layers)
    self.register_buffer(
      "_causal_mask",
      _make_causal_mask(self.history_length),
      persistent=False,
    )

  def _attention_latent_dim(self) -> int:
    return self.frame_dim + 2 * self.d_model

  def _attention_latent_from_flat(self, flat_obs: torch.Tensor) -> torch.Tensor:
    terms = self._term_history(flat_obs)
    proprio = self._proprio_history_from_terms(terms)
    proprio_tokens = self.proprio_proj(proprio) + self.proprio_pos_embedding
    dynamics_tokens = self.history_encoder(proprio_tokens, mask=self._causal_mask)
    dynamics = dynamics_tokens[:, -1]

    ref_tokens = self.command_token_proj(self._current_command_tokens_from_terms(terms))
    ref_tokens = ref_tokens + self.joint_embedding
    command_query = dynamics.unsqueeze(1)
    for block in self.cross_blocks:
      command_query = block(command_query, ref_tokens)
    command_embedding = command_query.squeeze(1)

    current_full_obs = self._frame_history_from_terms(terms)[:, -1]
    return torch.cat((current_full_obs, dynamics, command_embedding), dim=-1)
