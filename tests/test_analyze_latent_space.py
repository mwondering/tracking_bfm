"""Tests for latent-space analysis helpers."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.scripts import analyze_latent_space as latent_analysis
from mjlab.scripts.analyze_latent_space import (
  LatentSpaceAnalysisConfig,
  _apply_latent_analysis_overrides,
  _coerce_plot_range,
  _normalize_to_unit_sphere,
  collect_latent_batches,
  save_latent_plots,
)


class _DummyAutoResetEnv:
  def __init__(self, num_envs: int = 3):
    self.num_envs = num_envs
    self.device = torch.device("cpu")
    self.step_count = 0
    self.actions: list[torch.Tensor] = []

  def get_observations(self) -> TensorDict:
    base = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(-1)
    step = torch.full((self.num_envs, 1), float(self.step_count))
    return TensorDict(
      {
        "teacher_actor": torch.cat([base, step], dim=-1),
        "proprio_actor": torch.cat([base + 10.0, step], dim=-1),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions: torch.Tensor):
    self.actions.append(actions.detach().clone())
    self.step_count += 1
    dones = torch.zeros(self.num_envs, dtype=torch.long)
    if self.step_count % 2 == 0:
      dones[1] = 1
    return self.get_observations(), torch.zeros(self.num_envs), dones, {}


class _DummyLatentPolicy(torch.nn.Module):
  latent_dim = 2
  encoder_obs_group = "teacher_actor"
  decoder_obs_group = "proprio_actor"

  def encode(self, obs: TensorDict):
    teacher = obs["teacher_actor"]
    mu = torch.stack([teacher[:, 0], teacher[:, 1]], dim=-1)
    log_std = torch.zeros_like(mu)
    return mu, log_std

  def act(self, obs: TensorDict, deterministic: bool = True) -> torch.Tensor:
    del deterministic
    return torch.zeros(obs.batch_size[0], 3)


def test_collect_latent_batches_stops_at_requested_points_and_continues_after_done() -> None:
  env = _DummyAutoResetEnv(num_envs=3)
  policy = _DummyLatentPolicy()

  result = collect_latent_batches(
    env=env,
    policy=policy,
    num_points=7,
    deterministic=True,
    device=torch.device("cpu"),
  )

  assert result["mu"].shape == (7, 2)
  assert result["z"].shape == (7, 2)
  assert result["log_std"].shape == (7, 2)
  assert result["dones"].shape == (7,)
  assert result["step"].tolist() == [0, 0, 0, 1, 1, 1, 2]
  assert result["dones"].sum().item() == 1
  assert len(env.actions) == 3


class _DummyMotionCfg:
  def __init__(self) -> None:
    self.motion_path = None
    self.sampling_mode = "default"
    self.history_steps = 5
    self.future_steps = 5


class _DummyTerm:
  def __init__(self, history_length: int = 0) -> None:
    self.history_length = history_length


class _DummyGroup:
  def __init__(self) -> None:
    self.terms = {
      "projected_gravity": _DummyTerm(),
      "base_ang_vel": _DummyTerm(),
      "joint_pos": _DummyTerm(),
      "joint_vel": _DummyTerm(),
      "actions": _DummyTerm(),
    }


class _DummyScene:
  num_envs = 1


class _DummyEnvCfg:
  def __init__(self) -> None:
    self.scene = _DummyScene()
    self.commands = {"motion": _DummyMotionCfg()}
    self.observations = {"proprio_actor": _DummyGroup()}


def test_apply_latent_analysis_overrides_keeps_checkpoint_observation_shape_compatible() -> None:
  env_cfg = _DummyEnvCfg()
  cfg = LatentSpaceAnalysisConfig(
    checkpoint_file="checkpoint.pt",
    output_dir="out",
    motion_path="/motions",
    num_envs=8,
    sampling_mode="uniform",
    motion_history_steps=0,
    motion_future_steps=1,
    proprio_history_length=20,
  )

  _apply_latent_analysis_overrides(env_cfg, cfg)

  assert env_cfg.scene.num_envs == 8
  assert env_cfg.commands["motion"].motion_path == "/motions"
  assert env_cfg.commands["motion"].sampling_mode == "uniform"
  assert env_cfg.commands["motion"].history_steps == 0
  assert env_cfg.commands["motion"].future_steps == 1
  assert {
    name: term.history_length
    for name, term in env_cfg.observations["proprio_actor"].terms.items()
  } == {
    "projected_gravity": 20,
    "base_ang_vel": 20,
    "joint_pos": 20,
    "joint_vel": 20,
    "actions": 20,
  }


def test_latent_analysis_defaults_to_fixed_plot_range() -> None:
  cfg = LatentSpaceAnalysisConfig(checkpoint_file="checkpoint.pt", output_dir="out")

  assert cfg.plot_range == (-20.0, 20.0)
  assert _coerce_plot_range(cfg.plot_range) == (-20.0, 20.0)


def test_latent_analysis_can_disable_fixed_plot_range() -> None:
  assert _coerce_plot_range(None) is None


def test_normalize_to_unit_sphere_keeps_nonzero_latents_on_sphere() -> None:
  samples = torch.tensor(
    [
      [3.0, 4.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, -5.0, 12.0],
    ]
  )

  normalized = _normalize_to_unit_sphere(samples)

  assert torch.allclose(normalized[0].norm(), torch.tensor(1.0))
  assert torch.allclose(normalized[1], torch.zeros(3))
  assert torch.allclose(normalized[2].norm(), torch.tensor(1.0))


def test_save_latent_plots_writes_only_spherical_tsne_images(
  tmp_path,
  monkeypatch,
) -> None:
  latents = {
    "z": torch.tensor(
      [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
        [5.0, 0.0, 0.0],
      ]
    ),
    "mu": torch.zeros(4, 3),
    "log_std": torch.zeros(4, 3),
    "dones": torch.zeros(4, dtype=torch.bool),
    "step": torch.arange(4),
  }
  stale_plot = tmp_path / "pca_z_vs_prior.png"
  stale_plot.write_text("old")
  seen: dict[int, torch.Tensor] = {}

  def fake_tsne_embedding(
    samples: torch.Tensor,
    n_components: int,
    *,
    random_state: int = 0,
    perplexity: float = 30.0,
  ):
    del random_state, perplexity
    seen[n_components] = samples.detach().clone()
    coords = torch.arange(samples.shape[0] * n_components, dtype=torch.float32)
    return coords.view(samples.shape[0], n_components).numpy()

  monkeypatch.setattr(latent_analysis, "_tsne_embedding", fake_tsne_embedding)

  save_latent_plots(
    latents,
    tmp_path,
    max_plot_points=4,
    plot_range=None,
  )

  assert sorted(path.name for path in tmp_path.glob("*.png")) == [
    "tsne_sphere_z_2d.png",
    "tsne_sphere_z_3d.png",
  ]
  assert not stale_plot.exists()
  assert torch.allclose(seen[2].norm(dim=-1), torch.ones(4))
  assert torch.allclose(seen[3], seen[2])
