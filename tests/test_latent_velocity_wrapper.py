"""Tests for latent decoder VecEnv wrapper."""

import torch
import pytest
from tensordict import TensorDict

from mjlab.utils.spaces import Box, batch_space
from mjlab.tasks.latentvelocity.rl.latent_decoder_wrapper import (
  LatentDecoderVecEnvWrapper,
)
from mjlab.tasks.latentvelocity.rl.runner import LatentVelocityOnPolicyRunner
from mjlab.tasks.distillation.rl.models import build_latent_student_model


class _DummyBaseEnv:
  def __init__(self):
    self.num_envs = 2
    self.num_actions = 29
    self.device = torch.device("cpu")
    self.max_episode_length = 100
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self.cfg = object()
    self.last_actions = None
    self.single_action_space = Box(shape=(self.num_actions,), low=-1.0, high=1.0)
    self.action_space = batch_space(self.single_action_space, self.num_envs)

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    return TensorDict(
      {
        "actor": torch.zeros(self.num_envs, 4),
        "critic": torch.zeros(self.num_envs, 5),
        "proprio_actor": torch.ones(self.num_envs, 3),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions):
    self.last_actions = actions.detach().clone()
    return (
      self.get_observations(),
      torch.ones(self.num_envs),
      torch.zeros(self.num_envs, dtype=torch.long),
      {},
    )

  def reset(self):
    return self.get_observations(), {}

  def close(self):
    pass


class _DummyDecoder(torch.nn.Module):
  def decode(self, obs, z):
    proprio = obs["proprio_actor"]
    return torch.cat([proprio, z], dim=-1)


def test_latent_wrapper_exposes_latent_action_dim_and_steps_decoded_actions() -> None:
  base_env = _DummyBaseEnv()
  wrapper = LatentDecoderVecEnvWrapper(
    base_env,
    decoder=_DummyDecoder(),
    latent_dim=2,
    proprio_obs_group="proprio_actor",
    latent_action_clip=6.0,
  )

  obs = wrapper.get_observations()
  actions = torch.tensor([[10.0, -10.0], [0.5, -0.5]])
  next_obs, rewards, dones, extras = wrapper.step(actions)

  assert wrapper.num_actions == 2
  assert wrapper.single_action_space.shape == (2,)
  assert wrapper.single_action_space.low == -6.0
  assert wrapper.single_action_space.high == 6.0
  assert wrapper.action_space.shape == (2, 2)
  assert obs.batch_size == torch.Size([2])
  assert next_obs.batch_size == torch.Size([2])
  assert rewards.tolist() == [1.0, 1.0]
  assert dones.tolist() == [0, 0]
  torch.testing.assert_close(
    base_env.last_actions,
    torch.tensor([[1.0, 1.0, 1.0, 6.0, -6.0], [1.0, 1.0, 1.0, 0.5, -0.5]]),
  )
  assert extras["log"]["latent/norm_mean"] > 0.0
  assert extras["log"]["latent/abs_max"] == 6.0


def test_latent_wrapper_freezes_decoder_parameters() -> None:
  decoder = torch.nn.Linear(3, 2)
  base_env = _DummyBaseEnv()

  wrapper = LatentDecoderVecEnvWrapper(
    base_env,
    decoder=decoder,
    latent_dim=2,
    proprio_obs_group="proprio_actor",
    latent_action_clip=6.0,
  )

  assert not wrapper.decoder.training
  assert all(not p.requires_grad for p in wrapper.decoder.parameters())


def test_latent_runner_requires_decoder_checkpoint() -> None:
  base_env = _DummyBaseEnv()
  cfg = {
    "actor": {
      "hidden_dims": [16],
      "activation": "elu",
      "obs_normalization": False,
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    },
    "critic": {"hidden_dims": [16], "activation": "elu", "obs_normalization": False},
    "algorithm": {
      "class_name": "PPO",
      "num_learning_epochs": 1,
      "num_mini_batches": 1,
      "learning_rate": 1.0e-3,
      "schedule": "fixed",
      "gamma": 0.99,
      "lam": 0.95,
      "entropy_coef": 0.0,
      "desired_kl": 0.01,
      "max_grad_norm": 1.0,
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": True,
      "clip_param": 0.2,
    },
    "obs_groups": {"actor": ("actor",), "critic": ("critic",)},
    "num_steps_per_env": 2,
    "max_iterations": 1,
    "save_interval": 50,
    "experiment_name": "test",
    "logger": "tensorboard",
    "upload_model": False,
    "clip_actions": None,
    "latent_decoder_checkpoint_path": "",
    "latent_dim": 2,
    "latent_action_clip": 6.0,
    "proprio_obs_group": "proprio_actor",
  }

  with pytest.raises(ValueError, match="latent_decoder_checkpoint_path"):
    LatentVelocityOnPolicyRunner(base_env, cfg, log_dir=None, device="cpu")


def test_latent_runner_loads_decoder_checkpoint(tmp_path) -> None:
  env = _DummyBaseEnv()
  env.num_actions = 5
  obs = TensorDict(
    {
      "teacher_actor": torch.zeros(env.num_envs, 4),
      "proprio_actor": torch.ones(env.num_envs, 3),
    },
    batch_size=[env.num_envs],
  )
  model = build_latent_student_model(
    obs=obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    action_dim=env.num_actions,
    latent_dim=2,
    encoder_hidden_dims=(4,),
    decoder_hidden_dims=(4,),
    activation="elu",
    obs_normalization=True,
  )
  checkpoint_path = tmp_path / "latent.pt"
  torch.save(
    {
      "model_type": "latent",
      "policy_state_dict": model.state_dict(),
      "latent_cfg": {"latent_dim": 2},
    },
    checkpoint_path,
  )

  loaded = LatentVelocityOnPolicyRunner._load_decoder(
    env,
    {
      "latent_decoder_checkpoint_path": str(checkpoint_path),
      "latent_dim": 2,
      "proprio_obs_group": "proprio_actor",
    },
    device="cpu",
  )

  actions = loaded.decode(obs, torch.zeros(env.num_envs, 2))
  assert actions.shape == (env.num_envs, env.num_actions)
  assert not loaded.training
