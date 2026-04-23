"""Smoke tests for the distillation runner."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict

from mjlab.tasks.distillation.config.g1.rl_cfg import DistillationRunnerCfg
from mjlab.tasks.distillation.rl.runner import DistillationRunner, mix_rollout_actions
from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter


class _DummyVecEnv:
  def __init__(self, num_envs: int = 4, obs_dim: int = 6, action_dim: int = 3):
    self.num_envs = num_envs
    self.device = torch.device("cpu")
    self.num_actions = action_dim
    self.max_episode_length = 8
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self._obs_dim = obs_dim
    self._step_count = torch.zeros(num_envs, dtype=torch.long)
    self.unwrapped = SimpleNamespace(common_step_counter=0)
    self.cfg = SimpleNamespace(is_finite_horizon=False)

  def reset(self):
    self._step_count.zero_()
    self.episode_length_buf.zero_()
    return self.get_observations(), {}

  def get_observations(self):
    base = torch.stack(
      [
        torch.linspace(-1.0, 1.0, self._obs_dim) + idx
        for idx in range(self.num_envs)
      ],
      dim=0,
    )
    return TensorDict(
      {
        "actor": base.clone(),
        "teacher_actor": base.clone(),
        "student_actor": base.clone(),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions: torch.Tensor):
    self._step_count += 1
    self.episode_length_buf += 1
    self.unwrapped.common_step_counter += self.num_envs
    rewards = -actions.square().sum(dim=-1)
    dones = (self._step_count >= self.max_episode_length).to(dtype=torch.long)
    if torch.any(dones > 0):
      self._step_count[dones > 0] = 0
      self.episode_length_buf[dones > 0] = 0
    extras = {
      "episode": {
        "return": rewards.mean(),
        "episode_len": torch.tensor(float(self.max_episode_length)),
        "anchor_pos_err": actions.abs().mean(),
      }
    }
    return self.get_observations(), rewards, dones, extras

  def close(self) -> None:
    return None


class _LogInfoVecEnv(_DummyVecEnv):
  def step(self, actions: torch.Tensor):
    obs, rewards, dones, _ = super().step(actions)
    extras = {
      "log": {
        "Metrics/rollout_reward": rewards.mean(),
        "Metrics/action_abs": actions.abs().mean(),
      }
    }
    return obs, rewards, dones, extras


def test_mix_rollout_actions_respects_extreme_betas() -> None:
  student = torch.zeros(4, 2)
  teacher = torch.ones(4, 2)

  rollout, mask = mix_rollout_actions(student, teacher, beta=1.0)
  assert torch.equal(rollout, teacher)
  assert torch.all(mask)

  rollout, mask = mix_rollout_actions(student, teacher, beta=0.0)
  assert torch.equal(rollout, student)
  assert not torch.any(mask)


def test_distillation_runner_learn_smoke() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    save_interval=1,
    num_steps_per_env=3,
    max_iterations=1,
    num_learning_epochs=2,
    num_mini_batches=2,
    upload_model=False,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  with TemporaryDirectory() as tmpdir:
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=tmpdir,
      device="cpu",
      teacher_adapter=teacher_adapter,
    )

    runner.learn(num_learning_iterations=1)

    assert "action_mse" in runner.last_loss_dict
    assert "teacher_action_ratio" in runner.last_train_metrics
    assert Path(tmpdir, "model_0.pt").exists()


def test_distillation_runner_accepts_tracking_runner_kwargs() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  runner = DistillationRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
    teacher_adapter=teacher_adapter,
    registry_name="dummy-registry-name",
  )

  assert runner is not None


def test_distillation_runner_allows_inference_without_teacher_checkpoint() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  runner = DistillationRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
  )

  policy = runner.get_inference_policy(device="cpu")

  assert policy is runner.student_policy


def test_distillation_runner_load_accepts_generic_play_signature() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  with TemporaryDirectory() as tmpdir:
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=tmpdir,
      device="cpu",
    )
    save_path = Path(tmpdir, "model_test.pt")
    runner.save(str(save_path))

    reloaded_runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cpu",
    )
    infos = reloaded_runner.load(
      str(save_path),
      load_cfg={"actor": True},
      strict=True,
      map_location="cpu",
    )

  assert infos is not None


def test_distillation_runner_load_rejects_teacher_checkpoint_shape() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  with TemporaryDirectory() as tmpdir:
    teacher_like_path = Path(tmpdir, "teacher_like.pt")
    torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}}, teacher_like_path)

    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cpu",
    )

    with pytest.raises(ValueError, match="tracking/teacher checkpoint"):
      runner.load(
        str(teacher_like_path),
        load_cfg={"actor": True},
        strict=True,
        map_location="cpu",
      )


def test_distillation_runner_prints_terminal_log(capsys) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    upload_model=False,
    save_interval=1,
    num_steps_per_env=2,
    max_iterations=1,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  with TemporaryDirectory() as tmpdir:
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=tmpdir,
      device="cpu",
      teacher_adapter=teacher_adapter,
    )
    runner.learn(num_learning_iterations=1)

  out = capsys.readouterr().out
  assert "Learning iteration 0/1" in out
  assert "action_mse" in out
  assert "beta_teacher" in out


def test_distillation_runner_consumes_log_extras(capsys) -> None:
  env = _LogInfoVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    upload_model=False,
    save_interval=1,
    num_steps_per_env=2,
    max_iterations=1,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  with TemporaryDirectory() as tmpdir:
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=tmpdir,
      device="cpu",
      teacher_adapter=teacher_adapter,
    )
    runner.learn(num_learning_iterations=1)

  out = capsys.readouterr().out
  assert "Metrics/rollout_reward" in out
  assert "Metrics/action_abs" in out
