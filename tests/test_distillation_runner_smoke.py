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
from unittest.mock import patch

from mjlab.tasks.distillation.config.g1.rl_cfg import DistillationRunnerCfg
from mjlab.tasks.distillation.rl.algorithm import ActionDistillationAlgorithm
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


def test_distillation_runner_configures_multi_gpu_state_from_environment(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

  with (
    patch("torch.distributed.init_process_group") as init_pg,
    patch("torch.cuda.set_device") as set_device,
    patch.object(TensorDict, "to", lambda self, *args, **kwargs: self),
    patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
  ):
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:1",
      teacher_adapter=TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25),
    )

  assert runner.is_distributed is True
  assert runner.gpu_world_size == 2
  assert runner.gpu_global_rank == 1
  assert runner.gpu_local_rank == 1
  assert runner.disable_logs is True
  init_pg.assert_called_once()
  set_device.assert_called_once_with(1)


def test_distillation_runner_rejects_mismatched_local_rank_device(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

  with pytest.raises(ValueError, match="does not match expected device"):
    DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:0",
      teacher_adapter=TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25),
    )


def test_distillation_runner_distributed_learn_broadcasts_and_skips_nonzero_rank_outputs(
  monkeypatch,
) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    upload_model=False,
    save_interval=1,
    num_steps_per_env=2,
    max_iterations=1,
    num_learning_epochs=1,
    num_mini_batches=1,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

  with TemporaryDirectory() as tmpdir:
    with (
      patch("torch.distributed.init_process_group"),
      patch("torch.cuda.set_device"),
      patch("torch.distributed.all_reduce"),
      patch.object(TensorDict, "to", lambda self, *args, **kwargs: self),
      patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
      patch.object(DistillationRunner, "_prepare_logging_writer") as prepare_writer,
      patch.object(DistillationRunner, "save") as save,
      patch.object(ActionDistillationAlgorithm, "broadcast_parameters") as broadcast,
    ):
      runner = DistillationRunner(
        env,
        asdict(cfg),
        log_dir=tmpdir,
        device="cuda:1",
        teacher_adapter=teacher_adapter,
      )
      runner.device = torch.device("cpu")
      runner.learn(num_learning_iterations=1)

  broadcast.assert_called_once()
  prepare_writer.assert_not_called()
  save.assert_not_called()


def test_distillation_runner_reduces_logged_scalars_across_ranks(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "0")
  monkeypatch.setenv("LOCAL_RANK", "0")

  with (
    patch("torch.distributed.init_process_group"),
    patch("torch.cuda.set_device"),
    patch.object(TensorDict, "to", lambda self, *args, **kwargs: self),
    patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
  ):
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:0",
      teacher_adapter=teacher_adapter,
    )

  runner.device = torch.device("cpu")
  reduced = []

  def _fake_all_reduce(tensor: torch.Tensor, op=None):
    reduced.append(float(tensor.item()))
    tensor.mul_(2.0)

  with patch("torch.distributed.all_reduce", side_effect=_fake_all_reduce):
    value = runner._distributed_mean(3.0)

  assert value == pytest.approx(3.0)
  assert reduced == [3.0]
