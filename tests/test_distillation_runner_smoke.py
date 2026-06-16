"""Smoke tests for the distillation runner."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

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
        "proprio_actor": base[:, :4].clone(),
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


def test_latent_distillation_runner_learn_smoke() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    save_interval=1,
    num_steps_per_env=3,
    max_iterations=1,
    num_learning_epochs=2,
    num_mini_batches=2,
    upload_model=False,
    student_model_type="latent",
    student_obs_group="unused_student_actor",
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    latent_dim=5,
    encoder_hidden_dims=(32, 32),
    decoder_hidden_dims=(32, 32),
    kl_weight=1.0e-3,
    kl_warmup_iterations=10,
    free_nats_per_dim=0.01,
    latent_smooth_weight=1.0e-3,
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
    assert "kl_loss" in runner.last_loss_dict
    assert "latent_std_mean" in runner.last_loss_dict
    assert Path(tmpdir, "model_0.pt").exists()


def test_distillation_runner_wandb_logger_uses_rsl_rl_logger(
  tmp_path: Path,
) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="wandb", upload_model=False)
  constructed: list[dict] = []

  class _WriterProbe:
    pass

  class _LoggerProbe:
    def __init__(
      self,
      *,
      log_dir,
      cfg,
      env_cfg,
      num_envs,
      is_distributed,
      gpu_world_size,
      gpu_global_rank,
      device,
    ) -> None:
      constructed.append(
        {
          "log_dir": log_dir,
          "cfg": cfg,
          "env_cfg": env_cfg,
          "num_envs": num_envs,
          "is_distributed": is_distributed,
          "gpu_world_size": gpu_world_size,
          "gpu_global_rank": gpu_global_rank,
          "device": device,
        }
      )
      self.writer = None
      self.logger_type = None
      self.disable_logs = bool(is_distributed and gpu_global_rank != 0)

    def init_logging_writer(self) -> None:
      self.logger_type = "WandbLogWriter"
      self.writer = _WriterProbe()

  with patch(
    "mjlab.tasks.distillation.rl.runner.Logger",
    _LoggerProbe,
    create=True,
  ):
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=str(tmp_path),
      device="cpu",
      teacher_adapter=TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25),
    )
    runner._prepare_logging_writer()

  assert len(constructed) == 1
  assert constructed[0]["cfg"]["logger"] == "wandb"
  assert constructed[0]["cfg"]["algorithm"]["rnd_cfg"] is None
  assert runner.logger is not None
  assert isinstance(runner.writer, _WriterProbe)
  assert runner.logger_type == "WandbLogWriter"


def test_latent_distillation_runner_can_use_wae_mmd_regularization() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    save_interval=1,
    num_steps_per_env=3,
    max_iterations=1,
    num_learning_epochs=1,
    num_mini_batches=1,
    upload_model=False,
    student_model_type="latent",
    student_obs_group="unused_student_actor",
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    latent_regularization="wae_mmd",
    mmd_weight=1.0e-2,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  runner = DistillationRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
    teacher_adapter=teacher_adapter,
  )

  runner.learn(num_learning_iterations=1)

  assert runner.last_loss_dict["kl_weight"] == 0.0
  assert runner.last_loss_dict["mmd_weight"] == 1.0e-2
  assert "mmd_loss" in runner.last_loss_dict


def test_latent_distillation_runner_can_use_bfmzero_sphere_regularization() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    save_interval=1,
    num_steps_per_env=3,
    max_iterations=1,
    num_learning_epochs=1,
    num_mini_batches=1,
    upload_model=False,
    student_model_type="latent",
    student_obs_group="unused_student_actor",
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    latent_regularization="bfmzero_sphere",
    sphere_orthonormal_weight=1.0e-2,
    sphere_knn_smooth_weight=1.0e-2,
    sphere_knn_k=2,
    sphere_knn_max_samples=8,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  runner = DistillationRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
    teacher_adapter=teacher_adapter,
  )

  runner.learn(num_learning_iterations=1)

  assert runner.last_loss_dict["kl_weight"] == 0.0
  assert runner.last_loss_dict["mmd_weight"] == 0.0
  assert runner.last_loss_dict["sphere_orthonormal_weight"] == 1.0e-2
  assert runner.last_loss_dict["sphere_knn_smooth_weight"] == 1.0e-2
  assert "sphere_orthonormal_loss" in runner.last_loss_dict
  assert "sphere_knn_smooth_loss" in runner.last_loss_dict


def test_latent_distillation_runner_passes_rollout_shape_and_dones_to_algorithm() -> None:
  env = _DummyVecEnv(num_envs=3)
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    num_steps_per_env=2,
    max_iterations=1,
    num_learning_epochs=1,
    num_mini_batches=1,
    upload_model=False,
    student_model_type="latent",
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
    latent_smooth_weight=1.0e-3,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)
  runner = DistillationRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
    teacher_adapter=teacher_adapter,
  )

  with patch.object(runner.alg, "update", wraps=runner.alg.update) as update:
    runner.learn(num_learning_iterations=1)

  kwargs = update.call_args.kwargs
  assert kwargs["rollout_shape"] == (2, 3)
  assert kwargs["dones"].shape == (2, 3)


def test_latent_distillation_runner_save_load_round_trip() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    upload_model=False,
    student_model_type="latent",
    student_obs_group="unused_student_actor",
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
    latent_dim=5,
    encoder_hidden_dims=(16, 16),
    decoder_hidden_dims=(16, 16),
  )

  with TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir, "latent_model.pt")
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=tmpdir,
      device="cpu",
    )
    runner.save(str(save_path))

    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_type"] == "latent"
    assert "encoder_state_dict" in checkpoint
    assert "decoder_state_dict" in checkpoint
    assert checkpoint["latent_cfg"]["latent_dim"] == 5

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


def test_distillation_runner_logs_scalars_into_separate_wandb_groups() -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)
  runner = DistillationRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
    teacher_adapter=TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25),
  )

  class _WriterProbe:
    def __init__(self) -> None:
      self.scalars: dict[str, float] = {}

    def add_scalar(self, key: str, value: float, step: int) -> None:
      assert step == 7
      self.scalars[key] = value

  writer = _WriterProbe()
  runner.writer = writer
  runner.last_loss_dict = {
    "action_mse": 0.1,
    "kl_loss": 0.2,
    "kl_weight": 0.3,
  }
  runner.last_train_metrics = {
    "beta_teacher": 0.4,
    "teacher_action_ratio": 0.5,
  }

  runner._log_train_iteration(
    it=7,
    total_iterations=8,
    collection_time=1.0,
    learn_time=1.0,
    env_metrics={"mean_reward": 2.0, "mean_episode_length": 3.0},
    aggregated_ep_info={
      "Episode_Reward/motion_body_pos": 4.0,
      "Episode_Metrics/error_body_pos": 5.0,
      "Episode_Termination/anchor_pos": 6.0,
      "Metrics/motion/sampling_entropy": 7.0,
      "return": 8.0,
    },
  )

  assert writer.scalars["Train/loss/action_mse"] == pytest.approx(0.1)
  assert writer.scalars["Train/loss/kl_loss"] == pytest.approx(0.2)
  assert writer.scalars["Train/loss/kl_weight"] == pytest.approx(0.3)
  assert writer.scalars["Train/metrics/beta_teacher"] == pytest.approx(0.4)
  assert writer.scalars["Train/metrics/teacher_action_ratio"] == pytest.approx(0.5)
  assert writer.scalars["Train/reward/mean_reward"] == pytest.approx(2.0)
  assert writer.scalars["Train/metrics/mean_episode_length"] == pytest.approx(3.0)
  assert writer.scalars["Train/reward/motion_body_pos"] == pytest.approx(4.0)
  assert writer.scalars["Train/metrics/error_body_pos"] == pytest.approx(5.0)
  assert writer.scalars["Train/termination/anchor_pos"] == pytest.approx(6.0)
  assert writer.scalars["Train/metrics/motion/sampling_entropy"] == pytest.approx(7.0)
  assert writer.scalars["Train/reward/return"] == pytest.approx(8.0)
  assert "Train/distill/action_mse" not in writer.scalars
  assert "Train/env/Episode_Reward/motion_body_pos" not in writer.scalars


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


def test_build_teacher_adapter_masks_distributed_env_for_nested_teacher_runner(
  monkeypatch,
) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    upload_model=False,
    teacher_checkpoint_path="dummy_teacher.pt",
    teacher_task_id="Dummy-Teacher-Task",
  )

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "0")
  monkeypatch.setenv("LOCAL_RANK", "0")

  seen_env: list[tuple[str | None, str | None, str | None]] = []

  class _TeacherRunnerProbe:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
      seen_env.append(
        (
          os.environ.get("WORLD_SIZE"),
          os.environ.get("RANK"),
          os.environ.get("LOCAL_RANK"),
        )
      )

    def load(self, path, map_location=None):
      return None

    def get_inference_policy(self, device=None):
      return lambda obs: obs["actor"][..., :3] * 0.5

  with (
    patch("torch.distributed.init_process_group"),
    patch("torch.cuda.set_device"),
    patch.object(TensorDict, "to", lambda self, *args, **kwargs: self),
    patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
    patch("mjlab.tasks.distillation.rl.runner.load_runner_cls", return_value=_TeacherRunnerProbe),
    patch(
      "mjlab.tasks.distillation.rl.runner.load_rl_cfg",
      return_value=DistillationRunnerCfg(logger="tensorboard", upload_model=False),
    ),
  ):
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:0",
    )
    runner._build_teacher_adapter()

  assert seen_env == [(None, None, None)]


def test_distillation_runner_collects_distributed_env_metrics_without_writer(
  monkeypatch,
) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

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
      device="cuda:1",
      teacher_adapter=teacher_adapter,
    )

  runner.device = torch.device("cpu")
  rewbuffer = deque([2.0, 4.0], maxlen=100)
  lenbuffer = deque([8.0, 10.0], maxlen=100)
  ep_infos = [{"reward_metric": torch.tensor(6.0)}]
  reduced = []

  def _fake_all_reduce(tensor: torch.Tensor, op=None):
    reduced.append(float(tensor.item()))
    tensor.mul_(2.0)

  with patch("torch.distributed.all_reduce", side_effect=_fake_all_reduce):
    env_metrics, aggregated = runner._collect_distributed_log_data(
      ep_infos=ep_infos,
      rewbuffer=rewbuffer,
      lenbuffer=lenbuffer,
    )

  assert env_metrics["mean_reward"] == pytest.approx(3.0)
  assert env_metrics["mean_episode_length"] == pytest.approx(9.0)
  assert aggregated["reward_metric"] == pytest.approx(6.0)
  assert reduced == [3.0, 9.0, 6.0]
