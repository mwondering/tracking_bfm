"""Tests for the G1 BFM wbteleop tracking task."""

from __future__ import annotations

from dataclasses import asdict
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.wbteleop.algorithm import WbTeleopPPO, cosine_bc_weight
from mjlab.tasks.tracking.wbteleop.env_cfg import (
  unitree_g1_flat_tracking_bfm_wbteleop_env_cfg,
)
from mjlab.tasks.tracking.wbteleop.runner import WbTeleopTrackingRunner


TASK_ID = "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop"


def test_wbteleop_task_is_registered() -> None:
  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is WbTeleopTrackingRunner


def test_wbteleop_actor_obs_terms_are_exact() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert set(cfg.observations["actor"].terms.keys()) == {
    "command",
    "motion_ref_ang_vel",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  }


def test_wbteleop_actor_obs_excludes_privileged_terms() -> None:
  cfg = load_env_cfg(TASK_ID)
  terms = set(cfg.observations["actor"].terms.keys())

  assert "motion_anchor_pos_b" not in terms
  assert "motion_anchor_ori_b" not in terms
  assert "body_pos" not in terms
  assert "body_ori" not in terms
  assert "base_lin_vel" not in terms


def test_wbteleop_teacher_actor_is_teacher_only() -> None:
  env_cfg = load_env_cfg(TASK_ID)
  rl_cfg = load_rl_cfg(TASK_ID)

  assert "teacher_actor" in env_cfg.observations
  assert env_cfg.observations["teacher_actor"].enable_corruption is False
  assert asdict(rl_cfg)["obs_groups"] == {
    "actor": ("actor",),
    "critic": ("critic",),
  }


@pytest.mark.parametrize("play", [False, True])
def test_wbteleop_play_and_train_observation_structure_match(play: bool) -> None:
  cfg = load_env_cfg(TASK_ID, play=play)

  assert "actor" in cfg.observations
  assert "critic" in cfg.observations
  assert "teacher_actor" in cfg.observations
  assert cfg.observations["teacher_actor"].enable_corruption is False
  if play:
    assert cfg.observations["actor"].enable_corruption is False
  else:
    assert cfg.observations["actor"].enable_corruption is True


def test_wbteleop_history_support_sets_robot_history_only() -> None:
  cfg = unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(
    history_steps=10,
    future_steps=1,
  )
  terms = cfg.observations["actor"].terms

  assert cfg.commands["motion"].history_steps == 10
  assert cfg.commands["motion"].future_steps == 1
  assert getattr(terms["command"], "history_length", 0) in (0, None)
  assert getattr(terms["motion_ref_ang_vel"], "history_length", 0) in (0, None)
  for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
    assert terms[name].history_length == 11


def test_wbteleop_bc_weight_schedule_values() -> None:
  assert cosine_bc_weight(0, start=0.5, end=0.1, decay_steps=10_000) == pytest.approx(0.5)
  assert cosine_bc_weight(10_000, start=0.5, end=0.1, decay_steps=10_000) == pytest.approx(0.1)
  assert cosine_bc_weight(20_000, start=0.5, end=0.1, decay_steps=10_000) == pytest.approx(0.1)

  midpoint = cosine_bc_weight(5_000, start=0.5, end=0.1, decay_steps=10_000)
  assert 0.1 < midpoint < 0.5
  assert midpoint == pytest.approx(0.3)


def test_wbteleop_bc_weight_schedule_rejects_invalid_decay() -> None:
  with pytest.raises(ValueError, match="bc_decay_steps must be positive"):
    cosine_bc_weight(0, start=0.5, end=0.1, decay_steps=0)


def _make_wbteleop_algorithm_for_test() -> tuple[WbTeleopPPO, TensorDict]:
  obs = TensorDict(
    {
      "actor": torch.randn(4, 6),
      "critic": torch.randn(4, 5),
      "teacher_actor": torch.randn(4, 7),
    },
    batch_size=[4],
  )
  obs_groups = {"actor": ["actor"], "critic": ["critic"]}
  actor = MLPModel(
    obs,
    obs_groups,
    "actor",
    output_dim=3,
    hidden_dims=(16, 16),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.5,
      "std_type": "scalar",
    },
  )
  critic = MLPModel(
    obs,
    obs_groups,
    "critic",
    output_dim=1,
    hidden_dims=(16, 16),
    activation="elu",
    obs_normalization=False,
  )
  storage = RolloutStorage("rl", 4, 2, obs, [3], "cpu")
  alg = WbTeleopPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    learning_rate=1.0e-3,
    bc_weight_start=0.5,
    bc_weight_end=0.1,
    bc_decay_steps=10_000,
    device="cpu",
  )
  alg.set_teacher_adapter(
    TeacherPolicyAdapter(lambda teacher_obs: teacher_obs["teacher_actor"][..., :3] * 0.25)
  )
  return alg, obs


def test_wbteleop_ppo_update_reports_bc_metrics() -> None:
  alg, obs = _make_wbteleop_algorithm_for_test()

  for _ in range(2):
    actions = alg.act(obs)
    rewards = torch.ones(4)
    dones = torch.zeros(4, dtype=torch.long)
    alg.process_env_step(obs, rewards, dones, {})
  alg.compute_returns(obs)

  metrics = alg.update()

  assert "bc_mse" in metrics
  assert "bc_weight" in metrics
  assert "bc_loss" in metrics
  assert metrics["bc_weight"] == pytest.approx(0.5)
  assert metrics["bc_mse"] >= 0.0
  assert metrics["bc_loss"] >= 0.0


class _TeacherRunnerProbe:
  loaded_path = None
  last_cfg = None

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    self.env = env
    self.train_cfg = train_cfg
    self.device = device
    _TeacherRunnerProbe.last_cfg = train_cfg

  def load(self, path, map_location=None):
    _TeacherRunnerProbe.loaded_path = path

  def get_inference_policy(self, device=None):
    return lambda obs: obs["teacher_actor"][..., :3]


class _RunnerEnvProbe:
  def __init__(self):
    self.unwrapped = SimpleNamespace(common_step_counter=123)


def test_wbteleop_runner_builds_teacher_adapter() -> None:
  cfg = asdict(load_rl_cfg(TASK_ID))
  cfg["algorithm"]["teacher_checkpoint_path"] = "/tmp/teacher.pt"
  env = _RunnerEnvProbe()

  runner = WbTeleopTrackingRunner.__new__(WbTeleopTrackingRunner)
  runner.env = env
  runner.cfg = cfg
  runner.device = torch.device("cpu")

  with (
    patch("mjlab.tasks.tracking.wbteleop.runner.load_runner_cls", return_value=_TeacherRunnerProbe),
    patch(
      "mjlab.tasks.tracking.wbteleop.runner.load_rl_cfg",
      return_value=load_rl_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1"),
    ),
  ):
    adapter = runner._build_teacher_adapter()

  obs = TensorDict(
    {"teacher_actor": torch.ones(2, 5)},
    batch_size=[2],
  )
  assert _TeacherRunnerProbe.loaded_path == "/tmp/teacher.pt"
  assert adapter.act_mean(obs).shape == (2, 3)
  assert env.unwrapped.common_step_counter == 123
  assert _TeacherRunnerProbe.last_cfg["obs_groups"]["actor"] == ("teacher_actor",)


def test_wbteleop_runner_rejects_missing_teacher_checkpoint() -> None:
  cfg = asdict(load_rl_cfg(TASK_ID))
  cfg["algorithm"]["teacher_checkpoint_path"] = ""
  runner = WbTeleopTrackingRunner.__new__(WbTeleopTrackingRunner)
  runner.cfg = cfg
  runner.device = torch.device("cpu")

  with pytest.raises(ValueError, match="teacher_checkpoint_path must be provided"):
    runner._build_teacher_adapter()


def test_wbteleop_train_help_loads() -> None:
  result = subprocess.run(
    ["uv", "run", "train", TASK_ID, "--help"],
    check=False,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0
  assert TASK_ID in result.stdout
  assert "teacher-checkpoint-path" in result.stdout
