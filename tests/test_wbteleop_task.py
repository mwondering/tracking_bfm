"""Tests for the G1 BFM wbteleop tracking task."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
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
from mjlab.tasks.tracking.wbteleop import observations as wbteleop_observations
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
    "ref_limb_ee_pose_b",
    "robot_limb_ee_pose_b",
    "motion_ref_ang_vel",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  }


def test_wbteleop_adds_pelvis_limb_ee_pose_rewards() -> None:
  cfg = load_env_cfg(TASK_ID)
  pos_reward = cfg.rewards["pelvis_limb_ee_pos"]
  ori_reward = cfg.rewards["pelvis_limb_ee_ori"]

  assert pos_reward.weight == 1.0
  assert ori_reward.weight == 1.0
  assert pos_reward.params["std"] == 0.3
  assert ori_reward.params["std"] == 0.4
  for reward in (pos_reward, ori_reward):
    assert reward.params["command_name"] == "motion"
    assert reward.params["anchor_body_name"] == "pelvis"
    assert reward.params["body_names"] == (
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
      "left_ankle_roll_link",
      "right_ankle_roll_link",
    )


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
  rl_dict = asdict(rl_cfg)

  assert "teacher_actor" in env_cfg.observations
  assert env_cfg.observations["teacher_actor"].enable_corruption is False
  assert rl_dict["obs_groups"] == {
    "actor": ("actor",),
    "critic": ("critic",),
  }
  assert rl_dict["algorithm"]["pure_bc_enabled"] is False
  assert rl_dict["algorithm"]["pure_bc_weight"] == 1.0
  assert rl_dict["algorithm"]["pure_bc_rollout"] == "student"
  assert rl_dict["algorithm"]["bc_actor_checkpoint_path"] == ""
  assert rl_dict["algorithm"]["init_actor_std_from_teacher"] is False
  assert rl_dict["algorithm"]["init_critic_from_teacher"] is True


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
    future_steps=3,
  )
  terms = cfg.observations["actor"].terms

  assert cfg.commands["motion"].history_steps == 10
  assert cfg.commands["motion"].future_steps == 3
  assert getattr(terms["command"], "history_length", 0) in (0, None)
  assert getattr(terms["ref_limb_ee_pose_b"], "history_length", 0) in (0, None)
  assert terms["ref_limb_ee_pose_b"].params["history_steps"] == 10
  assert terms["ref_limb_ee_pose_b"].params["future_steps"] == 3
  assert terms["robot_limb_ee_pose_b"].history_length == 11
  assert "history_steps" not in terms["robot_limb_ee_pose_b"].params
  assert "future_steps" not in terms["robot_limb_ee_pose_b"].params
  assert getattr(terms["motion_ref_ang_vel"], "history_length", 0) in (0, None)
  for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
    assert terms[name].history_length == 11


class _WbTeleopCommandManager:
  def __init__(self, command):
    self._command = command

  def get_term(self, name: str):
    assert name == "motion"
    return self._command


def _make_wbteleop_obs_env():
  body_names = (
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=body_names, history_steps=0, future_steps=1),
    body_pos_w=torch.tensor(
      [
        [
          [1.0, 2.0, 3.0],
          [1.2, 2.1, 3.1],
          [0.8, 2.1, 3.2],
          [1.1, 1.9, 2.5],
          [0.9, 1.9, 2.4],
        ]
      ],
      dtype=torch.float32,
    ),
    body_quat_w=identity.repeat(1, len(body_names), 1),
    robot_body_pos_w=torch.tensor(
      [
        [
          [10.0, 20.0, 30.0],
          [10.3, 20.1, 30.2],
          [9.7, 20.1, 30.4],
          [10.2, 19.8, 29.5],
          [9.8, 19.8, 29.4],
        ]
      ],
      dtype=torch.float32,
    ),
    robot_body_quat_w=identity.repeat(1, len(body_names), 1),
  )
  return SimpleNamespace(
    num_envs=1,
    command_manager=_WbTeleopCommandManager(command),
  )


def test_wbteleop_limb_ee_pose_terms_use_reference_and_robot_pelvis_frames() -> None:
  env = _make_wbteleop_obs_env()
  body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  ref = wbteleop_observations.ref_limb_ee_pose_b(
    env,
    command_name="motion",
    body_names=body_names,
    anchor_body_name="pelvis",
    history_steps=0,
    future_steps=1,
  )
  robot = wbteleop_observations.robot_limb_ee_pose_b(
    env,
    command_name="motion",
    body_names=body_names,
    anchor_body_name="pelvis",
  )

  assert ref.shape == (1, 36)
  assert robot.shape == (1, 36)
  torch.testing.assert_close(ref[0, :3], torch.tensor([0.2, 0.1, 0.1]))
  torch.testing.assert_close(robot[0, :3], torch.tensor([0.3, 0.1, 0.2]))


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
    alg.act(obs)
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


def test_wbteleop_bc_only_update_updates_actor_not_critic() -> None:
  alg, obs = _make_wbteleop_algorithm_for_test()
  actor_before = {
    key: value.detach().clone() for key, value in alg.actor.state_dict().items()
  }
  critic_before = {
    key: value.detach().clone() for key, value in alg.critic.state_dict().items()
  }

  for _ in range(2):
    alg.act(obs)
    rewards = torch.ones(4)
    dones = torch.zeros(4, dtype=torch.long)
    alg.process_env_step(obs, rewards, dones, {})

  metrics = alg.update_bc_only()

  assert "pure_bc_mse" in metrics
  assert "pure_bc_loss" in metrics
  assert metrics["pure_bc_weight"] == pytest.approx(1.0)
  assert any(
    not torch.equal(value, actor_before[key])
    for key, value in alg.actor.state_dict().items()
  )
  assert all(
    torch.equal(value, critic_before[key])
    for key, value in alg.critic.state_dict().items()
  )


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


class _PureBcAlgProbe:
  def __init__(self):
    self.learning_rate = 1.0e-3
    self.rnd = None
    self.update_calls = 0
    self.processed_steps = 0
    self.trained = False

  def train_mode(self):
    self.trained = True

  def act(self, obs):
    return torch.zeros(obs.batch_size[0], 3)

  def process_env_step(self, obs, rewards, dones, extras):
    self.processed_steps += 1

  def update_bc_only(self):
    self.update_calls += 1
    return {"pure_bc_mse": 0.25, "pure_bc_loss": 0.25, "pure_bc_weight": 1.0}

  def get_policy(self):
    return SimpleNamespace(output_std=torch.ones(3))


class _PureBcEnvProbe:
  def __init__(self):
    self.num_envs = 2
    self.device = torch.device("cpu")
    self.max_episode_length = 8
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self.step_calls = 0

  def get_observations(self):
    return TensorDict(
      {
        "actor": torch.ones(self.num_envs, 4),
        "critic": torch.ones(self.num_envs, 5),
        "teacher_actor": torch.ones(self.num_envs, 6),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions):
    self.step_calls += 1
    rewards = torch.ones(self.num_envs)
    dones = torch.zeros(self.num_envs, dtype=torch.long)
    extras = {"episode": {"reward": rewards.mean()}}
    return self.get_observations(), rewards, dones, extras


class _LoggerProbe:
  def __init__(self):
    self.writer = None
    self.log_dir = None
    self.logged_losses = []
    self.env_steps = 0
    self.initialized = False

  def init_logging_writer(self):
    self.initialized = True

  def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
    self.env_steps += 1

  def log(
    self,
    *,
    it,
    start_it,
    total_it,
    collect_time,
    learn_time,
    loss_dict,
    learning_rate,
    action_std,
    rnd_weight,
  ):
    self.logged_losses.append(loss_dict)


def test_wbteleop_pure_bc_learn_uses_adaptive_iteration_and_bc_update() -> None:
  runner = WbTeleopTrackingRunner.__new__(WbTeleopTrackingRunner)
  runner.env = _PureBcEnvProbe()
  runner.alg = _PureBcAlgProbe()
  runner.logger = _LoggerProbe()
  runner.device = torch.device("cpu")
  runner.current_learning_iteration = 0
  runner.is_distributed = False
  runner.cfg = {
    "num_steps_per_env": 2,
    "save_interval": 100,
    "algorithm": {"rnd_cfg": None},
  }
  adaptive_iterations = []
  runner._begin_adaptive_sampling_iteration = adaptive_iterations.append

  runner._learn_pure_bc(num_learning_iterations=1, init_at_random_ep_len=False)

  assert adaptive_iterations == [0]
  assert runner.env.step_calls == 2
  assert runner.alg.processed_steps == 2
  assert runner.alg.update_calls == 1
  assert runner.logger.logged_losses == [
    {"pure_bc_mse": 0.25, "pure_bc_loss": 0.25, "pure_bc_weight": 1.0}
  ]


def test_wbteleop_scratch_initializes_actor_and_critic(tmp_path) -> None:
  alg, _ = _make_wbteleop_algorithm_for_test()
  actor_source, _ = _make_wbteleop_algorithm_for_test()
  critic_source, _ = _make_wbteleop_algorithm_for_test()

  for param in actor_source.actor.parameters():
    param.data.fill_(0.123)
  for param in critic_source.critic.parameters():
    param.data.fill_(0.456)

  actor_ckpt = tmp_path / "bc_actor.pt"
  teacher_ckpt = tmp_path / "teacher.pt"
  torch.save({"actor_state_dict": actor_source.actor.state_dict()}, actor_ckpt)
  torch.save({"critic_state_dict": critic_source.critic.state_dict()}, teacher_ckpt)

  runner = WbTeleopTrackingRunner.__new__(WbTeleopTrackingRunner)
  runner.alg = alg
  runner.cfg = {
    "resume": False,
    "algorithm": {
      "bc_actor_checkpoint_path": str(actor_ckpt),
      "teacher_checkpoint_path": str(teacher_ckpt),
      "init_critic_from_teacher": True,
      "strict_init": True,
    },
  }
  runner.device = torch.device("cpu")

  runner._maybe_initialize_from_pretrained()

  for key, value in alg.actor.state_dict().items():
    assert torch.equal(value, actor_source.actor.state_dict()[key])
  for key, value in alg.critic.state_dict().items():
    assert torch.equal(value, critic_source.critic.state_dict()[key])


def test_wbteleop_pure_bc_scratch_initializes_only_actor_std_from_teacher(
  tmp_path,
) -> None:
  alg, _ = _make_wbteleop_algorithm_for_test()
  teacher_source, _ = _make_wbteleop_algorithm_for_test()
  alg.pure_bc_enabled = True

  actor_before = {
    key: value.detach().clone() for key, value in alg.actor.state_dict().items()
  }
  teacher_actor_state = teacher_source.actor.state_dict()
  teacher_actor_state["distribution.std_param"].fill_(0.234)
  for key, value in teacher_actor_state.items():
    if key != "distribution.std_param":
      value.fill_(0.789)

  teacher_ckpt = tmp_path / "teacher.pt"
  torch.save({"actor_state_dict": teacher_actor_state}, teacher_ckpt)

  runner = WbTeleopTrackingRunner.__new__(WbTeleopTrackingRunner)
  runner.alg = alg
  runner.cfg = {
    "resume": False,
    "algorithm": {
      "teacher_checkpoint_path": str(teacher_ckpt),
      "init_actor_std_from_teacher": True,
    },
  }
  runner.device = torch.device("cpu")

  runner._maybe_initialize_from_pretrained()

  for key, value in alg.actor.state_dict().items():
    if key == "distribution.std_param":
      assert torch.equal(value, teacher_actor_state[key])
    else:
      assert torch.equal(value, actor_before[key])


def test_wbteleop_resume_skips_scratch_initialization(tmp_path) -> None:
  alg, _ = _make_wbteleop_algorithm_for_test()
  actor_before = {
    key: value.detach().clone() for key, value in alg.actor.state_dict().items()
  }
  critic_before = {
    key: value.detach().clone() for key, value in alg.critic.state_dict().items()
  }

  runner = WbTeleopTrackingRunner.__new__(WbTeleopTrackingRunner)
  runner.alg = alg
  runner.cfg = {
    "resume": True,
    "algorithm": {
      "bc_actor_checkpoint_path": str(tmp_path / "missing_actor.pt"),
      "teacher_checkpoint_path": str(tmp_path / "missing_teacher.pt"),
      "init_critic_from_teacher": True,
      "strict_init": True,
    },
  }
  runner.device = torch.device("cpu")

  runner._maybe_initialize_from_pretrained()

  for key, value in alg.actor.state_dict().items():
    assert torch.equal(value, actor_before[key])
  for key, value in alg.critic.state_dict().items():
    assert torch.equal(value, critic_before[key])


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
