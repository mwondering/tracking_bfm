"""Tests for action trunk support."""

from unittest.mock import Mock

import torch

import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg


def _make_mock_action_term(action_dim: int):
  def factory(env):
    term = Mock()
    term.action_dim = action_dim
    term.raw_action = torch.zeros(env.num_envs, action_dim, device=env.device)
    term.process_actions = Mock()
    term.apply_actions = Mock()
    term.reset = Mock()
    return term

  return factory


def _make_mock_env(action_trunk_len: int = 1):
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  env.cfg = Mock(action_trunk_len=action_trunk_len, decimation=action_trunk_len)
  return env


def _make_fake_step_env(action_trunk_len: int):
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  env = object.__new__(ManagerBasedRlEnv)
  env.cfg = Mock(
    decimation=4,
    action_trunk_len=action_trunk_len,
    auto_reset=True,
    is_finite_horizon=False,
    sim=Mock(mujoco=Mock(timestep=0.01)),
  )
  env._manual_reset_pending = torch.zeros(1, dtype=torch.bool)
  env._sim_step_counter = 0
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)
  env.common_step_counter = 0
  env.extras = {}

  env.action_manager = Mock()
  env.action_manager.apply_action = Mock()
  env.scene = Mock()
  env.sim = Mock(device="cpu")
  env.metrics_manager = Mock()
  env.termination_manager = Mock()
  env.termination_manager.compute.return_value = torch.zeros(1, dtype=torch.bool)
  env.termination_manager.terminated = torch.zeros(1, dtype=torch.bool)
  env.termination_manager.time_outs = torch.zeros(1, dtype=torch.bool)
  env.reward_manager = Mock()
  env.reward_manager.compute.return_value = torch.zeros(1)
  env.command_manager = Mock()
  env.event_manager = Mock()
  env.event_manager.available_modes = set()
  env.observation_manager = Mock()
  env.observation_manager.compute.return_value = {"actor": torch.zeros(1, 1)}
  env.recorder_manager = Mock()
  return env


def test_action_trunk_task_is_registered() -> None:
  assert "Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk" in list_tasks()


def test_action_trunk_task_config_uses_four_slices() -> None:
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk")

  assert cfg.action_trunk_len == 4
  assert cfg.decimation == 4


def test_action_manager_policy_dim_expands_with_trunk_len() -> None:
  env = _make_mock_env(action_trunk_len=4)
  cfg = Mock()
  cfg.build = _make_mock_action_term(action_dim=3)
  cfg.entity_name = "robot"

  manager = ActionManager({"action": cfg}, env)

  assert manager.total_action_dim == 3
  assert manager.policy_action_dim == 12
  assert manager.action.shape == (2, 12)
  assert manager.applied_action.shape == (2, 3)


def test_rsl_rl_wrapper_uses_policy_action_dim() -> None:
  class FakeEnv:
    def __init__(self):
      self.num_envs = 2
      self.device = "cpu"
      self.max_episode_length = 10
      self.action_manager = Mock(total_action_dim=3, policy_action_dim=12)

    @property
    def unwrapped(self):
      return self

    def reset(self):
      return {"actor": torch.zeros(2, 1)}, {}

  wrapped = RslRlVecEnvWrapper(FakeEnv())

  assert wrapped.num_actions == 12


def test_last_action_observation_returns_full_trunk() -> None:
  from mjlab.envs.mdp.observations import last_action

  env = _make_mock_env(action_trunk_len=4)
  cfg = Mock()
  cfg.build = _make_mock_action_term(action_dim=3)
  cfg.entity_name = "robot"
  env.action_manager = ActionManager({"action": cfg}, env)
  action = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12)

  env.action_manager.process_action(action)

  torch.testing.assert_close(last_action(env), action)
  assert last_action(env).shape == (2, 12)


def test_action_trunk_len_must_match_decimation_for_trunk_mode() -> None:
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=Mock(),
    action_trunk_len=2,
  )
  assert cfg.action_trunk_len == 2


def test_step_loop_passes_substep_indices_to_action_manager_in_trunk_mode() -> None:
  """The decimation loop applies one trunk slice per physics substep."""
  env = _make_fake_step_env(action_trunk_len=4)

  env.step(torch.zeros(1, 12))

  assert [
    call.kwargs["substep_idx"]
    for call in env.action_manager.apply_action.call_args_list
  ] == [0, 1, 2, 3]


def test_step_loop_preserves_action_repeat_for_standard_tracking() -> None:
  """Default action mode keeps the old process-once/apply-repeat behavior."""
  env = _make_fake_step_env(action_trunk_len=1)

  env.step(torch.zeros(1, 3))

  assert env.action_manager.process_action.call_count == 1
  assert env.action_manager.apply_action.call_count == 4
  assert all(
    call.kwargs == {} for call in env.action_manager.apply_action.call_args_list
  )


def test_action_rate_l2_penalizes_first_trunk_slice_only() -> None:
  from mjlab.envs.mdp.rewards import action_rate_l2

  env = _make_mock_env(action_trunk_len=4)
  cfg = Mock()
  cfg.build = _make_mock_action_term(action_dim=1)
  cfg.entity_name = "robot"
  env.action_manager = ActionManager({"action": cfg}, env)

  prev = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
  current = torch.tensor([[2.0, 4.0, 7.0, 11.0], [2.0, 2.0, 2.0, 2.0]])

  env.action_manager.process_action(prev)
  env.action_manager.process_action(current)

  result = action_rate_l2(env)

  expected_env0 = (2.0 - 0.0) ** 2
  expected_env1 = (2.0 - 1.0) ** 2
  expected = torch.tensor([expected_env0, expected_env1])

  torch.testing.assert_close(result, expected)
