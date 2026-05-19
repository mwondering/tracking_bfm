# Action Trunk Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated G1 tracking task whose policy outputs a 4-slice action trunk (`29 * 4 = 116`) while the environment applies one 29-D slice per decimation substep.

**Architecture:** Add optional trunk support at the framework level with `action_trunk_len=1` as the default, so existing tasks remain unchanged. The new task sets `action_trunk_len=4`, the wrapper exposes `policy_action_dim`, and `ManagerBasedRlEnv.step()` passes the current decimation substep to `ActionManager.apply_action()` only when trunk mode is enabled. Only `action_rate_l2` changes reward semantics: it penalizes actual executed substep action differences instead of treating the trunk as one flat vector.

**Tech Stack:** Python dataclasses, PyTorch tensors, MuJoCo env loop, rsl_rl `VecEnv`, pytest.

---

## File Structure

- Modify `src/mjlab/envs/manager_based_rl_env.py`: add `action_trunk_len` config, validate it, use policy action dim for action spaces, and pass `substep_idx` inside `step()` only for trunk-enabled tasks.
- Modify `src/mjlab/managers/action_manager.py`: track both base action dimension and policy action dimension, store flat trunk history, select substep slices, and expose the currently applied 29-D slice.
- Modify `src/mjlab/envs/mdp/rewards.py`: update only `action_rate_l2` to compute smoothness over the executed substep sequence.
- Modify `src/mjlab/rl/vecenv_wrapper.py`: expose `num_actions` as policy action dimension.
- Modify `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`: add the action-trunk G1 BFM task config builder.
- Modify `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`: add an action-trunk PPO runner config with a separate experiment name.
- Modify `src/mjlab/tasks/tracking/config/g1/__init__.py`: register the new task ID.
- Modify `tests/test_action_manager.py`: cover trunk shape validation, substep slice routing, and reset behavior.
- Modify `tests/test_tracking_task.py`: assert the new task is registered and keeps the expected G1 action scale.
- Add `tests/test_action_trunk.py`: cover env config/action space integration and trunk-aware `action_rate_l2`.

---

### Task 1: Add Config and Action Space Plumbing

**Files:**
- Modify: `src/mjlab/envs/manager_based_rl_env.py`
- Modify: `src/mjlab/rl/vecenv_wrapper.py`
- Test: `tests/test_action_trunk.py`

- [ ] **Step 1: Write the failing config/action-space tests**

Add `tests/test_action_trunk.py` with:

```python
"""Tests for action trunk support."""

from unittest.mock import Mock

import pytest
import torch

import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, list_tasks


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


def test_action_trunk_len_must_match_decimation_for_trunk_mode() -> None:
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=Mock(),
    action_trunk_len=2,
  )
  assert cfg.action_trunk_len == 2
```

The last test intentionally only checks dataclass construction. The runtime validation belongs in `ManagerBasedRlEnv.__init__`, because `scene` is a mock and this test should not construct MuJoCo state.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_action_trunk.py::test_action_trunk_task_config_uses_four_slices tests/test_action_trunk.py::test_action_manager_policy_dim_expands_with_trunk_len -v
```

Expected: FAIL because `action_trunk_len`, `policy_action_dim`, and the task registration do not exist.

- [ ] **Step 3: Add `action_trunk_len` to env config and action spaces**

In `src/mjlab/envs/manager_based_rl_env.py`, add this field to `ManagerBasedRlEnvCfg` after `decimation`:

```python
  action_trunk_len: int = 1
  """Number of per-substep action slices emitted by one policy step.

  The default value of 1 preserves standard action repeat behavior: one policy
  action is processed once and applied on every decimation substep. When greater
  than 1, the policy action is interpreted as a flat trunk of
  ``action_trunk_len * base_action_dim`` values. Trunk mode requires
  ``action_trunk_len == decimation`` so each physics substep receives exactly one
  action slice.
  """
```

In `ManagerBasedRlEnv.__init__`, immediately after `self.cfg = cfg`, add:

```python
    if self.cfg.action_trunk_len < 1:
      raise ValueError("action_trunk_len must be >= 1.")
    if self.cfg.action_trunk_len > 1 and self.cfg.action_trunk_len != self.cfg.decimation:
      raise ValueError(
        "action_trunk_len must equal decimation when trunk mode is enabled. "
        f"Received action_trunk_len={self.cfg.action_trunk_len}, "
        f"decimation={self.cfg.decimation}."
      )
```

In `_configure_gym_env_spaces()`, replace:

```python
    action_dim = sum(self.action_manager.action_term_dim)
```

with:

```python
    action_dim = self.action_manager.policy_action_dim
```

- [ ] **Step 4: Update rsl_rl wrapper action dimension**

In `src/mjlab/rl/vecenv_wrapper.py`, replace:

```python
    self.num_actions = self.unwrapped.action_manager.total_action_dim
```

with:

```python
    self.num_actions = self.unwrapped.action_manager.policy_action_dim
```

- [ ] **Step 5: Run existing wrapper/action-space tests**

Run:

```bash
uv run pytest tests/test_runner.py::test_runner_persists_common_step_counter tests/test_action_trunk.py::test_action_manager_policy_dim_expands_with_trunk_len -v
```

Expected: PASS after Task 2 adds `policy_action_dim`; until then the action-trunk test still fails.

- [ ] **Step 6: Commit**

```bash
git add src/mjlab/envs/manager_based_rl_env.py src/mjlab/rl/vecenv_wrapper.py tests/test_action_trunk.py
git commit -m "feat: add action trunk config plumbing"
```

---

### Task 2: Implement Trunk-Aware ActionManager

**Files:**
- Modify: `src/mjlab/managers/action_manager.py`
- Modify: `tests/test_action_manager.py`
- Test: `tests/test_action_manager.py`

- [ ] **Step 1: Update the mock env fixture for trunk defaults**

In `tests/test_action_manager.py`, update the `mock_env` fixture:

```python
@pytest.fixture
def mock_env(device):
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.cfg = Mock(action_trunk_len=1, decimation=1)
  return env
```

- [ ] **Step 2: Add failing substep routing test**

Append to `tests/test_action_manager.py`:

```python
def test_action_trunk_routes_substep_slices(device, action_term_cfg):
  """Action trunk mode routes one base action slice per substep."""
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.cfg = Mock(action_trunk_len=4, decimation=4)
  manager = ActionManager({"action": action_term_cfg}, env)
  term = manager.get_term("action")

  action = torch.arange(2 * 12, dtype=torch.float32, device=device).reshape(2, 12)

  manager.process_action(action)
  torch.testing.assert_close(manager.applied_action, action[:, 0:3])
  torch.testing.assert_close(term.process_actions.call_args.args[0], action[:, 0:3])

  manager.apply_action(substep_idx=2)
  torch.testing.assert_close(manager.applied_action, action[:, 6:9])
  torch.testing.assert_close(term.process_actions.call_args.args[0], action[:, 6:9])

  manager.apply_action(substep_idx=3)
  torch.testing.assert_close(manager.applied_action, action[:, 9:12])
  torch.testing.assert_close(term.process_actions.call_args.args[0], action[:, 9:12])
```

- [ ] **Step 3: Add failing shape validation test**

Append:

```python
def test_action_trunk_validates_policy_action_shape(device, action_term_cfg):
  """Trunk mode validates policy action dimension, not base action dimension."""
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.cfg = Mock(action_trunk_len=4, decimation=4)
  manager = ActionManager({"action": action_term_cfg}, env)

  with pytest.raises(ValueError, match="expected: 12"):
    manager.process_action(torch.zeros(2, 3, device=device))
```

- [ ] **Step 4: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_action_manager.py::test_action_trunk_routes_substep_slices tests/test_action_manager.py::test_action_trunk_validates_policy_action_shape -v
```

Expected: FAIL because `policy_action_dim`, `applied_action`, and `apply_action(substep_idx=...)` are not implemented.

- [ ] **Step 5: Implement policy/base action dimensions and buffers**

In `src/mjlab/managers/action_manager.py`, in `__init__`, replace the buffer block with:

```python
    self._action_trunk_len = int(getattr(env.cfg, "action_trunk_len", 1))

    # Flat policy output history. In trunk mode this is
    # (num_envs, action_trunk_len * total_action_dim).
    self._action = torch.zeros(
      (self.num_envs, self.policy_action_dim), device=self.device
    )
    self._prev_action = torch.zeros_like(self._action)
    self._prev_prev_action = torch.zeros_like(self._action)

    # The base action slice currently routed to action terms.
    self._applied_action = torch.zeros(
      (self.num_envs, self.total_action_dim), device=self.device
    )
```

Add these properties after `total_action_dim`:

```python
  @property
  def action_trunk_len(self) -> int:
    return self._action_trunk_len

  @property
  def policy_action_dim(self) -> int:
    return self.total_action_dim * self.action_trunk_len

  @property
  def applied_action(self) -> torch.Tensor:
    """Base action slice currently applied to action terms."""
    return self._applied_action

  @property
  def action_sequence(self) -> torch.Tensor:
    """Current policy output as ``(num_envs, trunk_len, base_action_dim)``."""
    return self._action.view(self.num_envs, self.action_trunk_len, self.total_action_dim)

  @property
  def prev_action_sequence(self) -> torch.Tensor:
    """Previous policy output as ``(num_envs, trunk_len, base_action_dim)``."""
    return self._prev_action.view(
      self.num_envs, self.action_trunk_len, self.total_action_dim
    )
```

- [ ] **Step 6: Implement substep slice routing**

In `src/mjlab/managers/action_manager.py`, replace `process_action()` with:

```python
  def process_action(self, action: torch.Tensor) -> None:
    """Store the raw policy output and route the first substep slice."""
    if self.policy_action_dim != action.shape[1]:
      raise ValueError(
        f"Invalid action shape, expected: {self.policy_action_dim},"
        f" received: {action.shape[1]}."
      )
    self._prev_prev_action[:] = self._prev_action
    self._prev_action[:] = self._action
    self._action[:] = action.to(self.device)
    self._process_substep_action(0)
```

Add this private helper below `process_action()`:

```python
  def _process_substep_action(self, substep_idx: int) -> None:
    if substep_idx < 0 or substep_idx >= self.action_trunk_len:
      raise ValueError(
        f"substep_idx must be in [0, {self.action_trunk_len}); "
        f"received {substep_idx}."
      )
    base_action = self.action_sequence[:, substep_idx, :]
    self._applied_action[:] = base_action

    idx = 0
    for term in self._terms.values():
      term_actions = base_action[:, idx : idx + term.action_dim]
      term.process_actions(term_actions)
      idx += term.action_dim
```

Replace `apply_action()` with:

```python
  def apply_action(self, substep_idx: int | None = None) -> None:
    """Write processed actions to entity actuator targets.

    If ``substep_idx`` is provided, first route that trunk slice to each action
    term. Existing callers may omit it; they will apply the most recently
    processed slice.
    """
    if substep_idx is not None:
      self._process_substep_action(substep_idx)
    for term in self._terms.values():
      term.apply_actions()
```

- [ ] **Step 7: Update reset to clear applied action**

In `reset()`, after `self._action[env_ids] = 0.0`, add:

```python
    self._applied_action[env_ids] = 0.0
```

- [ ] **Step 8: Run action manager tests**

Run:

```bash
uv run pytest tests/test_action_manager.py tests/test_action_trunk.py::test_action_manager_policy_dim_expands_with_trunk_len -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/mjlab/managers/action_manager.py tests/test_action_manager.py tests/test_action_trunk.py
git commit -m "feat: route action trunk substep slices"
```

---

### Task 3: Apply Trunk Slices in the Env Step Loop

**Files:**
- Modify: `src/mjlab/envs/manager_based_rl_env.py`
- Test: `tests/test_action_trunk.py`

- [ ] **Step 1: Add a focused step-loop test with a fake action manager**

Append to `tests/test_action_trunk.py`:

```python
def test_step_loop_passes_substep_indices_to_action_manager_in_trunk_mode() -> None:
  """The decimation loop applies one trunk slice per physics substep."""
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  env = object.__new__(ManagerBasedRlEnv)
  env.cfg = Mock(
    decimation=4,
    action_trunk_len=4,
    auto_reset=True,
    is_finite_horizon=False,
  )
  env.device = "cpu"
  env._manual_reset_pending = torch.zeros(1, dtype=torch.bool)
  env._sim_step_counter = 0
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)
  env.common_step_counter = 0
  env.extras = {}

  env.action_manager = Mock()
  env.action_manager.apply_action = Mock()
  env.scene = Mock()
  env.sim = Mock()
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

  env.step(torch.zeros(1, 12))

  assert [call.kwargs["substep_idx"] for call in env.action_manager.apply_action.call_args_list] == [
    0,
    1,
    2,
    3,
  ]


def test_step_loop_preserves_action_repeat_for_standard_tracking() -> None:
  """Default action mode keeps the old process-once/apply-repeat behavior."""
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  env = object.__new__(ManagerBasedRlEnv)
  env.cfg = Mock(
    decimation=4,
    action_trunk_len=1,
    auto_reset=True,
    is_finite_horizon=False,
  )
  env.device = "cpu"
  env._manual_reset_pending = torch.zeros(1, dtype=torch.bool)
  env._sim_step_counter = 0
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)
  env.common_step_counter = 0
  env.extras = {}

  env.action_manager = Mock()
  env.action_manager.apply_action = Mock()
  env.scene = Mock()
  env.sim = Mock()
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

  env.step(torch.zeros(1, 3))

  assert env.action_manager.process_action.call_count == 1
  assert env.action_manager.apply_action.call_count == 4
  assert all(call.kwargs == {} for call in env.action_manager.apply_action.call_args_list)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_action_trunk.py::test_step_loop_passes_substep_indices_to_action_manager_in_trunk_mode tests/test_action_trunk.py::test_step_loop_preserves_action_repeat_for_standard_tracking -v
```

Expected: FAIL because trunk mode does not pass `substep_idx` yet, and standard mode protection is not implemented yet.

- [ ] **Step 3: Pass `substep_idx` in the decimation loop**

In `src/mjlab/envs/manager_based_rl_env.py`, replace:

```python
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
```

with:

```python
    for substep_idx in range(self.cfg.decimation):
      self._sim_step_counter += 1
      if self.cfg.action_trunk_len > 1:
        self.action_manager.apply_action(substep_idx=substep_idx)
      else:
        self.action_manager.apply_action()
```

- [ ] **Step 4: Run step-loop and existing env tests**

Run:

```bash
uv run pytest tests/test_action_trunk.py::test_step_loop_passes_substep_indices_to_action_manager_in_trunk_mode tests/test_action_trunk.py::test_step_loop_preserves_action_repeat_for_standard_tracking tests/test_runner.py::test_runner_persists_common_step_counter -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab/envs/manager_based_rl_env.py tests/test_action_trunk.py
git commit -m "feat: apply action trunk slices during decimation"
```

---

### Task 4: Make `action_rate_l2` Trunk-Aware

**Files:**
- Modify: `src/mjlab/envs/mdp/rewards.py`
- Test: `tests/test_action_trunk.py`

- [ ] **Step 1: Add failing reward test**

Append to `tests/test_action_trunk.py`:

```python
def test_action_rate_l2_penalizes_executed_trunk_sequence() -> None:
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

  expected_env0 = (2.0 - 1.0) ** 2 + (4.0 - 2.0) ** 2 + (7.0 - 4.0) ** 2 + (11.0 - 7.0) ** 2
  expected_env1 = (2.0 - 1.0) ** 2 + (2.0 - 2.0) ** 2 + (2.0 - 2.0) ** 2 + (2.0 - 2.0) ** 2
  expected = torch.tensor([expected_env0, expected_env1])

  torch.testing.assert_close(result, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_action_trunk.py::test_action_rate_l2_penalizes_executed_trunk_sequence -v
```

Expected: FAIL because the current implementation compares flat current trunk against flat previous trunk.

- [ ] **Step 3: Update only `action_rate_l2`**

In `src/mjlab/envs/mdp/rewards.py`, replace `action_rate_l2()` with:

```python
def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize actual executed action changes.

  For standard action repeat this is identical to the old behavior:
  ``current_action - previous_action``. In trunk mode, the penalty follows the
  executed substep sequence from the previous trunk's final slice into every
  current trunk slice.
  """
  action_manager = env.action_manager
  if getattr(action_manager, "action_trunk_len", 1) == 1:
    return torch.sum(
      torch.square(action_manager.action - action_manager.prev_action), dim=1
    )

  prev_tail = action_manager.prev_action_sequence[:, -1:, :]
  current_sequence = action_manager.action_sequence
  executed_sequence = torch.cat((prev_tail, current_sequence), dim=1)
  action_rate = executed_sequence[:, 1:, :] - executed_sequence[:, :-1, :]
  return torch.sum(torch.square(action_rate), dim=(1, 2))
```

- [ ] **Step 4: Run reward tests**

Run:

```bash
uv run pytest tests/test_action_trunk.py::test_action_rate_l2_penalizes_executed_trunk_sequence tests/test_rewards.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab/envs/mdp/rewards.py tests/test_action_trunk.py
git commit -m "fix: compute action rate over trunk execution sequence"
```

---

### Task 5: Register the New G1 Action-Trunk Task

**Files:**
- Modify: `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/__init__.py`
- Modify: `tests/test_tracking_task.py`
- Test: `tests/test_tracking_task.py`, `tests/test_action_trunk.py`

- [ ] **Step 1: Add failing tracking task test**

In `tests/test_tracking_task.py`, append:

```python
def test_tracking_bfm_action_trunk_task_config() -> None:
  """The action-trunk tracking task should expose a 4-slice policy action."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk")

  assert cfg.action_trunk_len == 4
  assert cfg.decimation == 4
  assert "joint_pos" in cfg.actions
  assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
  assert cfg.actions["joint_pos"].scale == G1_ACTION_SCALE
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_tracking_task.py::test_tracking_bfm_action_trunk_task_config -v
```

Expected: FAIL because the task is not registered.

- [ ] **Step 3: Add the env cfg builder**

In `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`, after `unitree_g1_flat_tracking_bfm_env_cfg()`, add:

```python
def unitree_g1_flat_tracking_bfm_action_trunk_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the multi-motion Unitree G1 tracking task with 4-slice action trunk."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)
  cfg.action_trunk_len = cfg.decimation
  return cfg
```

- [ ] **Step 4: Add the RL cfg builder**

In `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`, after `unitree_g1_trackingbfm_ppo_runner_cfg()`, add:

```python
def unitree_g1_trackingbfm_action_trunk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the G1 BFM action-trunk task."""
  cfg = unitree_g1_trackingbfm_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_action_trunk"
  return cfg
```

- [ ] **Step 5: Register the new task**

In `src/mjlab/tasks/tracking/config/g1/__init__.py`, update imports:

```python
from .env_cfgs import (
  unitree_g1_flat_tracking_bfm_1stage_env_cfg,
  unitree_g1_flat_tracking_bfm_action_trunk_env_cfg,
  unitree_g1_flat_tracking_bfm_env_cfg,
  unitree_g1_flat_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_trackingbfm_action_trunk_ppo_runner_cfg,
  unitree_g1_trackingbfm_ppo_runner_cfg,
)
```

Append:

```python
register_mjlab_task(
  task_id="Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk",
  env_cfg=unitree_g1_flat_tracking_bfm_action_trunk_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_action_trunk_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_action_trunk_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
```

- [ ] **Step 6: Run task registration tests**

Run:

```bash
uv run pytest tests/test_action_trunk.py::test_action_trunk_task_is_registered tests/test_tracking_task.py::test_tracking_bfm_action_trunk_task_config -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mjlab/tasks/tracking/config/g1/env_cfgs.py src/mjlab/tasks/tracking/config/g1/rl_cfg.py src/mjlab/tasks/tracking/config/g1/__init__.py tests/test_tracking_task.py tests/test_action_trunk.py
git commit -m "feat: register g1 action trunk tracking task"
```

---

### Task 6: End-to-End Verification

**Files:**
- No source changes expected.
- Test: `tests/test_action_manager.py`, `tests/test_action_trunk.py`, `tests/test_tracking_task.py`, `tests/test_runner.py`

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/test_action_manager.py tests/test_action_trunk.py tests/test_tracking_task.py tests/test_runner.py::test_runner_persists_common_step_counter -v
```

Expected: PASS.

- [ ] **Step 2: Run task listing smoke check**

Run:

```bash
uv run list_envs
```

Expected: output includes `Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk`.

- [ ] **Step 3: Run one short train construction smoke test**

Run:

```bash
uv run train Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk --agent.max_iterations 1 --agent.num_steps_per_env 2 --env.scene.num_envs 2 --headless
```

Expected: runner starts, constructs actor with 116 output actions, performs one iteration, and exits without action shape errors.

- [ ] **Step 4: Commit if verification required generated no source changes**

```bash
git status --short
```

Expected: clean working tree. If generated log files appear under ignored paths, no commit is needed.

---

## Self-Review

- Spec coverage: The plan creates a new isolated task, keeps framework defaults compatible, changes the step loop to execute one trunk slice per decimation substep only in trunk mode, preserves standard action repeat for existing tracking tasks, and modifies only `action_rate_l2` among rewards.
- Placeholder scan: No unresolved placeholders or open-ended implementation instructions remain.
- Type consistency: `action_trunk_len`, `policy_action_dim`, `applied_action`, `action_sequence`, and `prev_action_sequence` are introduced before later tasks use them.
- Deliberate non-goals: No changes to other reward terms, no distillation label changes, no ONNX/deployment adaptation beyond normal policy output shape. Those can be separate tasks after the training experiment proves useful.
