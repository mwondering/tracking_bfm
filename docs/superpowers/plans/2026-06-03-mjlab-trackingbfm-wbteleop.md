# Mjlab Tracking BFM Wbteleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop`, a G1 BFM tracking task whose actor uses non-sparse non-privileged observations and trains with PPO plus teacher-action MSE.

**Architecture:** Add a self-contained `src/mjlab/tasks/tracking/wbteleop/` package. Compose the existing G1 BFM tracking env, replace the actor observation, keep privileged critic and teacher-only observations, and use a local PPO variant that adds a scheduled teacher-action MSE term.

**Tech Stack:** Python, dataclasses, PyTorch, TensorDict, rsl_rl PPO, MJLab task registry, pytest.

---

## File Structure

Create:

- `src/mjlab/tasks/tracking/wbteleop/__init__.py`: registers `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop`.
- `src/mjlab/tasks/tracking/wbteleop/env_cfg.py`: builds wbteleop environment config.
- `src/mjlab/tasks/tracking/wbteleop/rl_cfg.py`: defines PPO runner and BC config defaults.
- `src/mjlab/tasks/tracking/wbteleop/observations.py`: defines `motion_ref_ang_vel`.
- `src/mjlab/tasks/tracking/wbteleop/algorithm.py`: defines `cosine_bc_weight` and `WbTeleopPPO`.
- `src/mjlab/tasks/tracking/wbteleop/runner.py`: defines `WbTeleopTrackingRunner`.
- `tests/test_wbteleop_task.py`: verifies task registration, observation structure, history support, schedule, and algorithm smoke behavior.

Do not modify the installed rsl_rl package. Do not change existing tracking or distillation task behavior.

---

### Task 1: Add Failing Wbteleop Task Config Tests

**Files:**
- Create: `tests/test_wbteleop_task.py`

- [ ] **Step 1: Write failing tests for registration, observations, and history**

Create `tests/test_wbteleop_task.py` with:

```python
"""Tests for the G1 BFM wbteleop tracking task."""

from __future__ import annotations

from dataclasses import asdict

import pytest

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
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
```

- [ ] **Step 2: Run tests and verify they fail because wbteleop does not exist**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py -q
```

Expected: fail during import with `ModuleNotFoundError: No module named 'mjlab.tasks.tracking.wbteleop'`.

- [ ] **Step 3: Commit failing tests**

Run:

```bash
git add tests/test_wbteleop_task.py
git commit -m "test: add wbteleop task config expectations"
```

---

### Task 2: Implement Wbteleop Observations, Env Config, RL Config, and Registration

**Files:**
- Create: `src/mjlab/tasks/tracking/wbteleop/__init__.py`
- Create: `src/mjlab/tasks/tracking/wbteleop/env_cfg.py`
- Create: `src/mjlab/tasks/tracking/wbteleop/rl_cfg.py`
- Create: `src/mjlab/tasks/tracking/wbteleop/observations.py`
- Create: `src/mjlab/tasks/tracking/wbteleop/runner.py`

- [ ] **Step 1: Implement `observations.py`**

Create `src/mjlab/tasks/tracking/wbteleop/observations.py`:

```python
"""Observation terms for the G1 BFM wbteleop tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.multi_commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_ref_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return reference anchor angular velocity from the motion command window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.anchor_ang_vel_w
```

- [ ] **Step 2: Implement `env_cfg.py`**

Create `src/mjlab/tasks/tracking/wbteleop/env_cfg.py`:

```python
"""Environment config for G1 BFM wbteleop tracking."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_bfm_env_cfg,
)
from mjlab.tasks.tracking.mdp.multi_commands import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import observations as wbteleop_observations


def _robot_history_length(history_steps: int) -> int:
  history_steps = int(history_steps)
  if history_steps <= 0:
    return 0
  return history_steps + 1


def _history_kwargs(history_steps: int) -> dict[str, int]:
  history_length = _robot_history_length(history_steps)
  if history_length <= 0:
    return {}
  return {"history_length": history_length}


def _wbteleop_actor_cfg(
  *,
  history_steps: int,
  enable_corruption: bool,
) -> ObservationGroupCfg:
  robot_history = _history_kwargs(history_steps)
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "motion"},
      ),
      "motion_ref_ang_vel": ObservationTermCfg(
        func=wbteleop_observations.motion_ref_ang_vel,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.05, n_max=0.05),
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        **robot_history,
      ),
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        **robot_history,
      ),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        **robot_history,
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        **robot_history,
      ),
      "actions": ObservationTermCfg(
        func=mdp.last_action,
        **robot_history,
      ),
    },
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(
  *,
  history_steps: int = 0,
  future_steps: int = 1,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 BFM wbteleop tracking environment config."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.history_steps = int(history_steps)
  motion_cmd.future_steps = int(future_steps)

  teacher_actor = deepcopy(cfg.observations["actor"])
  teacher_actor.enable_corruption = False
  cfg.observations["teacher_actor"] = teacher_actor
  cfg.observations["actor"] = _wbteleop_actor_cfg(
    history_steps=motion_cmd.history_steps,
    enable_corruption=not play,
  )
  return cfg
```

- [ ] **Step 3: Implement `rl_cfg.py`**

Create `src/mjlab/tasks/tracking/wbteleop/rl_cfg.py`:

```python
"""RL config for G1 BFM wbteleop tracking."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class WbTeleopPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "mjlab.tasks.tracking.wbteleop.algorithm:WbTeleopPPO"
  teacher_task_id: str = "Mjlab-Trackingbfm-Flat-Unitree-G1"
  teacher_checkpoint_path: str = ""
  teacher_obs_group: str = "teacher_actor"
  bc_weight_start: float = 0.5
  bc_weight_end: float = 0.1
  bc_decay_steps: int = 10_000


def unitree_g1_trackingbfm_wbteleop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the runner config for the G1 BFM wbteleop task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=WbTeleopPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="g1_tracking_wbteleop",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=300_000,
  )
```

- [ ] **Step 4: Add a temporary runner shell**

Create `src/mjlab/tasks/tracking/wbteleop/runner.py`:

```python
"""Runner for G1 BFM wbteleop tracking."""

from __future__ import annotations

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class WbTeleopTrackingRunner(MotionTrackingOnPolicyRunner):
  """Tracking runner for wbteleop PPO plus teacher-action BC."""
```

- [ ] **Step 5: Register the task**

Create `src/mjlab/tasks/tracking/wbteleop/__init__.py`:

```python
"""G1 BFM wbteleop tracking task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import unitree_g1_flat_tracking_bfm_wbteleop_env_cfg
from .rl_cfg import unitree_g1_trackingbfm_wbteleop_ppo_runner_cfg
from .runner import WbTeleopTrackingRunner


register_mjlab_task(
  task_id="Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop",
  env_cfg=unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_wbteleop_ppo_runner_cfg(),
  runner_cls=WbTeleopTrackingRunner,
)
```

- [ ] **Step 6: Run config tests**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py -q
```

Expected: config tests pass, except tests that later import `algorithm.py` may still fail if added in later tasks.

- [ ] **Step 7: Run generic task integrity tests**

Run:

```bash
uv run pytest tests/test_task_configs.py -q
```

Expected: pass.

- [ ] **Step 8: Commit env and config implementation**

Run:

```bash
git add src/mjlab/tasks/tracking/wbteleop tests/test_wbteleop_task.py
git commit -m "feat: register wbteleop tracking task"
```

---

### Task 3: Add BC Weight Schedule Tests and Implementation

**Files:**
- Modify: `tests/test_wbteleop_task.py`
- Create: `src/mjlab/tasks/tracking/wbteleop/algorithm.py`

- [ ] **Step 1: Add failing schedule tests**

Append to `tests/test_wbteleop_task.py`:

```python
from mjlab.tasks.tracking.wbteleop.algorithm import cosine_bc_weight


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
```

- [ ] **Step 2: Run schedule tests and verify they fail**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py::test_wbteleop_bc_weight_schedule_values tests/test_wbteleop_task.py::test_wbteleop_bc_weight_schedule_rejects_invalid_decay -q
```

Expected: fail with `ModuleNotFoundError` or missing `cosine_bc_weight`.

- [ ] **Step 3: Implement schedule and algorithm class constructor**

Create `src/mjlab/tasks/tracking/wbteleop/algorithm.py`:

```python
"""PPO variant for wbteleop tracking with teacher-action MSE."""

from __future__ import annotations

import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter


def cosine_bc_weight(
  iteration: int,
  *,
  start: float,
  end: float,
  decay_steps: int,
) -> float:
  """Cosine decay from start to end, clamped at end after decay_steps."""
  if decay_steps <= 0:
    raise ValueError("bc_decay_steps must be positive")
  progress = min(max(int(iteration), 0), int(decay_steps)) / float(decay_steps)
  return float(end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class WbTeleopPPO(PPO):
  """PPO with an additional scheduled teacher-action MSE loss."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    *args,
    teacher_task_id: str = "Mjlab-Trackingbfm-Flat-Unitree-G1",
    teacher_checkpoint_path: str = "",
    teacher_obs_group: str = "teacher_actor",
    bc_weight_start: float = 0.5,
    bc_weight_end: float = 0.1,
    bc_decay_steps: int = 10_000,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.teacher_task_id = teacher_task_id
    self.teacher_checkpoint_path = teacher_checkpoint_path
    self.teacher_obs_group = teacher_obs_group
    self.bc_weight_start = float(bc_weight_start)
    self.bc_weight_end = float(bc_weight_end)
    self.bc_decay_steps = int(bc_decay_steps)
    self.teacher_adapter: TeacherPolicyAdapter | None = None
    self.current_learning_iteration = 0

  def set_teacher_adapter(self, teacher_adapter: TeacherPolicyAdapter) -> None:
    self.teacher_adapter = teacher_adapter

  def set_learning_iteration(self, iteration: int) -> None:
    self.current_learning_iteration = int(iteration)

  def _current_bc_weight(self) -> float:
    return cosine_bc_weight(
      self.current_learning_iteration,
      start=self.bc_weight_start,
      end=self.bc_weight_end,
      decay_steps=self.bc_decay_steps,
    )
```

- [ ] **Step 4: Run schedule tests**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py::test_wbteleop_bc_weight_schedule_values tests/test_wbteleop_task.py::test_wbteleop_bc_weight_schedule_rejects_invalid_decay -q
```

Expected: pass.

- [ ] **Step 5: Commit schedule implementation**

Run:

```bash
git add src/mjlab/tasks/tracking/wbteleop/algorithm.py tests/test_wbteleop_task.py
git commit -m "feat: add wbteleop bc weight schedule"
```

---

### Task 4: Implement WbTeleopPPO Update With Teacher MSE

**Files:**
- Modify: `src/mjlab/tasks/tracking/wbteleop/algorithm.py`
- Modify: `tests/test_wbteleop_task.py`

- [ ] **Step 1: Add a small algorithm smoke test**

Append to `tests/test_wbteleop_task.py`:

```python
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter
from mjlab.tasks.tracking.wbteleop.algorithm import WbTeleopPPO


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
```

- [ ] **Step 2: Run smoke test and verify it fails**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py::test_wbteleop_ppo_update_reports_bc_metrics -q
```

Expected: fail because `WbTeleopPPO.update()` still uses base PPO and does not report BC metrics.

- [ ] **Step 3: Implement `WbTeleopPPO.update()`**

In `src/mjlab/tasks/tracking/wbteleop/algorithm.py`, copy the installed rsl_rl `PPO.update()` body from:

```text
.venv/lib/python3.13/site-packages/rsl_rl/algorithms/ppo.py
```

Then apply these exact semantic changes in the local copy:

1. Initialize BC accumulators near the existing loss accumulators:

```python
mean_bc_mse = 0.0
mean_bc_loss = 0.0
bc_weight = self._current_bc_weight()
```

2. After the base PPO loss is computed and after optional symmetry/RND loss blocks are prepared, compute teacher MSE:

```python
if self.teacher_adapter is None:
  raise ValueError("teacher_adapter must be set before WbTeleopPPO.update()")

student_action_mean = self.actor(batch.observations[:original_batch_size])
with torch.no_grad():
  teacher_action_mean = self.teacher_adapter.act_mean(
    batch.observations[:original_batch_size]
  )
if student_action_mean.shape != teacher_action_mean.shape:
  raise ValueError(
    "Teacher and student action shapes must match: "
    f"student={tuple(student_action_mean.shape)}, "
    f"teacher={tuple(teacher_action_mean.shape)}"
  )
bc_mse = torch.nn.functional.mse_loss(student_action_mean, teacher_action_mean)
bc_loss = bc_weight * bc_mse
loss = loss + bc_loss
```

3. Accumulate BC metrics after the optimizer step:

```python
mean_bc_mse += bc_mse.item()
mean_bc_loss += bc_loss.item()
```

4. Divide BC metrics by `num_updates` with the other losses:

```python
mean_bc_mse /= num_updates
mean_bc_loss /= num_updates
```

5. Add metrics to `loss_dict`:

```python
loss_dict.update(
  {
    "bc_mse": mean_bc_mse,
    "bc_weight": float(bc_weight),
    "bc_loss": mean_bc_loss,
  }
)
```

6. Keep `self.storage.clear()` behavior identical to base PPO.

7. Keep RND, symmetry, multi-GPU gradient reduction, adaptive KL schedule, recurrent batch handling, and normalization behavior from the base PPO copy unchanged.

- [ ] **Step 4: Run smoke test**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py::test_wbteleop_ppo_update_reports_bc_metrics -q
```

Expected: pass.

- [ ] **Step 5: Run existing PPO-adjacent tests**

Run:

```bash
uv run pytest tests/test_runner.py tests/test_distillation_algorithm.py tests/test_wbteleop_task.py -q
```

Expected: pass.

- [ ] **Step 6: Commit algorithm update**

Run:

```bash
git add src/mjlab/tasks/tracking/wbteleop/algorithm.py tests/test_wbteleop_task.py
git commit -m "feat: add teacher mse to wbteleop ppo"
```

---

### Task 5: Implement Teacher Loading in WbTeleop Runner

**Files:**
- Modify: `src/mjlab/tasks/tracking/wbteleop/runner.py`
- Modify: `tests/test_wbteleop_task.py`

- [ ] **Step 1: Add runner teacher loading tests**

Append to `tests/test_wbteleop_task.py`:

```python
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch


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
    patch("mjlab.tasks.tracking.wbteleop.runner.load_rl_cfg", return_value=load_rl_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1")),
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
```

- [ ] **Step 2: Run teacher loading tests and verify they fail**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py::test_wbteleop_runner_builds_teacher_adapter tests/test_wbteleop_task.py::test_wbteleop_runner_rejects_missing_teacher_checkpoint -q
```

Expected: fail because runner methods are not implemented.

- [ ] **Step 3: Implement runner teacher loading**

Replace `src/mjlab/tasks/tracking/wbteleop/runner.py` with:

```python
"""Runner for G1 BFM wbteleop tracking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import os

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class WbTeleopTrackingRunner(MotionTrackingOnPolicyRunner):
  """Tracking runner for wbteleop PPO plus teacher-action BC."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device, registry_name=registry_name)
    self.teacher_adapter: TeacherPolicyAdapter | None = None

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    if self.teacher_adapter is None:
      self.teacher_adapter = self._build_teacher_adapter()
    self.alg.set_teacher_adapter(self.teacher_adapter)
    return super().learn(num_learning_iterations, init_at_random_ep_len)

  def _begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    if hasattr(self.alg, "set_learning_iteration"):
      self.alg.set_learning_iteration(iteration)
    super()._begin_adaptive_sampling_iteration(iteration)

  def _build_teacher_adapter(self) -> TeacherPolicyAdapter:
    algorithm_cfg = self.cfg.get("algorithm", {})
    checkpoint_path = algorithm_cfg.get("teacher_checkpoint_path", "")
    if not checkpoint_path:
      raise ValueError(
        "teacher_checkpoint_path must be provided for wbteleop training"
      )

    teacher_task_id = algorithm_cfg.get(
      "teacher_task_id",
      "Mjlab-Trackingbfm-Flat-Unitree-G1",
    )
    teacher_obs_group = algorithm_cfg.get("teacher_obs_group", "teacher_actor")
    teacher_runner_cls = load_runner_cls(teacher_task_id) or MjlabOnPolicyRunner
    teacher_cfg = asdict(load_rl_cfg(teacher_task_id))
    teacher_cfg["obs_groups"]["actor"] = (teacher_obs_group,)

    common_step_counter = getattr(self.env.unwrapped, "common_step_counter", None)
    with self._suppress_distributed_env_for_nested_runner():
      teacher_runner = teacher_runner_cls(
        self.env,
        teacher_cfg,
        log_dir=None,
        device=str(self.device),
      )
    teacher_runner.load(checkpoint_path, map_location=str(self.device))
    if common_step_counter is not None:
      self.env.unwrapped.common_step_counter = common_step_counter

    return TeacherPolicyAdapter(
      teacher_runner.get_inference_policy(device=self.device),
      obs_group=teacher_obs_group,
      policy_input_key=teacher_obs_group,
    )

  @contextmanager
  def _suppress_distributed_env_for_nested_runner(self):
    keys = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    old_values = {key: os.environ.get(key) for key in keys}
    for key in keys:
      os.environ.pop(key, None)
    try:
      yield
    finally:
      for key, value in old_values.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value
```

- [ ] **Step 4: Run teacher loading tests**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py::test_wbteleop_runner_builds_teacher_adapter tests/test_wbteleop_task.py::test_wbteleop_runner_rejects_missing_teacher_checkpoint -q
```

Expected: pass.

- [ ] **Step 5: Commit runner implementation**

Run:

```bash
git add src/mjlab/tasks/tracking/wbteleop/runner.py tests/test_wbteleop_task.py
git commit -m "feat: load wbteleop teacher policy"
```

---

### Task 6: Add End-to-End CLI and Regression Verification

**Files:**
- Modify: `tests/test_wbteleop_task.py`

- [ ] **Step 1: Add CLI help smoke test**

Append to `tests/test_wbteleop_task.py`:

```python
import subprocess


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
```

- [ ] **Step 2: Run wbteleop tests**

Run:

```bash
uv run pytest tests/test_wbteleop_task.py -q
```

Expected: pass.

- [ ] **Step 3: Run task config regression tests**

Run:

```bash
uv run pytest tests/test_task_configs.py tests/test_tracking_task.py tests/test_distillation_task.py -q
```

Expected: pass.

- [ ] **Step 4: Run a broader targeted regression set**

Run:

```bash
uv run pytest \
  tests/test_runner.py \
  tests/test_distillation_runner_smoke.py \
  tests/test_distillation_teacher_adapter.py \
  tests/test_wbteleop_task.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit final tests**

Run:

```bash
git add tests/test_wbteleop_task.py
git commit -m "test: verify wbteleop train cli"
```

---

### Task 7: Final Verification

**Files:**
- No file edits.

- [ ] **Step 1: Check worktree**

Run:

```bash
git status --short
```

Expected: no uncommitted implementation changes.

- [ ] **Step 2: Run final targeted suite**

Run:

```bash
uv run pytest \
  tests/test_wbteleop_task.py \
  tests/test_task_configs.py \
  tests/test_tracking_task.py \
  tests/test_runner.py \
  tests/test_distillation_runner_smoke.py \
  tests/test_distillation_teacher_adapter.py \
  -q
```

Expected: pass.

- [ ] **Step 3: Confirm training command help**

Run:

```bash
uv run train Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop --help
```

Expected: exits 0 and displays `teacher-checkpoint-path`, `bc-weight-start`, `bc-weight-end`, and `bc-decay-steps`.

- [ ] **Step 4: Report result**

Report:

```text
Implemented Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop under src/mjlab/tasks/tracking/wbteleop.
Actor observations are the seven non-privileged wbteleop terms.
PPO update now logs bc_mse, bc_weight, and bc_loss.
History support is preserved through motion command reference windows and proprio observation history.
```
