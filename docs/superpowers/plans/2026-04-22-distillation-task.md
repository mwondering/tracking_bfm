# Distillation Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `mjlab.tasks.distillation` task package that reuses the tracking teacher task and checkpoint while training a sparse-observation student with mixed teacher/student rollout and pure action distillation.

**Architecture:** Build the new task as a thin composition layer over the existing tracking task. Reuse tracking multi-motion commands, teacher actor observations, rewards, events, and terminations. Add student-specific sparse command extraction, student observation groups, a frozen teacher adapter, a Bernoulli rollout mixing schedule, and a minimal action-distillation runner/algorithm.

**Tech Stack:** Python, PyTorch, existing `mjlab` task registry, `ManagerBasedRlEnvCfg`, current tracking task modules, current `MjlabOnPolicyRunner` patterns, pytest.

---

### Task 1: Scaffold The Distillation Task Package And Registration

**Files:**
- Create: `src/mjlab/tasks/distillation/__init__.py`
- Create: `src/mjlab/tasks/distillation/distillation_env_cfg.py`
- Create: `src/mjlab/tasks/distillation/mdp/__init__.py`
- Create: `src/mjlab/tasks/distillation/rl/__init__.py`
- Create: `src/mjlab/tasks/distillation/config/__init__.py`
- Create: `src/mjlab/tasks/distillation/config/g1/__init__.py`
- Create: `src/mjlab/tasks/distillation/config/g1/env_cfgs.py`
- Create: `src/mjlab/tasks/distillation/config/g1/rl_cfg.py`
- Modify: `src/mjlab/tasks/__init__.py` or the package import entry that ensures task modules register on import
- Test: `tests/test_distillation_task.py`

- [ ] **Step 1: Write the failing task-registration test**

```python
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


def test_distillation_task_is_registered() -> None:
  assert "Mjlab-Distillation-Flat-Unitree-G1" in list_tasks()


def test_distillation_task_loads_cfgs() -> None:
  env_cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1")
  rl_cfg = load_rl_cfg("Mjlab-Distillation-Flat-Unitree-G1")

  assert "teacher_actor" in env_cfg.observations
  assert "student_actor" in env_cfg.observations
  assert rl_cfg.class_name == "DistillationRunner"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_task.py -v`
Expected: FAIL because `mjlab.tasks.distillation` and task registration do not exist yet.

- [ ] **Step 3: Add the package skeleton and task registration**

```python
# src/mjlab/tasks/distillation/config/g1/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.distillation.rl import DistillationRunner
from .env_cfgs import unitree_g1_flat_distillation_env_cfg
from .rl_cfg import unitree_g1_distillation_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Distillation-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_distillation_env_cfg(),
  play_env_cfg=unitree_g1_flat_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_distillation_runner_cfg(),
  runner_cls=DistillationRunner,
)
```

```python
# src/mjlab/tasks/distillation/distillation_env_cfg.py
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_bfm_env_cfg


def make_distillation_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)
  return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation_task.py -v`
Expected: PASS with the task visible in the registry and both observation groups present.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_task.py src/mjlab/tasks/distillation src/mjlab/tasks/__init__.py
git commit -m "feat: scaffold distillation task package"
```

### Task 2: Add Sparse Command Extraction And Student Observation Groups

**Files:**
- Create: `src/mjlab/tasks/distillation/mdp/commands.py`
- Create: `src/mjlab/tasks/distillation/mdp/observations.py`
- Create: `src/mjlab/tasks/distillation/mdp/terminations.py`
- Modify: `src/mjlab/tasks/distillation/distillation_env_cfg.py`
- Modify: `src/mjlab/tasks/distillation/config/g1/env_cfgs.py`
- Test: `tests/test_distillation_sparse_command.py`
- Test: `tests/test_distillation_student_obs.py`

- [ ] **Step 1: Write failing sparse-command tests**

```python
import torch

from mjlab.tasks.distillation.mdp import commands as distill_cmds


def test_student_sparse_command_dim(mock_tracking_env) -> None:
  out = distill_cmds.student_sparse_command(
    mock_tracking_env,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    future_steps=(0,),
  )
  assert out.shape == (mock_tracking_env.num_envs, 25)


def test_student_anchor_height_matches_anchor_z(mock_tracking_env) -> None:
  out = distill_cmds.student_anchor_height_w(mock_tracking_env, command_name="motion")
  assert torch.allclose(
    out.squeeze(-1),
    mock_tracking_env.command_manager.get_term("motion").anchor_pos_w[:, 2],
  )
```

```python
from mjlab.tasks.registry import load_env_cfg


def test_student_obs_terms_are_exact() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1")
  terms = set(cfg.observations["student_actor"].terms.keys())
  assert terms == {
    "ee_pose",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "anchor_height_w",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distillation_sparse_command.py tests/test_distillation_student_obs.py -v`
Expected: FAIL because sparse extraction functions and `student_actor` terms do not exist yet.

- [ ] **Step 3: Implement sparse command extraction and student observation builders**

```python
# src/mjlab/tasks/distillation/mdp/commands.py
def student_anchor_height_w(env, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.anchor_pos_w[:, 2:3]


def student_sparse_command(env, command_name: str, ee_body_names: tuple[str, str], future_steps: tuple[int, ...] = (0,)) -> torch.Tensor:
  assert future_steps == (0,)
  return torch.cat(
    [
      student_ee_pose_b(env, command_name=command_name, ee_body_names=ee_body_names),
      student_base_lin_vel_w(env, command_name=command_name),
      student_base_ang_vel_w(env, command_name=command_name),
      student_anchor_height_w(env, command_name=command_name),
    ],
    dim=-1,
  )
```

```python
# src/mjlab/tasks/distillation/mdp/observations.py
def build_student_actor_terms() -> dict[str, ObservationTermCfg]:
  return {
    "ee_pose": ObservationTermCfg(func=distill_mdp.student_ee_pose_b, params={"command_name": "motion", "ee_body_names": (...) }),
    "base_lin_vel_w": ObservationTermCfg(func=distill_mdp.student_base_lin_vel_w, params={"command_name": "motion"}),
    "base_ang_vel_w": ObservationTermCfg(func=distill_mdp.student_base_ang_vel_w, params={"command_name": "motion"}),
    "anchor_height_w": ObservationTermCfg(func=distill_mdp.student_anchor_height_w, params={"command_name": "motion"}),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
```

- [ ] **Step 4: Wire the env factory to expose `teacher_actor` and `student_actor`**

```python
# src/mjlab/tasks/distillation/distillation_env_cfg.py
cfg.observations["teacher_actor"] = cfg.observations["actor"]
cfg.observations["student_actor"] = ObservationGroupCfg(
  terms=build_student_actor_terms(),
  concatenate_terms=True,
  enable_corruption=not play,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_distillation_sparse_command.py tests/test_distillation_student_obs.py -v`
Expected: PASS with 25-D sparse commands and exact student observation term coverage.

- [ ] **Step 6: Commit**

```bash
git add tests/test_distillation_sparse_command.py tests/test_distillation_student_obs.py src/mjlab/tasks/distillation/mdp src/mjlab/tasks/distillation/distillation_env_cfg.py src/mjlab/tasks/distillation/config/g1/env_cfgs.py
git commit -m "feat: add sparse command extraction and student observations"
```

### Task 3: Add Teacher Adapter And Mixing Schedule

**Files:**
- Create: `src/mjlab/tasks/distillation/rl/teacher.py`
- Create: `src/mjlab/tasks/distillation/rl/schedules.py`
- Modify: `src/mjlab/tasks/distillation/config/g1/rl_cfg.py`
- Test: `tests/test_distillation_teacher_adapter.py`
- Test: `tests/test_distillation_mix_schedule.py`

- [ ] **Step 1: Write failing teacher and schedule tests**

```python
import torch

from mjlab.tasks.distillation.rl.schedules import LinearTeacherMixSchedule


def test_linear_teacher_mix_schedule_decays() -> None:
  schedule = LinearTeacherMixSchedule(beta_start=1.0, beta_end=0.0, decay_steps=100)
  assert schedule(0) == 1.0
  assert schedule(100) == 0.0
  assert schedule(50) < 1.0
```

```python
def test_teacher_adapter_uses_mean_action(mock_teacher_adapter, mock_teacher_obs) -> None:
  action = mock_teacher_adapter.act_mean(mock_teacher_obs)
  assert action.shape[-1] == mock_teacher_adapter.action_dim
  assert mock_teacher_adapter.uses_deterministic_mean_action is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distillation_teacher_adapter.py tests/test_distillation_mix_schedule.py -v`
Expected: FAIL because neither the adapter nor the schedule exists.

- [ ] **Step 3: Implement the schedule and frozen teacher adapter**

```python
# src/mjlab/tasks/distillation/rl/schedules.py
class LinearTeacherMixSchedule:
  def __init__(self, beta_start: float, beta_end: float, decay_steps: int):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.decay_steps = decay_steps

  def __call__(self, iteration: int) -> float:
    if iteration >= self.decay_steps:
      return self.beta_end
    alpha = iteration / self.decay_steps
    return self.beta_start + alpha * (self.beta_end - self.beta_start)
```

```python
# src/mjlab/tasks/distillation/rl/teacher.py
class TeacherPolicyAdapter:
  uses_deterministic_mean_action = True

  def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      return self.runner.alg.get_policy()(obs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_distillation_teacher_adapter.py tests/test_distillation_mix_schedule.py -v`
Expected: PASS with deterministic teacher-mean action behavior and monotonic schedule decay.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_teacher_adapter.py tests/test_distillation_mix_schedule.py src/mjlab/tasks/distillation/rl/teacher.py src/mjlab/tasks/distillation/rl/schedules.py src/mjlab/tasks/distillation/config/g1/rl_cfg.py
git commit -m "feat: add teacher adapter and rollout mixing schedule"
```

### Task 4: Implement Student Model, Distillation Algorithm, And Runner

**Files:**
- Create: `src/mjlab/tasks/distillation/rl/models.py`
- Create: `src/mjlab/tasks/distillation/rl/algorithm.py`
- Create: `src/mjlab/tasks/distillation/rl/runner.py`
- Modify: `src/mjlab/tasks/distillation/rl/__init__.py`
- Modify: `src/mjlab/tasks/distillation/config/g1/rl_cfg.py`
- Test: `tests/test_distillation_runner_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
def test_distillation_runner_one_iteration_smoke(make_distillation_env, make_distillation_cfg) -> None:
  env = make_distillation_env()
  cfg = make_distillation_cfg(max_iterations=1, num_steps_per_env=2)
  runner = DistillationRunner(env, cfg, device="cpu")
  logs = runner.learn(1)
  assert "Train/distill/action_mse" in logs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_runner_smoke.py -v`
Expected: FAIL because the student model, algorithm, and runner do not exist.

- [ ] **Step 3: Implement the minimal MLP student and action distillation update**

```python
# src/mjlab/tasks/distillation/rl/models.py
class DistillMlpPolicy(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...], activation: str):
    super().__init__()
    self.net = build_mlp(obs_dim, action_dim, hidden_dims, activation)

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return self.net(obs)
```

```python
# src/mjlab/tasks/distillation/rl/algorithm.py
class ActionDistillationAlgorithm:
  def update(self, student_obs: torch.Tensor, teacher_action: torch.Tensor) -> dict[str, float]:
    pred = self.student.act(student_obs)
    loss = F.mse_loss(pred, teacher_action)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return {"Train/distill/action_mse": float(loss.detach().cpu())}
```

- [ ] **Step 4: Implement the mixed-rollout runner**

```python
# src/mjlab/tasks/distillation/rl/runner.py
teacher_mask = torch.rand(env.num_envs, device=self.device) < beta
rollout_action = torch.where(
  teacher_mask[:, None],
  teacher_action,
  student_action,
)
```

```python
logs.update(
  {
    "Train/distill/beta_teacher": beta,
    "Train/distill/teacher_action_ratio": float(teacher_mask.float().mean().cpu()),
  }
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_distillation_runner_smoke.py -v`
Expected: PASS with one short train iteration producing distillation logs.

- [ ] **Step 6: Commit**

```bash
git add tests/test_distillation_runner_smoke.py src/mjlab/tasks/distillation/rl
git commit -m "feat: implement distillation runner and algorithm"
```

### Task 5: Add Student-Only Evaluation And Verification Coverage

**Files:**
- Modify: `src/mjlab/tasks/distillation/rl/runner.py`
- Modify: `src/mjlab/tasks/distillation/config/g1/rl_cfg.py`
- Test: `tests/test_distillation_runner_smoke.py`
- Test: `tests/test_distillation_task.py`

- [ ] **Step 1: Write the failing evaluation assertions**

```python
def test_distillation_runner_logs_student_eval(make_distillation_env, make_distillation_cfg) -> None:
  env = make_distillation_env()
  cfg = make_distillation_cfg(max_iterations=1, num_steps_per_env=2, student_eval_interval=1)
  runner = DistillationRunner(env, cfg, device="cpu")
  logs = runner.learn(1)
  assert "EvalStudent/env/return" in logs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_runner_smoke.py -v`
Expected: FAIL because no `student-only eval` metrics are logged yet.

- [ ] **Step 3: Implement periodic student-only evaluation**

```python
def evaluate_student_only(self) -> dict[str, float]:
  # beta = 0, no teacher action execution
  return {
    "EvalStudent/env/return": mean_return,
    "EvalStudent/env/episode_len": mean_episode_len,
    "EvalStudent/env/anchor_pos_err": mean_anchor_pos_err,
  }
```

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_distillation_task.py tests/test_distillation_runner_smoke.py -v`
Expected: PASS with `EvalStudent/...` metrics present and task wiring intact.

- [ ] **Step 5: Run the focused distillation test suite**

Run: `pytest tests/test_distillation_task.py tests/test_distillation_sparse_command.py tests/test_distillation_student_obs.py tests/test_distillation_teacher_adapter.py tests/test_distillation_mix_schedule.py tests/test_distillation_runner_smoke.py -v`
Expected: PASS for all new distillation tests.

- [ ] **Step 6: Commit**

```bash
git add tests/test_distillation_task.py tests/test_distillation_sparse_command.py tests/test_distillation_student_obs.py tests/test_distillation_teacher_adapter.py tests/test_distillation_mix_schedule.py tests/test_distillation_runner_smoke.py src/mjlab/tasks/distillation
git commit -m "feat: add student-only evaluation for distillation"
```
