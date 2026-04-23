# Distillation Student Play Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sparse student-reference visualization for distillation task playback while reusing the existing `src/mjlab/scripts/play.py` entrypoint.

**Architecture:** Keep `play.py` generic and implement student play visualization inside `src/mjlab/tasks/distillation/mdp/commands.py`. Reuse the existing tracking motion command as the truth source, and add a distillation-local debug visualization adapter that draws only two EE spheres, two base-velocity arrows, and one base-height cylinder.

**Tech Stack:** Python, PyTorch tensors, existing `DebugVisualizer` abstraction, existing `play.py`, pytest.

---

### Task 1: Add Sparse Student Visualization Helpers

**Files:**
- Modify: `src/mjlab/tasks/distillation/mdp/commands.py`
- Test: `tests/test_distillation_student_play_viz.py`

- [ ] **Step 1: Write the failing extraction and dispatch tests**

```python
from types import SimpleNamespace

import numpy as np
import torch

from mjlab.tasks.distillation.mdp import commands as distill_cmds


class _MockVisualizer:
  def __init__(self):
    self.env_idx = 0
    self.show_all_envs = False
    self.meansize = 1.0
    self.spheres = []
    self.arrows = []
    self.cylinders = []

  def get_env_indices(self, num_envs: int):
    return [0] if num_envs > 0 else []

  def add_sphere(self, center, radius, color, label=None):
    self.spheres.append((np.asarray(center), radius, color, label))

  def add_arrow(self, start, end, color, width=0.015, label=None):
    self.arrows.append((np.asarray(start), np.asarray(end), color, width, label))

  def add_cylinder(self, start, end, radius, color, label=None):
    self.cylinders.append((np.asarray(start), np.asarray(end), radius, color, label))


def _make_mock_env():
  command = SimpleNamespace(
    cfg=SimpleNamespace(
      body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
      history_steps=0,
      future_steps=1,
    ),
    anchor_pos_w=torch.tensor([[1.0, 2.0, 1.2]], dtype=torch.float32),
    body_pos_w=torch.tensor(
      [[[1.2, 2.1, 1.3], [0.8, 1.9, 1.25]]], dtype=torch.float32
    ),
    anchor_lin_vel_w=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
    anchor_ang_vel_w=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
  )
  return SimpleNamespace(
    num_envs=1,
    command_manager=SimpleNamespace(get_term=lambda name: command),
  )


def test_student_play_reference_extractors_return_world_values() -> None:
  env = _make_mock_env()

  left, right = distill_cmds.get_student_ee_reference(
    env,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
  )
  lin_vel = distill_cmds.get_student_base_lin_vel_reference(env, command_name="motion")
  ang_vel = distill_cmds.get_student_base_ang_vel_reference(env, command_name="motion")
  height = distill_cmds.get_student_base_height_reference(env, command_name="motion")

  torch.testing.assert_close(left, torch.tensor([[1.2, 2.1, 1.3]]))
  torch.testing.assert_close(right, torch.tensor([[0.8, 1.9, 1.25]]))
  torch.testing.assert_close(lin_vel, torch.tensor([[0.5, 0.0, 0.0]]))
  torch.testing.assert_close(ang_vel, torch.tensor([[0.0, 0.0, 1.0]]))
  torch.testing.assert_close(height, torch.tensor([[1.2]]))


def test_debug_vis_student_sparse_command_draws_expected_primitives() -> None:
  env = _make_mock_env()
  visualizer = _MockVisualizer()

  distill_cmds.debug_vis_student_sparse_command(
    env,
    visualizer,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    lin_vel_scale=0.2,
    ang_vel_scale=0.1,
    ee_sphere_radius=0.03,
    height_radius=0.01,
  )

  assert len(visualizer.spheres) == 2
  assert len(visualizer.arrows) == 2
  assert len(visualizer.cylinders) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_student_play_viz.py -v`
Expected: FAIL because the distillation command module does not yet provide the student play extraction and visualization entrypoints.

- [ ] **Step 3: Add extraction helpers and primitive drawing helpers**

```python
# src/mjlab/tasks/distillation/mdp/commands.py
def get_student_ee_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
  ee_body_names: tuple[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
  command = _get_command(env, command_name)
  body_indexes = _get_body_indexes(command, ee_body_names)
  ee_pos_w = command.body_pos_w[:, body_indexes, :]
  return ee_pos_w[:, 0, :], ee_pos_w[:, 1, :]


def get_student_base_lin_vel_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  return _extract_current_step_velocity(command, command.anchor_lin_vel_w)


def get_student_base_ang_vel_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  return _extract_current_step_velocity(command, command.anchor_ang_vel_w)


def get_student_base_height_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.anchor_pos_w[:, 2:3]


def draw_student_ee_reference(
  visualizer: DebugVisualizer,
  left_ee: np.ndarray,
  right_ee: np.ndarray,
  radius: float,
  env_idx: int,
) -> None:
  visualizer.add_sphere(
    center=left_ee,
    radius=radius,
    color=(0.2, 0.8, 1.0, 1.0),
    label=f"student_ref_left_ee_{env_idx}",
  )
  visualizer.add_sphere(
    center=right_ee,
    radius=radius,
    color=(1.0, 0.5, 0.2, 1.0),
    label=f"student_ref_right_ee_{env_idx}",
  )
```

```python
# src/mjlab/tasks/distillation/mdp/commands.py
def draw_student_base_velocity_reference(
  visualizer: DebugVisualizer,
  anchor_pos: np.ndarray,
  base_lin_vel: np.ndarray,
  base_ang_vel: np.ndarray,
  lin_vel_scale: float,
  ang_vel_scale: float,
  env_idx: int,
) -> None:
  visualizer.add_arrow(
    start=anchor_pos,
    end=anchor_pos + base_lin_vel * lin_vel_scale,
    color=(0.1, 0.9, 0.9, 1.0),
    label=f"student_ref_base_lin_vel_{env_idx}",
  )
  visualizer.add_arrow(
    start=anchor_pos,
    end=anchor_pos + base_ang_vel * ang_vel_scale,
    color=(1.0, 0.8, 0.1, 1.0),
    label=f"student_ref_base_ang_vel_{env_idx}",
  )


def draw_student_base_height_reference(
  visualizer: DebugVisualizer,
  anchor_pos: np.ndarray,
  radius: float,
  env_idx: int,
) -> None:
  ground_point = anchor_pos.copy()
  ground_point[2] = 0.0
  visualizer.add_cylinder(
    start=ground_point,
    end=anchor_pos,
    radius=radius,
    color=(0.7, 1.0, 0.2, 0.6),
    label=f"student_ref_base_height_{env_idx}",
  )
```

```python
# src/mjlab/tasks/distillation/mdp/commands.py
def debug_vis_student_sparse_command(
  env: ManagerBasedRlEnv,
  visualizer: DebugVisualizer,
  command_name: str,
  ee_body_names: tuple[str, str],
  lin_vel_scale: float,
  ang_vel_scale: float,
  ee_sphere_radius: float,
  height_radius: float,
) -> None:
  env_indices = visualizer.get_env_indices(env.num_envs)
  if not env_indices:
    return

  left_ee, right_ee = get_student_ee_reference(
    env,
    command_name=command_name,
    ee_body_names=ee_body_names,
  )
  base_lin_vel = get_student_base_lin_vel_reference(env, command_name=command_name)
  base_ang_vel = get_student_base_ang_vel_reference(env, command_name=command_name)
  command = _get_command(env, command_name)

  for env_idx in env_indices:
    anchor_pos = command.anchor_pos_w[env_idx].cpu().numpy()
    draw_student_ee_reference(
      visualizer,
      left_ee[env_idx].cpu().numpy(),
      right_ee[env_idx].cpu().numpy(),
      radius=ee_sphere_radius,
      env_idx=env_idx,
    )
    draw_student_base_velocity_reference(
      visualizer,
      anchor_pos=anchor_pos,
      base_lin_vel=base_lin_vel[env_idx].cpu().numpy(),
      base_ang_vel=base_ang_vel[env_idx].cpu().numpy(),
      lin_vel_scale=lin_vel_scale,
      ang_vel_scale=ang_vel_scale,
      env_idx=env_idx,
    )
    draw_student_base_height_reference(
      visualizer,
      anchor_pos=anchor_pos,
      radius=height_radius,
      env_idx=env_idx,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation_student_play_viz.py -v`
Expected: PASS with correct extraction values and exactly 2 spheres, 2 arrows, and 1 cylinder.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_student_play_viz.py src/mjlab/tasks/distillation/mdp/commands.py
git commit -m "feat: add student play visualization helpers"
```

### Task 2: Wire Student Sparse Visualization Into Distillation Play Mode

**Files:**
- Modify: `src/mjlab/tasks/distillation/distillation_env_cfg.py`
- Modify: `src/mjlab/tasks/distillation/mdp/commands.py`
- Test: `tests/test_distillation_play_cfg.py`

- [ ] **Step 1: Write the failing play-config wiring test**

```python
import mjlab.tasks.distillation.config.g1  # noqa: F401

from mjlab.tasks.registry import load_env_cfg


def test_distillation_play_cfg_uses_student_sparse_visualization() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1", play=True)
  motion_cfg = cfg.commands["motion"]

  assert getattr(motion_cfg, "viz", None) is not None
  assert motion_cfg.viz.mode == "student_sparse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_play_cfg.py -v`
Expected: FAIL because the distillation play config still inherits the tracking play visualization mode.

- [ ] **Step 3: Add a distillation-local visualization mode contract**

```python
# src/mjlab/tasks/distillation/mdp/commands.py
STUDENT_PLAY_EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
STUDENT_PLAY_LIN_VEL_SCALE = 0.2
STUDENT_PLAY_ANG_VEL_SCALE = 0.12
STUDENT_PLAY_EE_SPHERE_RADIUS = 0.025
STUDENT_PLAY_HEIGHT_RADIUS = 0.01


def maybe_debug_vis_student_sparse_command(command, visualizer: DebugVisualizer) -> bool:
  if getattr(command.cfg.viz, "mode", None) != "student_sparse":
    return False
  debug_vis_student_sparse_command(
    env=command._env,
    visualizer=visualizer,
    command_name=command.cfg.name,
    ee_body_names=STUDENT_PLAY_EE_BODY_NAMES,
    lin_vel_scale=STUDENT_PLAY_LIN_VEL_SCALE,
    ang_vel_scale=STUDENT_PLAY_ANG_VEL_SCALE,
    ee_sphere_radius=STUDENT_PLAY_EE_SPHERE_RADIUS,
    height_radius=STUDENT_PLAY_HEIGHT_RADIUS,
  )
  return True
```

```python
# src/mjlab/tasks/distillation/distillation_env_cfg.py
def make_distillation_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)
  ...
  if play:
    cfg.commands["motion"].viz.mode = "student_sparse"
  return cfg
```

- [ ] **Step 4: Route student sparse mode through the distillation command/debug-vis hook**

```python
# src/mjlab/tasks/distillation/mdp/commands.py
def debug_vis_distillation_command(command, visualizer: DebugVisualizer) -> None:
  if maybe_debug_vis_student_sparse_command(command, visualizer):
    return
  command._debug_vis_impl(visualizer)
```

In this step, update the distillation task’s command-side integration so that play mode routes `"student_sparse"` to the new sparse visualizer and otherwise leaves the existing tracking visualization behavior untouched.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_distillation_play_cfg.py -v`
Expected: PASS with the play config selecting `viz.mode == "student_sparse"`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_distillation_play_cfg.py src/mjlab/tasks/distillation/distillation_env_cfg.py src/mjlab/tasks/distillation/mdp/commands.py
git commit -m "feat: wire student sparse visualization into distillation play"
```

### Task 3: Ensure Play Entry Works With Distillation Student Visualization

**Files:**
- Modify: `src/mjlab/scripts/play.py` only if a minimal generic hook is actually required
- Test: `tests/test_distillation_play_integration.py`

- [ ] **Step 1: Write the failing play integration test**

```python
from unittest.mock import MagicMock

import mjlab.tasks.distillation.config.g1  # noqa: F401

from mjlab.tasks.registry import load_env_cfg


def test_distillation_play_cfg_keeps_generic_play_entry_requirements() -> None:
  cfg = load_env_cfg("Mjlab-Distillation-Flat-Unitree-G1", play=True)

  assert "motion" in cfg.commands
  assert "student_actor" in cfg.observations
  assert cfg.terminations == {}
```

- [ ] **Step 2: Run test to verify it fails only if additional glue is required**

Run: `pytest tests/test_distillation_play_integration.py -v`
Expected: either PASS immediately or FAIL with a concrete integration gap between distillation play cfg and generic play assumptions.

- [ ] **Step 3: Add the minimal generic glue only if the test proves it is necessary**

If the generic `play.py` path already works, do not modify it.

If a small generic hook is needed, keep it narrow and task-agnostic. For example:

```python
# src/mjlab/scripts/play.py
# Keep all task-specific visualization semantics out of this file.
# Only add a generic hook if the play lifecycle needs to trigger command-local debug vis.
```

Do not add teacher/student branching logic to `play.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation_play_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_play_integration.py src/mjlab/scripts/play.py
git commit -m "feat: keep generic play entry compatible with distillation student play"
```

### Task 4: Add a Focused Playback Smoke Test and Final Verification

**Files:**
- Test: `tests/test_distillation_student_play_viz.py`
- Test: `tests/test_distillation_play_cfg.py`
- Test: `tests/test_distillation_play_integration.py`
- Modify: `scripts/train_distillation.sh` only if documentation or invocation notes need alignment

- [ ] **Step 1: Add a smoke test for stable labels and primitive counts across environments**

```python
def test_debug_vis_student_sparse_command_labels_are_stable() -> None:
  env = _make_mock_env()
  visualizer = _MockVisualizer()

  debug_vis_student_sparse_command(
    env,
    visualizer,
    command_name="motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    lin_vel_scale=0.2,
    ang_vel_scale=0.1,
    ee_sphere_radius=0.03,
    height_radius=0.01,
  )

  labels = [item[-1] for item in visualizer.spheres + visualizer.arrows + visualizer.cylinders]
  assert "student_ref_left_ee_0" in labels
  assert "student_ref_right_ee_0" in labels
  assert "student_ref_base_lin_vel_0" in labels
  assert "student_ref_base_ang_vel_0" in labels
  assert "student_ref_base_height_0" in labels
```

- [ ] **Step 2: Run the focused visualization test suite**

Run: `pytest tests/test_distillation_student_play_viz.py tests/test_distillation_play_cfg.py tests/test_distillation_play_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Run the existing distillation regression suite**

Run: `pytest tests/test_distillation_task.py tests/test_distillation_sparse_command.py tests/test_distillation_student_obs.py tests/test_distillation_mix_schedule.py tests/test_distillation_teacher_adapter.py tests/test_distillation_algorithm.py tests/test_distillation_runner_smoke.py -v`
Expected: PASS with no regressions to distillation training behavior.

- [ ] **Step 4: Do a final compile check for the touched distillation files**

Run: `python -m compileall src/mjlab/tasks/distillation src/mjlab/scripts/play.py`
Expected: no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_student_play_viz.py tests/test_distillation_play_cfg.py tests/test_distillation_play_integration.py src/mjlab/tasks/distillation src/mjlab/scripts/play.py scripts/train_distillation.sh
git commit -m "feat: add student sparse visualization for distillation play"
```
