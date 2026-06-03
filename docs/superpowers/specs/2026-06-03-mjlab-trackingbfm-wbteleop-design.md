# Mjlab Tracking BFM Wbteleop Design

## Goal

Implement `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop` as a full-body teleoperation tracking task that trains a non-sparse, non-privileged actor with PPO plus a teacher-action behavior cloning loss.

Only training code is in scope. Play behavior and ONNX export behavior are not part of the first implementation, except that the task should still register a play config with matching observation structure so existing task integrity tests keep passing.

## Scope

Implementation should live in:

```text
src/mjlab/tasks/tracking/wbteleop/
```

The directory name is `wbteleop`, matching the task id suffix.

The task registration should be:

```text
Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop
```

## Architecture

The task composes the existing G1 BFM tracking environment and replaces only the actor observation with a non-privileged whole-body teleoperation observation group.

Recommended files:

```text
src/mjlab/tasks/tracking/wbteleop/__init__.py
src/mjlab/tasks/tracking/wbteleop/env_cfg.py
src/mjlab/tasks/tracking/wbteleop/rl_cfg.py
src/mjlab/tasks/tracking/wbteleop/observations.py
src/mjlab/tasks/tracking/wbteleop/algorithm.py
src/mjlab/tasks/tracking/wbteleop/runner.py
```

Responsibilities:

- `__init__.py` registers the task.
- `env_cfg.py` builds the environment config from `unitree_g1_flat_tracking_bfm_env_cfg()` and installs wbteleop observations.
- `rl_cfg.py` defines runner and algorithm config defaults.
- `observations.py` defines wbteleop-specific observation terms, including `motion_ref_ang_vel`.
- `algorithm.py` defines `WbTeleopPPO`, a PPO variant with teacher-action MSE.
- `runner.py` defines `WbTeleopTrackingRunner`, a tracking runner that loads the teacher and injects it into the algorithm.

The runner should preserve the existing `MotionTrackingOnPolicyRunner` behavior for rollout collection, adaptive sampling, logging, checkpointing, and training loop structure.

## Environment

`unitree_g1_flat_tracking_bfm_wbteleop_env_cfg()` should:

1. Start from `unitree_g1_flat_tracking_bfm_env_cfg(play=play)`.
2. Preserve the BFM motion command, rewards, terminations, randomization, actions, and privileged critic observation.
3. Replace `cfg.observations["actor"]` with the wbteleop actor observation group.
4. Add `cfg.observations["teacher_actor"]` as a corruption-free copy of the original tracking actor observation.
5. Keep `cfg.observations["critic"]` unchanged.
6. In play mode, disable actor corruption and teacher actor corruption.

The `teacher_actor` group is only used to compute teacher target actions for BC. It must not be included in student actor or critic observation groups.

## Actor Observations

The student actor uses exactly seven terms:

```text
command
motion_ref_ang_vel
projected_gravity
base_ang_vel
joint_pos
joint_vel
actions
```

Term definitions:

```python
"command": ObservationTermCfg(
  func=mdp.generated_commands,
  params={"command_name": "motion"},
)
"motion_ref_ang_vel": ObservationTermCfg(
  func=wbteleop_observations.motion_ref_ang_vel,
  params={"command_name": "motion"},
  noise=Unoise(n_min=-0.05, n_max=0.05),
)
"projected_gravity": ObservationTermCfg(
  func=mdp.projected_gravity,
  noise=Unoise(n_min=-0.05, n_max=0.05),
)
"base_ang_vel": ObservationTermCfg(
  func=mdp.builtin_sensor,
  params={"sensor_name": "robot/imu_ang_vel"},
  noise=Unoise(n_min=-0.2, n_max=0.2),
)
"joint_pos": ObservationTermCfg(
  func=mdp.joint_pos_rel,
  params={"biased": True},
  noise=Unoise(n_min=-0.01, n_max=0.01),
)
"joint_vel": ObservationTermCfg(
  func=mdp.joint_vel_rel,
  noise=Unoise(n_min=-0.5, n_max=0.5),
)
"actions": ObservationTermCfg(func=mdp.last_action)
```

The actor must not include privileged tracking terms such as:

```text
motion_anchor_pos_b
motion_anchor_ori_b
body_pos
body_ori
base_lin_vel
```

## History Support

The wbteleop task keeps history support.

Default values:

```text
history_steps = 0
future_steps = 1
```

The config factory should allow programmatic overrides:

```python
unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(history_steps=10, future_steps=1)
```

If history is changed through CLI after the registered task config has already been built, command history and proprio observation history must both be overridden. Observation `history_length` fields are not automatically recomputed from `commands.motion.history_steps`.

History semantics:

- `command` uses the motion command tensor and should not add `ObservationTermCfg.history_length`.
- `motion_ref_ang_vel` returns the command term's `anchor_ang_vel_w`, which already supports the command reference window.
- `projected_gravity`, `base_ang_vel`, `joint_pos`, `joint_vel`, and `actions` should use `history_length = history_steps + 1` when `history_steps > 0`.
- When `history_steps == 0`, those proprioceptive terms should not set additional history.

This avoids double-stacking reference history while still supporting proprioceptive history.

## Motion Reference Angular Velocity

The current repository does not provide `mdp.motion_ref_ang_vel`. Add the wbteleop-specific function:

```python
def motion_ref_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  return command.anchor_ang_vel_w
```

The implementation should type-check or cast against the tracking motion command interface as needed.

## RL Configuration

Use the existing BFM tracking network shape initially:

```text
hidden_dims = (2048, 2048, 1024, 1024, 512, 256, 128)
activation = "elu"
obs_normalization = True
distribution = GaussianDistribution(init_std=1.0, std_type="scalar")
```

Runner defaults:

```text
experiment_name = "g1_tracking_wbteleop"
save_interval = 1000
num_steps_per_env = 24
max_iterations = 300_000
```

Observation groups:

```python
obs_groups={
  "actor": ("actor",),
  "critic": ("critic",),
}
```

The `teacher_actor` group is intentionally omitted from `obs_groups`.

## Teacher Action BC Loss

Training uses PPO tracking plus teacher-action MSE:

```text
student_action_mean = current actor mean action
teacher_action_mean = teacher policy mean action
bc_mse = mse(student_action_mean, teacher_action_mean)
loss = ppo_loss + bc_weight * bc_mse
```

The student target is the actor mean, not the sampled rollout action. The teacher target is computed without gradients.

Teacher action dimensionality should match the environment action dimension, expected to be 29 DOF for Unitree G1.

The teacher should be loaded from a checkpoint for:

```text
teacher_task_id = "Mjlab-Trackingbfm-Flat-Unitree-G1"
teacher_checkpoint_path = ""
teacher_obs_group = "teacher_actor"
```

`teacher_checkpoint_path` must be provided for actual training.

## BC Weight Schedule

Use strict cosine decay from 0.5 to 0.1 over 10000 PPO iterations:

```text
bc_weight_start = 0.5
bc_weight_end = 0.1
bc_decay_steps = 10_000
```

Schedule:

```text
progress = min(iteration, bc_decay_steps) / bc_decay_steps
bc_weight = bc_weight_end + (bc_weight_start - bc_weight_end) * 0.5 * (1 + cos(pi * progress))
```

Expected values:

```text
iteration 0: 0.5
iteration 10000: 0.1
iteration > 10000: 0.1
```

Log metrics:

```text
bc_mse
bc_weight
bc_loss
```

## Algorithm Implementation

`WbTeleopPPO` should be a local subclass or local variant of the installed `rsl_rl.algorithms.PPO`.

Because upstream PPO does not expose a loss hook, the implementation may copy the current `PPO.update()` body into `wbteleop/algorithm.py` and insert the teacher-action MSE term before the backward pass. Keep the copy local to the wbteleop directory and avoid modifying site-packages or common PPO code.

The algorithm class should be referenced by fully qualified class name:

```text
mjlab.tasks.tracking.wbteleop.algorithm:WbTeleopPPO
```

This is supported by rsl_rl's callable resolver.

## Runner Implementation

`WbTeleopTrackingRunner` should inherit from `MotionTrackingOnPolicyRunner`.

It should:

1. Load the teacher task config and teacher checkpoint.
2. Build a teacher runner or teacher policy adapter following the existing distillation teacher loading pattern.
3. Override the nested teacher runner actor obs group to `("teacher_actor",)` before constructing the teacher, so the checkpoint model is built with teacher observation dimensions rather than wbteleop student actor dimensions.
4. Provide teacher inference on the `teacher_actor` observation group.
5. Attach the teacher adapter to `WbTeleopPPO` before training.

Teacher loading should fail clearly if `teacher_checkpoint_path` is empty during training.

## Training Command

Default training command:

```bash
uv run train Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop \
  --agent.algorithm.teacher_checkpoint_path /path/to/teacher/model.pt \
  --env.scene.num-envs 4096
```

With history:

```bash
uv run train Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop \
  --agent.algorithm.teacher_checkpoint_path /path/to/teacher/model.pt \
  --env.commands.motion.history_steps 10 \
  --env.commands.motion.future_steps 1 \
  --env.observations.actor.terms.projected_gravity.history_length 11 \
  --env.observations.actor.terms.base_ang_vel.history_length 11 \
  --env.observations.actor.terms.joint_pos.history_length 11 \
  --env.observations.actor.terms.joint_vel.history_length 11 \
  --env.observations.actor.terms.actions.history_length 11 \
  --env.scene.num-envs 4096
```

## Testing

Add focused tests for:

1. Task registration includes `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop`.
2. Actor observation terms are exactly the seven wbteleop terms.
3. Actor observation excludes privileged tracking terms.
4. `teacher_actor` exists and is corruption-free.
5. `rl_cfg.obs_groups["actor"] == ("actor",)` and `rl_cfg.obs_groups["critic"] == ("critic",)`.
6. History config sets proprioceptive `history_length` to `history_steps + 1` while leaving `command` and `motion_ref_ang_vel` without extra observation history.
7. BC weight schedule returns 0.5 at iteration 0, 0.1 at iteration 10000, and clamps to 0.1 afterwards.
8. A small algorithm smoke test reports `bc_mse`, `bc_weight`, and `bc_loss`.

## Risks

The main maintenance risk is copying `PPO.update()` locally. This is acceptable for the first implementation because it keeps the behavior isolated to the wbteleop task and avoids changing global PPO behavior.

The main behavior risk is teacher observation leakage. The design avoids this by keeping `teacher_actor` outside `obs_groups` and using it only inside the teacher adapter path.

The main configuration risk is ambiguity around history dimensions. The design separates command reference history from observation history to avoid double-stacking reference signals.
