# Distillation Task Design

**Date:** 2026-04-22

**Status:** Proposed

**Scope:** First-phase design for a standalone distillation task that reuses the existing tracking teacher environment and checkpoints while introducing a sparse-observation student trained with online action distillation.

## Goal

Add a new task package under `src/mjlab/tasks/distillation` for motion-command distillation without modifying general `mjlab` infrastructure and without forking the existing tracking task implementation.

The first implementation phase should:

- reuse the current multi-motion tracking command pipeline as the teacher-side motion source
- reuse an existing `tracking_bfm` checkpoint as a frozen teacher
- train a student from sparse observations using deterministic teacher mean actions
- use Bernoulli-switched mixed rollout between teacher and student actions, with teacher probability decaying over training
- keep termination behavior identical to the original task
- log both environment metrics and distillation-specific metrics

## Design Principles

- **Task isolation:** all distillation-specific code lives under `src/mjlab/tasks/distillation`
- **Teacher truth source:** teacher observations, teacher environment semantics, multi-motion sampling, rewards, and terminations come from the existing tracking task
- **Student modularity:** student observation construction and student network selection are explicit modules to support future ablations
- **No privileged student path:** unlike ProtoMotions-style architectures, the first phase keeps a single student action path
- **Future extensibility:** sparse command extraction is implemented with a future-step-aware interface, but first phase only enables single-frame extraction

## Package Layout

```text
src/mjlab/tasks/distillation/
  __init__.py
  distillation_env_cfg.py
  mdp/
    __init__.py
    commands.py
    observations.py
    terminations.py
  rl/
    __init__.py
    teacher.py
    models.py
    schedules.py
    algorithm.py
    runner.py
  config/
    __init__.py
    g1/
      __init__.py
      env_cfgs.py
      rl_cfg.py
```

## High-Level Architecture

The distillation task is a composition layer on top of tracking:

1. Build a base tracking environment configuration from the existing tracking task factory.
2. Keep the original full motion command and teacher observation semantics intact.
3. Add a new student observation group based on sparse commands plus fixed proprioceptive terms.
4. Load a frozen teacher checkpoint and its observation normalizer.
5. Run mixed rollout using a Bernoulli switch between teacher and student actions.
6. Train the student only from online `(student_obs, teacher_mean_action)` pairs.

This avoids copying tracking task logic while keeping all distillation code isolated.

## Environment Composition

### Base environment source

The distillation environment factory should call the tracking base environment factory and modify the returned configuration instead of re-declaring the tracking task.

The following teacher-side components are reused directly from tracking:

- full multi-motion command source
- scene/entities/sensors
- actions
- events
- rewards
- terminations
- teacher actor observations

### Observation groups

The resulting environment should expose at least:

- `teacher_actor`: original tracking actor observation group, reused without semantic changes
- `student_actor`: sparse student observation group defined by the distillation task
- `critic`: retained only if needed for compatibility with existing wrappers; first-phase distillation does not train a critic

## Motion Command Reuse and Sparse Extraction

### Motion command source

The distillation task should reuse the tracking multi-motion command implementation, specifically the existing multi-motion sampling logic. Distillation code should not duplicate motion file sampling, motion indexing, or adaptive sampling logic.

### Sparse command extraction

`src/mjlab/tasks/distillation/mdp/commands.py` should only add sparse extractors on top of the existing motion term.

The internal extractor interface should already accept `future_steps`, but first phase only supports:

- `future_steps = (0,)`

### Student command definition

The student command is split into four parts:

1. `ee_pose`
2. `base_vel`
3. `base_ang_vel`
4. `base_height`

#### `ee_pose`

- two end effectors
- expressed relative to the anchor frame
- anchor is the tracking anchor body; for the first G1 variant this is expected to be the pelvis anchor used by the task configuration
- parameterization per end effector:
  - position: 3
  - rotation: 6D rotation representation
- total dimension: `2 * (3 + 6) = 18`

#### `base_vel`

- target base linear velocity
- expressed in world frame
- dimension: `3`

#### `base_ang_vel`

- target base angular velocity
- expressed in world frame
- dimension: `3`

#### `base_height`

- target anchor height
- world-frame `z` scalar of the anchor
- dimension: `1`

#### Single-frame sparse command size

- total sparse command dimension: `18 + 3 + 3 + 1 = 25`

## Student Observations

`src/mjlab/tasks/distillation/mdp/observations.py` should define the student observation group as:

### Sparse command terms

- `ee_pose`
- `base_lin_vel_w`
- `base_ang_vel_w`
- `anchor_height_w`

### Fixed proprioceptive terms

- `projected_gravity`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `actions`

These five proprioceptive terms are fixed for the first phase and should use the exact intended observation functions and corruption settings chosen for the student task.

### Group construction

Student observation construction should be modular even in the first phase:

- command terms builder
- proprio term builder
- student actor group builder

This supports later ablations without rewriting the task.

## Termination Handling

`src/mjlab/tasks/distillation/mdp/terminations.py` should exist for package completeness, but first-phase behavior is simple:

- use the original tracking task termination behavior as-is
- do not add relaxed/minimal/custom distillation termination modes yet

This means the distillation task should inherit the existing termination structure, including `time_out` and tracking-related failure criteria, without introducing phase-one divergence.

## Teacher Integration

`src/mjlab/tasks/distillation/rl/teacher.py` should provide a frozen teacher adapter.

### Responsibilities

- load an existing `tracking_bfm` checkpoint
- restore the teacher policy on the target device
- restore any teacher observation normalization state required for inference
- expose a deterministic mean-action inference API
- keep teacher-side observation handling isolated from the runner

### Required behavior

- teacher observations are taken from `teacher_actor`
- teacher action targets use deterministic mean action only
- teacher parameters are frozen

## Student Model

`src/mjlab/tasks/distillation/rl/models.py` should host a student model registry.

### First-phase implementation

- one MLP student policy
- configurable hidden dimensions and activation
- no critic
- no latent privileged branch
- no VAE

### Design intent

The model builder should make later network ablations easy without rewriting the runner or algorithm.

## Teacher-Student Mixing Schedule

`src/mjlab/tasks/distillation/rl/schedules.py` should define the rollout mixing schedule.

### First-phase behavior

- Bernoulli switch between teacher and student action
- sampled per environment and per step
- teacher selection probability `beta(t)` decays during training

### Initial supported schedule family

- configurable decay schedule
- first implementation may choose a simple schedule such as linear or cosine decay

The schedule API should return teacher probability as a pure function of training iteration.

## Distillation Algorithm

`src/mjlab/tasks/distillation/rl/algorithm.py` should contain a pure online action distillation update.

### Inputs

- `student_obs`
- `teacher_action`

### First-phase loss

- mean squared error between student action and teacher mean action

### Explicit non-goals for phase one

- PPO
- actor-critic loss
- KL-to-teacher policy distribution
- offline dataset distillation
- privileged student branch
- latent reconstruction losses

## Runner

`src/mjlab/tasks/distillation/rl/runner.py` should implement the main training loop.

### Responsibilities

- instantiate distillation env
- instantiate student model
- instantiate frozen teacher adapter
- instantiate teacher mixing schedule
- collect online rollout data
- execute mixed rollout action
- train student on buffered `(student_obs, teacher_action)` samples
- save checkpoints every fixed interval
- run periodic student-only evaluation

### Per-step rollout logic

For each environment step:

1. read `student_actor` observations
2. read `teacher_actor` observations
3. compute `student_action`
4. compute `teacher_action`
5. compute current `beta`
6. sample Bernoulli teacher-selection mask
7. choose `rollout_action` from teacher or student action
8. step environment with `rollout_action`
9. store `student_obs` and `teacher_action` for supervised updates

### Action naming

The implementation should explicitly keep these tensors distinct:

- `student_action`
- `teacher_action`
- `rollout_action`

This avoids ambiguity in mixed-rollout training.

## Evaluation

### Training-time evaluation

Phase one should use:

- normal training rollout logs for mixed-policy behavior
- periodic `student-only` evaluation for true student performance

### Student-only evaluation

- teacher disabled for action execution
- effectively `beta = 0`
- no parameter updates
- separate logging namespace

### Initial frequency

- every `1000` training iterations

### Initial scale

- smaller evaluation workload than training
- target size around one quarter of training environment count
- `8` to `16` episodes per evaluation pass

## Logging and Metrics

Distillation metrics are added on top of the original environment metrics; they do not replace existing task metrics.

### Keep existing environment metrics

Tracking/environment metrics already produced by the underlying task should remain visible.

### Add first-phase distillation metrics

Training-time:

- `Train/distill/action_mse`
- `Train/distill/beta_teacher`
- `Train/distill/teacher_action_ratio`

Student-only evaluation:

- `EvalStudent/env/return`
- `EvalStudent/env/episode_len`
- `EvalStudent/env/anchor_pos_err`
- `EvalStudent/env/termination_breakdown/*`

These are the minimum first-phase metrics for debugging progress and verifying that decreasing teacher usage does not mask poor student quality.

## Checkpoint Policy

Phase one uses a simple fixed save policy:

- no best-checkpoint selection
- save every `500` training iterations

Student-only evaluation is used for monitoring, not for checkpoint selection.

## Configuration

### `config/g1/env_cfgs.py`

Responsibilities:

- create/register the G1 distillation task
- compose distillation config from tracking base config
- set robot-specific student command extraction details if needed

### `config/g1/rl_cfg.py`

Responsibilities:

- teacher checkpoint path/config
- student model hyperparameters
- rollout mixing schedule hyperparameters
- optimizer/training hyperparameters
- evaluation interval
- save interval

## Unit Test Plan

First-phase tests should focus on correctness of modular components rather than long training runs.

### Required tests

1. `test_distillation_task_loads`
   - task config can be constructed and registered

2. `test_sparse_command_extractor_single_frame`
   - sparse extractor returns expected dimensions and semantics for single-frame extraction

3. `test_student_observation_group_terms`
   - student observation group contains exactly the intended sparse-command and proprio terms

4. `test_teacher_mean_action_path`
   - teacher adapter uses deterministic mean action path

5. `test_beta_schedule_decay`
   - teacher probability schedule decays correctly and respects configured bounds

6. `test_mixed_rollout_selector`
   - Bernoulli action selection behaves correctly at `beta=0`, `beta=1`, and intermediate values

7. `test_distillation_runner_one_iteration_smoke`
   - one short train iteration can execute end-to-end without crashing

## Out of Scope for Phase One

- future-horizon sparse command training
- PPO + distillation joint training
- VAE / masked mimic style latent modeling
- custom termination modes for distillation
- best-checkpoint selection
- offline data generation pipeline
- action blending between teacher and student

## Acceptance Criteria

Phase one is considered complete when:

- a standalone distillation task exists under `src/mjlab/tasks/distillation`
- the task reuses tracking multi-motion commands and teacher observations without copying teacher semantics
- a frozen tracking teacher checkpoint can supervise a sparse-observation student
- mixed rollout with decaying teacher probability runs end-to-end
- student-only evaluation runs periodically during training
- checkpoints are saved every `500` iterations
- the planned unit tests pass
