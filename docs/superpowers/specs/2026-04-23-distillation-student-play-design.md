# Distillation Student Play Visualization Design

**Date:** 2026-04-23

**Status:** Proposed

**Scope:** Add student-oriented play visualization for the distillation task while reusing the existing `play.py` entrypoint and viewer infrastructure.

## Goal

Enable playing distilled student checkpoints with a visualization that matches the sparse student command semantics instead of the full motion-tracking reference visualization.

The new student play mode should:

- reuse the existing `src/mjlab/scripts/play.py` command entrypoint
- keep checkpoint loading, viewer selection, and video recording behavior unchanged
- avoid rendering the full reference motion ghost or full body frames
- render only sparse student reference signals:
  - end-effector reference points
  - base linear velocity arrow
  - base angular velocity arrow
  - base height indicator

## Recommendation

Reuse the current `play.py` entrypoint and implement task-specific visualization inside the distillation task.

Do **not** create a new standalone student play script.

## Alternatives Considered

### Option 1: Add teacher/student visualization branches directly in `play.py`

This would centralize the logic in one script but would make `play.py` aware of task-specific rendering details. It would couple generic play flow with per-task visualization semantics and would become harder to extend.

### Option 2: Reuse `play.py` and implement student visualization inside the distillation task

This keeps `play.py` generic and lets each task define its own debug visualization. It matches the current tracking architecture, where reference-motion visualization is already defined inside the command implementation rather than the play script.

### Option 3: Create a new `play_distillation_student.py`

This would isolate student play completely, but it would duplicate checkpoint loading, viewer setup, and future play-related fixes. It has the highest long-term maintenance cost.

## Decision

Choose **Option 2**.

The distillation task should own its student-specific visualization behavior. `play.py` should remain a general policy playback tool.

## Existing Architecture

Current tracking play works as follows:

- `src/mjlab/scripts/play.py` loads the task config, checkpoint, and viewer
- tracking reference visualization is defined inside the tracking motion command implementation
- the command implementation uses `DebugVisualizer` hooks such as `_debug_vis_impl()`

Relevant current files:

- `src/mjlab/scripts/play.py`
- `src/mjlab/tasks/tracking/mdp/commands.py`
- `src/mjlab/tasks/tracking/mdp/multi_commands.py`
- `src/mjlab/viewer/debug_visualizer.py`

This means the correct extension point is the distillation task’s command/debug-vis layer, not the generic play script.

## High-Level Design

### Entry point

Continue using:

- `src/mjlab/scripts/play.py`

No dedicated student play script will be added in the first phase.

### Task-local visualization

Student play visualization should live under:

- `src/mjlab/tasks/distillation/mdp/commands.py`

This module will provide task-local logic for:

- extracting student sparse reference signals from the existing motion command
- drawing those signals through `DebugVisualizer`

### Motion source

The distillation task should continue reusing the tracking motion command as the source of truth for motion data. The student visualization must **not** duplicate motion sampling, motion indexing, or time-step progression logic.

## Visualization Content

The first phase should render only the following sparse reference signals.

### 1. End-effector reference points

Render two spheres:

- left end effector reference point
- right end effector reference point

Properties:

- world-frame positions derived from the current reference motion
- no orientation frame in the first phase
- distinct left/right colors
- labels per environment, for example:
  - `student_ref_left_ee_{env}`
  - `student_ref_right_ee_{env}`

### 2. Base linear velocity reference

Render one arrow:

- start: anchor world position
- end: `anchor_pos + base_lin_vel_w * lin_vel_scale`

Properties:

- world-frame direction
- single dedicated color
- first phase uses a fixed scale constant

### 3. Base angular velocity reference

Render one arrow:

- start: anchor world position
- end: `anchor_pos + base_ang_vel_w * ang_vel_scale`

Properties:

- world-frame direction
- single dedicated color distinct from linear velocity
- treated as a directional magnitude cue, not a richer rotational visualization

### 4. Base height reference

Render one vertical height indicator:

- start: `(anchor_x, anchor_y, 0)`
- end: `(anchor_x, anchor_y, anchor_z)`

Representation:

- cylinder in the first phase

Reasoning:

- it expresses height as a scalar more clearly than another arrow

## What Will Not Be Rendered

The first phase student play visualization should **not** render:

- full-body desired frames
- full-body ghost mesh
- desired anchor frame
- end-effector orientation frames
- full motion command visualization from tracking play

This keeps the display aligned with the sparse student command and avoids clutter.

## Module Structure Inside `distillation/mdp/commands.py`

The implementation should be split into three layers.

### Layer 1: sparse reference extraction helpers

These functions return numeric reference values only and do not draw.

Suggested functions:

- `get_student_ee_reference(...)`
- `get_student_base_lin_vel_reference(...)`
- `get_student_base_ang_vel_reference(...)`
- `get_student_base_height_reference(...)`

Responsibilities:

- read the current tracking motion command term
- derive the current sparse reference values
- return world-frame values suitable for visualization

### Layer 2: visualization primitive helpers

These functions convert extracted values into viewer primitives.

Suggested functions:

- `draw_student_ee_reference(...)`
- `draw_student_base_velocity_reference(...)`
- `draw_student_base_height_reference(...)`

Responsibilities:

- call `DebugVisualizer.add_sphere`
- call `DebugVisualizer.add_arrow`
- call `DebugVisualizer.add_cylinder`

### Layer 3: command-level visualization entrypoint

Suggested entrypoint:

- `debug_vis_student_sparse_command(...)`

Responsibilities:

- resolve the active command term
- iterate over the environment indices selected by the viewer
- call the extraction and drawing helpers in a fixed order

## Integration With Play Mode

The distillation task should activate student sparse visualization in its own play configuration.

Recommended integration approach:

- keep `play.py` unchanged or nearly unchanged
- configure the distillation task’s play environment to use student sparse visualization instead of the tracking full reference visualization

This should be done in the distillation task package, not in the global play script.

## Configuration

Only a small set of first-phase configuration values should be introduced.

Suggested configurable values:

- `ee_body_names`
  - default: `("left_wrist_yaw_link", "right_wrist_yaw_link")`
- `lin_vel_scale`
- `ang_vel_scale`
- `ee_sphere_radius`
- `height_radius`

The anchor body should continue to come from the existing command configuration rather than being redefined separately.

First-phase colors may remain hard-coded.

## Data Flow

At each play update:

1. the existing motion command provides the current reference motion state
2. the distillation visualization helper extracts sparse student reference values from that state
3. the helper emits a small set of debug primitives through `DebugVisualizer`
4. the active viewer renders those primitives

This keeps the visualization dependent on the current motion truth source without depending on student observation tensors or replaying sparse observations directly.

## Design Constraints

### Keep `play.py` generic

`play.py` should not know what sparse student references are.

It should continue to do only generic responsibilities:

- load config
- resolve checkpoint
- instantiate runner/policy
- run viewer

### Do not derive visualization from student observations

The visualization should read from the command truth source, not from the student observation tensor. This avoids coupling visualization behavior to observation packing/layout.

### Do not fork the tracking motion command

The motion command remains the teacher/reference truth source. Student visualization is only a thin adapter layered on top.

## Testing Strategy

The first phase should add lightweight tests instead of viewer integration tests.

### Extraction tests

Use mock command data to verify:

- left and right end-effector world positions are extracted correctly
- base linear velocity reference is correct
- base angular velocity reference is correct
- base height reference is correct

### Visualization dispatch tests

Use a mock `DebugVisualizer` to verify:

- `add_sphere` is called twice
- `add_arrow` is called twice
- `add_cylinder` is called once

These tests should validate primitive counts and basic label/routing behavior without requiring a full viewer session.

## First-Phase Non-Goals

The following are explicitly out of scope:

- runtime GUI switching between teacher and student visualization modes
- rendering both full teacher reference and sparse student reference simultaneously
- a standalone `play_distillation_student.py` script
- end-effector orientation frame rendering
- advanced angular velocity visualization beyond a simple arrow cue

## Acceptance Criteria

The first implementation phase is complete when:

- a distilled student checkpoint can be played through the existing `play.py` entrypoint
- the distillation task no longer shows the full tracking reference visualization during play
- the viewer shows:
  - two end-effector reference spheres
  - one base linear velocity arrow
  - one base angular velocity arrow
  - one base height indicator
- the student play visualization is implemented inside the distillation task rather than the generic play script
- extraction and visualization dispatch tests pass
