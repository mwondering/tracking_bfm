# RoboJuDo Pico Sparse Deploy Design

**Date:** 2026-05-08

**Goal**

Integrate `tracking_bfm` sparse-controller policies into `RoboJuDo` for Pico-driven deployment, with phase-one acceptance scoped to `lightweight Pico -> sim2sim` on G1. The deployment contract must support both tracking and distillation-student checkpoints through a unified ONNX export path and a single RoboJuDo sparse-policy runtime.

**Scope**

Phase one includes:

- unified ONNX export from `tracking_bfm` tracking and distillation checkpoints
- ONNX export with embedded observation normalizer
- RoboJuDo sparse ONNX policy for G1 sim2sim
- RoboJuDo lightweight Pico controller that maps Pico inputs directly to sparse commands
- MuJoCo visualization for sparse references needed to debug command alignment

Phase one excludes:

- Pico retarget controller
- sim2real deployment
- motion-library or time-indexed tracker ONNX pipelines
- policy switching

## Architecture

### Repository responsibilities

`tracking_bfm` owns training-time model reconstruction and ONNX export. It is the only place that knows how to interpret training checkpoints and rebuild the correct actor module.

`RoboJuDo` owns runtime deployment. It reads Pico input, reconstructs sparse observations from controller and robot state, runs ONNX inference, and applies resulting actions in MuJoCo.

`deploy_pico` remains a reference source for Pico SDK integration patterns and command-mapping heuristics, but phase-one code lands directly in `RoboJuDo`.

### Deployment contract

The deployment model is a pure actor ONNX:

- input: `obs` tensor
- output: `action` tensor

The ONNX graph includes:

- actor weights
- observation normalizer

The ONNX graph does not include:

- motion references
- time-step inputs
- body trajectories
- controller logic
- robot action semantics beyond the actor itself

The ONNX file is written into the source checkpoint directory. If no explicit output name is passed, the default filename is:

- `deploy_<checkpoint_stem>.onnx`

Example:

- `model_5000.pt -> deploy_model_5000.onnx`

### Metadata minimization

The ONNX should remain minimal. Input and output dimensions are inferred directly from the graph. Observation-term definitions should be recovered from existing YAML/config sources rather than duplicated into a large sidecar manifest.

Minimal ONNX metadata is limited to identity and sanity-check fields:

- `task_id`
- `obs_group`
- `checkpoint_family`
- optional `robot_name`

No large manifest is required in phase one.

### Sparse observation reconstruction

RoboJuDo reconstructs deployment observations from two sources:

- Pico-derived sparse command data from the controller
- robot state history from the environment

The sparse command portion contains:

- `ee_pose`
- `base_lin_vel_b`
- `base_ang_vel_b`
- `anchor_height_w`

The robot-state portion contains:

- `projected_gravity`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `last_action`

The exact observation term order and history settings are not hardcoded into ONNX metadata. They are recovered from the checkpoint directory configuration already associated with the exported `.pt`.

## Unified export design

### Checkpoint families

Two checkpoint families must be supported:

- tracking checkpoints using `actor_state_dict`
- distillation checkpoints using `policy_state_dict`

They share the same deployment semantics for phase one: sparse actor inference for G1.

### Export flow

1. Inspect checkpoint family.
2. Rebuild the correct actor module for the requested task.
3. Load weights through a checkpoint-family adapter.
4. Export a pure actor ONNX with embedded normalizer.
5. Attach minimal ONNX metadata.
6. Save the ONNX next to the source checkpoint.

### Export CLI

The export CLI should be simple and deployment-oriented:

```bash
uv run export-tracking-bfm-onnx \
  --checkpoint /path/to/model.pt \
  --task-id Mjlab-Trackingbfm-Flat-Unitree-G1 \
  --output-name deploy_custom.onnx
```

Optional arguments:

- `--checkpoint-family auto|tracking|distillation`
- `--obs-group actor|student_actor`
- `--output-name <name>.onnx`

If `--output-name` is omitted, the exporter uses `deploy_<checkpoint_stem>.onnx`.

## RoboJuDo runtime design

### Controller

A new `PicoLightSparseCtrl` module maps Pico inputs directly to sparse commands.

Responsibilities:

- initialize and read Pico SDK data
- convert Unity-frame inputs into robot-aligned coordinates
- maintain teleop state machine: idle, active, pause, exit
- handle anchor/reset behavior
- map lightweight controller input to sparse commands
- provide structured controller output to the pipeline

It does not:

- assemble final actor observations
- run ONNX inference
- interpret robot action semantics

### Policy

A new `TrackingBfmSparseOnnxPolicy` module performs sparse-actor deployment.

Responsibilities:

- load the ONNX model
- recover sparse observation configuration from checkpoint-side YAML/config
- maintain robot-state history buffers
- assemble the final actor observation tensor
- run ONNX inference
- map model output to policy actions / PD targets
- render sparse reference debug visualization

It does not:

- talk to Pico SDK directly
- implement controller state transitions

### Pipeline and environment

Phase one reuses the existing RoboJuDo `RlPipeline` and MuJoCo environment path. New functionality should slot into existing `controller -> policy -> env` boundaries without introducing a bespoke deployment pipeline.

## Visualization

Phase-one visualization is intentionally minimal and diagnostic. It should expose the sparse references that are most useful for detecting frame/sign/scale mistakes:

- end-effector target markers
- base linear velocity arrow

This visualization belongs in policy debug visualization, since the policy owns the reconstructed sparse-observation semantics.

## Acceptance criteria

Phase one is accepted when all of the following are true:

- a tracking checkpoint exports to a deployable ONNX in-place
- a distillation-student checkpoint exports to a deployable ONNX in-place
- the exported ONNX contains the observation normalizer
- RoboJuDo can load the exported ONNX and run G1 MuJoCo sim2sim
- RoboJuDo lightweight Pico control drives the robot through sparse commands
- at least sparse EE targets and base velocity are visible in debug visualization

## Risks

Primary risks:

- Pico coordinate frame mismatch vs training-time sparse-command semantics
- `ee_pose` meaning mismatch between lightweight mapping and training references
- checkpoint-family reconstruction differences causing incorrect export
- environment-side action semantics not matching training joint order
- SDK environment conflicts between Pico tooling and RoboJuDo runtime

Mitigations:

- keep the ONNX contract minimal and stable
- keep sparse-command mapping conservative in phase one
- fail fast on observation/action dimension mismatch
- add targeted tests around export, obs assembly, and controller mapping

