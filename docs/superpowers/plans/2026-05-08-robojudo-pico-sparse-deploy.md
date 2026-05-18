# RoboJuDo Pico Sparse Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build phase-one support for lightweight Pico-driven G1 sim2sim deployment in RoboJuDo using sparse-controller ONNX exports from tracking_bfm.

**Architecture:** tracking_bfm exports a pure-actor ONNX with embedded observation normalization for both tracking and distillation checkpoints. RoboJuDo adds a sparse ONNX policy and a lightweight Pico controller, then composes them through its existing MuJoCo RL pipeline.

**Tech Stack:** Python, PyTorch, ONNX, onnxruntime, MuJoCo, RoboJuDo, xrobotoolkit_sdk, pytest, uv

---

### Task 1: Add Unified ONNX Export Coverage in tracking_bfm

**Files:**
- Create: `src/mjlab/deployment/checkpoint_inspector.py`
- Create: `src/mjlab/deployment/tracking_bfm_onnx_export.py`
- Create: `src/mjlab/scripts/export_tracking_bfm_onnx.py`
- Modify: `pyproject.toml`
- Test: `tests/test_tracking_bfm_onnx_export.py`

- [ ] **Step 1: Write the failing export tests**

Add tests that verify:
- tracking checkpoints are recognized
- distillation checkpoints are recognized
- default ONNX filename is `deploy_<stem>.onnx`
- explicit output name overrides the default

- [ ] **Step 2: Run the export tests to verify they fail**

Run: `uv run pytest tests/test_tracking_bfm_onnx_export.py -v`
Expected: FAIL because exporter modules and CLI do not exist yet.

- [ ] **Step 3: Implement checkpoint inspection and ONNX export**

Build a minimal deployment exporter that:
- detects checkpoint family by keys
- rebuilds the correct actor path
- embeds the actor normalizer in ONNX
- writes the ONNX next to the checkpoint

- [ ] **Step 4: Run the export tests to verify they pass**

Run: `uv run pytest tests/test_tracking_bfm_onnx_export.py -v`
Expected: PASS.

### Task 2: Validate Embedded Normalizer and Minimal Metadata

**Files:**
- Modify: `src/mjlab/deployment/tracking_bfm_onnx_export.py`
- Test: `tests/test_tracking_bfm_onnx_export.py`

- [ ] **Step 1: Write the failing normalizer test**

Add a test that exports a model and inspects the ONNX graph/metadata to verify:
- normalizer is embedded in the ONNX graph
- input/output shapes are inferable from ONNX
- only minimal identity metadata is attached

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `uv run pytest tests/test_tracking_bfm_onnx_export.py -k normalizer -v`
Expected: FAIL because the graph/metadata checks are not implemented yet.

- [ ] **Step 3: Implement the minimal metadata policy**

Ensure the exporter writes only:
- `task_id`
- `obs_group`
- `checkpoint_family`
- optional `robot_name`

- [ ] **Step 4: Re-run the targeted test**

Run: `uv run pytest tests/test_tracking_bfm_onnx_export.py -k normalizer -v`
Expected: PASS.

### Task 3: Add RoboJuDo Sparse ONNX Policy Skeleton

**Files:**
- Create: `robojudo/policy/tracking_bfm_sparse_onnx_policy.py`
- Modify: `robojudo/policy/__init__.py`
- Modify: `robojudo/policy/policy_cfgs.py`
- Test: `tests/test_tracking_bfm_sparse_onnx_policy.py`

- [ ] **Step 1: Write the failing policy-loading tests**

Add tests that verify the policy:
- loads an ONNX file
- infers input/output dimensions from ONNX
- rejects dimension mismatches cleanly

- [ ] **Step 2: Run the policy tests to verify they fail**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py -v`
Expected: FAIL because the policy class does not exist yet.

- [ ] **Step 3: Implement the ONNX sparse policy loader**

Implement a minimal policy that:
- opens ONNX via onnxruntime
- inspects input/output specs
- exposes the standard RoboJuDo policy interface

- [ ] **Step 4: Re-run the policy tests**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py -v`
Expected: PASS.

### Task 4: Assemble Sparse Observations in RoboJuDo

**Files:**
- Modify: `robojudo/policy/tracking_bfm_sparse_onnx_policy.py`
- Test: `tests/test_tracking_bfm_sparse_onnx_policy.py`

- [ ] **Step 1: Write the failing observation-assembly tests**

Add tests that verify:
- sparse command terms are concatenated in the expected order
- robot-state history is appended correctly
- the assembled observation dimension matches ONNX input dimension

- [ ] **Step 2: Run the observation tests to verify they fail**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py -k observation -v`
Expected: FAIL because observation assembly is incomplete.

- [ ] **Step 3: Implement sparse observation assembly**

Use checkpoint-side config recovery plus RoboJuDo state buffers to assemble:
- `ee_pose`
- `base_lin_vel_b`
- `base_ang_vel_b`
- `anchor_height_w`
- `projected_gravity`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `last_action`

- [ ] **Step 4: Re-run the observation tests**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py -k observation -v`
Expected: PASS.

### Task 5: Add Lightweight Pico Sparse Controller

**Files:**
- Create: `robojudo/controller/pico_light_sparse_ctrl.py`
- Modify: `robojudo/controller/__init__.py`
- Modify: `robojudo/controller/ctrl_cfgs.py`
- Test: `tests/test_pico_light_sparse_ctrl.py`

- [ ] **Step 1: Write the failing Pico controller tests**

Add tests that verify:
- state-machine transitions
- base command sign conventions
- anchor-height control behavior
- EE anchor/reset behavior

- [ ] **Step 2: Run the Pico controller tests to verify they fail**

Run: `pytest tests/test_pico_light_sparse_ctrl.py -v`
Expected: FAIL because the controller does not exist yet.

- [ ] **Step 3: Implement Pico lightweight sparse mapping**

Implement:
- Pico SDK input reader abstraction
- state machine
- coordinate conversion
- lightweight sparse command mapping

- [ ] **Step 4: Re-run the Pico controller tests**

Run: `pytest tests/test_pico_light_sparse_ctrl.py -v`
Expected: PASS.

### Task 6: Compose the G1 Sim2Sim Config in RoboJuDo

**Files:**
- Create: `robojudo/config/g1/policy/g1_tracking_bfm_sparse_onnx_policy_cfg.py`
- Create: `robojudo/config/g1/ctrl/g1_pico_light_sparse_ctrl_cfg.py`
- Modify: `robojudo/config/g1/g1_cfg.py`
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_tracking_bfm_g1_config.py`

- [ ] **Step 1: Write the failing config test**

Add a test that verifies a config named `g1_tracking_bfm_pico_light_sim` resolves to:
- MuJoCo env
- Pico sparse controller
- tracking_bfm sparse ONNX policy

- [ ] **Step 2: Run the config test to verify it fails**

Run: `pytest tests/test_tracking_bfm_g1_config.py -v`
Expected: FAIL because the config does not exist yet.

- [ ] **Step 3: Implement the config composition**

Wire the new policy and controller into G1 config registration without disrupting existing configs.

- [ ] **Step 4: Re-run the config test**

Run: `pytest tests/test_tracking_bfm_g1_config.py -v`
Expected: PASS.

### Task 7: Add Sparse Debug Visualization

**Files:**
- Modify: `robojudo/policy/tracking_bfm_sparse_onnx_policy.py`
- Modify: `robojudo/environment/utils/mujoco_viz.py`
- Test: `tests/test_tracking_bfm_sparse_onnx_policy.py`

- [ ] **Step 1: Write the failing debug-viz test**

Add a focused test that verifies debug-viz helpers are called with:
- EE target markers
- base velocity vector

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py -k debug_viz -v`
Expected: FAIL because visualization hooks are not implemented.

- [ ] **Step 3: Implement sparse debug visualization**

Add minimal policy debug visualization for:
- left and right EE targets
- base linear velocity arrow

- [ ] **Step 4: Re-run the targeted test**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py -k debug_viz -v`
Expected: PASS.

### Task 8: End-to-End Verification

**Files:**
- Modify: `tests/test_tracking_bfm_onnx_export.py`
- Modify: `tests/test_tracking_bfm_sparse_onnx_policy.py`
- Modify: `tests/test_pico_light_sparse_ctrl.py`

- [ ] **Step 1: Run tracking_bfm export verification**

Run: `uv run pytest tests/test_tracking_bfm_onnx_export.py -v`
Expected: PASS.

- [ ] **Step 2: Run RoboJuDo policy/controller verification**

Run: `pytest tests/test_tracking_bfm_sparse_onnx_policy.py tests/test_pico_light_sparse_ctrl.py tests/test_tracking_bfm_g1_config.py -v`
Expected: PASS.

- [ ] **Step 3: Run focused type/lint checks where available**

Run in `tracking_bfm`: `uv run pyright src/mjlab/deployment src/mjlab/scripts/export_tracking_bfm_onnx.py`
Expected: PASS.

Run in `RoboJuDo`: `pytest tests/test_full_imports.py -v`
Expected: PASS.

