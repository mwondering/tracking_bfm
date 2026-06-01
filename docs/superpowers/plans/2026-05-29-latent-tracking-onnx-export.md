# Latent Tracking ONNX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated CLI and helper module that exports a latent tracking actor plus frozen latent decoder as one deployable ONNX model.

**Architecture:** Add a focused deployment module that rebuilds latent tracking checkpoints through the existing task registry and exports a flat-tensor wrapper. The wrapper composes `actor.as_onnx()` with `decoder.decoder.as_onnx()` and applies latent action clipping between them.

**Tech Stack:** Python, PyTorch, ONNX, onnxruntime, rsl-rl, tensordict, pytest, uv

---

### Task 1: Combined Model Unit Tests

**Files:**
- Create: `tests/test_latent_tracking_bfm_onnx_export.py`

- [ ] **Step 1: Write failing tests**

Create tests that build small in-memory `MLPModel` actor/decoder modules, export through the new helper, and assert ONNX runtime parity and metadata.

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_latent_tracking_bfm_onnx_export.py -v`

Expected: FAIL because `mjlab.deployment.latent_tracking_bfm_onnx_export` does not exist.

### Task 2: Deployment Helper

**Files:**
- Create: `src/mjlab/deployment/latent_tracking_bfm_onnx_export.py`
- Modify: `src/mjlab/deployment/__init__.py`

- [ ] **Step 1: Implement path resolution and metadata helpers**

Add `resolve_latent_deploy_onnx_path()` and minimal metadata attachment.

- [ ] **Step 2: Implement `LatentTrackingDeploymentOnnxModel`**

Compose actor ONNX export with decoder ONNX export:

```text
obs -> actor -> clamp -> concat(proprio, latent) -> decoder -> actions
```

- [ ] **Step 3: Implement actor/decoder rebuild**

Load task env/runner config, inject `latent_decoder_checkpoint_path`, apply motion source override, instantiate `LatentTrackingOnPolicyRunner`, load the latent tracking checkpoint, and return actor plus decoder.

- [ ] **Step 4: Implement `export_latent_tracking_checkpoint_to_onnx()`**

Export the combined model with input names `obs` and `proprio`, output name `actions`, opset 18, and `dynamo=False`.

- [ ] **Step 5: Verify helper tests pass**

Run: `uv run pytest tests/test_latent_tracking_bfm_onnx_export.py -v`

Expected: PASS.

### Task 3: CLI Entry Point

**Files:**
- Create: `src/mjlab/scripts/export_latent_tracking_bfm_onnx.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add CLI parser**

Expose `--checkpoint`, `--decoder-checkpoint`, `--task-id`, `--motion-path`, `--motion-file`, `--obs-group`, `--proprio-obs-group`, `--latent-action-clip`, `--output-name`, `--robot-name`, `--device`, and `--verbose`.

- [ ] **Step 2: Register project script**

Add `export-latent-tracking-bfm-onnx = "mjlab.scripts.export_latent_tracking_bfm_onnx:main"` to `[project.scripts]`.

- [ ] **Step 3: Verify help works**

Run: `uv run export-latent-tracking-bfm-onnx --help`

Expected: command prints usage and lists all arguments.

### Task 4: Final Verification

**Files:**
- Test: `tests/test_latent_tracking_bfm_onnx_export.py`
- Test: `tests/test_tracking_bfm_onnx_export.py`

- [ ] **Step 1: Run focused export tests**

Run: `uv run pytest tests/test_latent_tracking_bfm_onnx_export.py tests/test_tracking_bfm_onnx_export.py -v`

Expected: PASS.

- [ ] **Step 2: Check git diff**

Run: `git diff -- src/mjlab/deployment/latent_tracking_bfm_onnx_export.py src/mjlab/scripts/export_latent_tracking_bfm_onnx.py pyproject.toml tests/test_latent_tracking_bfm_onnx_export.py`

Expected: diff only contains the latent ONNX exporter feature.

