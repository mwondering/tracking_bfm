# Large Dataset Motion Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an opt-in large-dataset tracking motion command that keeps full motion data off GPU, stages an active unique subset on GPU, synchronizes global adaptive bin stats, and prints per-iteration timing.

**Architecture:** Add a new `multi_command_largedataset.py` module with dataset metadata, active subset, and command classes. Keep the original `multi_commands.py` behavior intact. Add optional runner timing hooks that activate only when the command exposes large-dataset timing stats.

**Tech Stack:** Python, PyTorch, NumPy `.npz` loading, optional `torch.distributed`, pytest.

---

### Task 1: Active Subset Unit Tests

**Files:**
- Create: `tests/test_large_dataset_motion_command.py`
- Create/modify later: `src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py`

- [ ] Write tests for unique active ids, pending exclusion, min residence, and ref-count-protected eviction.
- [ ] Run `uv run pytest tests/test_large_dataset_motion_command.py -q` and confirm these tests fail because the module/classes are missing.
- [ ] Implement `ActiveMotionSubset` and its pure tensor/list bookkeeping methods.
- [ ] Re-run `uv run pytest tests/test_large_dataset_motion_command.py -q` and confirm the subset tests pass.

### Task 2: Motion Metadata and CPU Store

**Files:**
- Modify: `src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py`
- Modify: `tests/test_large_dataset_motion_command.py`

- [ ] Add tests that build tiny `.npz` motions and verify metadata is loaded without concatenating all frames to GPU.
- [ ] Add tests that loading selected motions produces GPU tensors only for requested motion ids.
- [ ] Implement `LargeDatasetMotionStore` for metadata discovery and selected-motion loading.
- [ ] Re-run `uv run pytest tests/test_large_dataset_motion_command.py -q`.

### Task 3: Global Adaptive Bin Pool

**Files:**
- Modify: `src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py`
- Modify: `tests/test_large_dataset_motion_command.py`

- [ ] Add tests for global bin delta accumulation and local no-op sync when distributed is unavailable.
- [ ] Add tests for active-only pair probability using global bin statistics.
- [ ] Implement `GlobalAdaptiveBinPool` helpers and probability calculations matching `MultiMotionCommand`.
- [ ] Re-run `uv run pytest tests/test_large_dataset_motion_command.py -q`.

### Task 4: Large Dataset Command

**Files:**
- Modify: `src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py`

- [ ] Add `LargeDatasetMultiMotionCommandCfg` and aliases `MotionCommandCfg`, `MotionCommand`.
- [ ] Implement `LargeDatasetMultiMotionCommand` by subclassing `MultiMotionCommand`, replacing initialization, gather, active sampling, stats accumulation, subset refresh, and timing hooks.
- [ ] Keep public properties inherited from `MultiMotionCommand` working by overriding `_gather_motion_field` and frame-index mapping.
- [ ] Re-run `uv run pytest tests/test_large_dataset_motion_command.py -q`.

### Task 5: Runner Timing Hook

**Files:**
- Modify: `src/mjlab/tasks/tracking/rl/runner.py`
- Modify: `src/mjlab/tasks/tracking/wbteleop/runner.py`
- Modify: `tests/test_tracking_runner_adaptive_window.py`
- Modify: `tests/test_wbteleop_task.py`

- [ ] Add tests proving timing output is printed when the command exposes `get_large_dataset_timing_stats`.
- [ ] Add tests proving old commands without the hook do not print extra timing.
- [ ] Implement optional runner helper methods for printing and writer scalar logging.
- [ ] Re-run runner tests:
  - `uv run pytest tests/test_tracking_runner_adaptive_window.py -q`
  - `uv run pytest tests/test_wbteleop_task.py -q`

### Task 6: Regression Verification

**Files:**
- No required file changes.

- [ ] Run `uv run pytest tests/test_multi_motion_command_sampling.py tests/test_large_dataset_motion_command.py tests/test_tracking_runner_adaptive_window.py tests/test_wbteleop_task.py -q`.
- [ ] Run `git diff -- src/mjlab/tasks/tracking/mdp/multi_commands.py` and confirm it is empty.
- [ ] Inspect `git status --short` and summarize changed files.

### Completion Criteria

- New large-dataset command is opt-in and importable from `mjlab.tasks.tracking.mdp.multi_command_largedataset`.
- Original `multi_commands.py` has no behavior changes.
- Adaptive sampling in the new command uses active subset for resets and global full-dataset bin stats for difficulty.
- Per-iteration timing prints `global_bin_update` and `subset_update` beside collection and learning times.
- Targeted tests pass.
