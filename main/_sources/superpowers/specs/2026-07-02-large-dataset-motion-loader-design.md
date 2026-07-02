# Large Dataset Motion Loader Design

## Goal

Add an opt-in tracking motion command that can train on motion datasets larger than GPU memory by keeping the full motion dataset on CPU/disk and staging only an active per-rank subset on GPU. The original `multi_commands.py` path must keep its current behavior and adaptive sampling semantics.

## Current Problem

`src/mjlab/tasks/tracking/mdp/multi_commands.py` constructs `MultiMotionLoader` by loading every motion field from every `.npz` directly onto `self.device`, then concatenates all frames into large GPU tensors. This is fast for small datasets, but it makes GPU memory scale with the full dataset size.

The existing adaptive sampling logic is valuable and should be preserved:

- Bin statistics are tracked by motion id and bin id.
- Resets sample motion/bin pairs from failure-rate-weighted probabilities mixed with uniform sampling.
- Failures observed during environment resets update the corresponding bin.
- Windowed failure-rate state, probability clipping, bin weights, and per-motion probability caps should keep the same semantics where possible.

## Architecture

The new command lives in `src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py` and is opt-in through its own config class. It reuses the public command API expected by observations, rewards, terminations, and visualization, but changes the motion storage layer.

Storage is split into three layers:

1. Full dataset metadata:
   - Motion file paths.
   - Per-motion lengths and fps.
   - Global bin counts, valid bin ids, bin lengths, and bin weights.
   - These are small and can live on GPU as tensors.

2. Full dataset frame data:
   - Loaded lazily from `.npz` files into CPU tensors or memory-mapped arrays.
   - This data is not concatenated into a full GPU tensor.

3. Active GPU subset:
   - A per-rank set of unique active global motion ids.
   - Default size: `active_subset_size=20000`.
   - Reset and per-step reference gathering only read from active subset GPU tensors.
   - Subset refresh defaults to `subset_refresh_count=10` motions per learning iteration.

## Active Subset Policy

Each active subset slot maps one local slot id to one global motion id. Within one rank, active global motion ids must be unique. Cross-rank duplicates are allowed.

Each iteration may refresh a small number of slots. A slot can be evicted only when:

- `slot_ref_count == 0`: no environment currently uses this motion.
- `slot_ready == True`: slot is not mid-load.
- `iteration - slot_loaded_iteration >= subset_min_resident_iterations`.

The default `subset_min_resident_iterations` is 50. This intentionally lets hard or frequently used motions stay resident longer.

Incoming motion ids are sampled from the full dataset excluding active and pending ids. Sampling uses a global motion-level score derived from the full bin failure statistics, mixed with uniform probability so cold or easy motions can still enter the subset.

## Adaptive Sampling

The adaptive sampling reset domain is the active ready subset, not the full dataset. The statistics domain remains the full dataset.

At reset time:

- Build active valid `(global_motion_id, global_bin_id)` pairs from active ready motions.
- Use the same probability recipe as the original command:
  - failure rate = `bin_failure_count / bin_episode_count`
  - clip by `mean * adaptive_failure_rate_max_over_mean`
  - normalize failure-based probability
  - mix with uniform by `adaptive_uniform_ratio`
  - multiply by bin weights
  - apply max-probability constraints
- Store sampled global motion ids in `motion_idx`.
- Store sampled local subset slot ids in a separate lookup for fast frame gather.

When collecting statistics:

- Local rollout deltas are accumulated by global motion id and global bin id.
- Counts are not subset-local.
- Windowed adaptive sampling increments are tracked in the same global bin coordinate system.

## Multi-GPU Synchronization

For distributed training, each rank maintains local bin deltas during rollout. At the beginning of each learning iteration, the command:

1. Advances adaptive sampling windows.
2. Synchronizes local episode/failure deltas across ranks with `torch.distributed.all_reduce(SUM)`.
3. Applies the reduced deltas to every rank's global bin statistics.
4. Clears local deltas.
5. Refreshes the active subset.

When distributed is not initialized, synchronization is a local no-op and the same local deltas are applied.

## Timing Output

The new command exposes a timing hook:

- `get_large_dataset_timing_stats() -> dict[str, float]`

The tracking runner keeps its existing `logger.log(...)` call unchanged. After that call, it checks whether the motion command exposes the timing hook. If present, it prints one extra line beside the existing `collect_time` and `learn_time` output:

```text
[LargeDatasetMotion] iter=123 collect_time=3.812s learning_time=1.044s global_bin_update=0.006s subset_update=0.018s
```

If a writer exists, the runner also writes:

- `Perf/global_bin_update_time`
- `Perf/subset_update_time`

This avoids changing the fixed `rsl_rl` logger signature and keeps old commands unaffected.

## Files

New:

- `src/mjlab/tasks/tracking/mdp/multi_command_largedataset.py`
- `tests/test_large_dataset_motion_command.py`
- `docs/superpowers/specs/2026-07-02-large-dataset-motion-loader-design.md`
- `docs/superpowers/plans/2026-07-02-large-dataset-motion-loader.md`

Modified:

- `src/mjlab/tasks/tracking/rl/runner.py`
- `src/mjlab/tasks/tracking/wbteleop/runner.py`
- `tests/test_tracking_runner_adaptive_window.py`
- `tests/test_wbteleop_task.py`

The original `src/mjlab/tasks/tracking/mdp/multi_commands.py` is not modified for behavior.

## Non-Goals

- No cross-rank uniqueness for active subset motions.
- No async disk prefetch thread in the first implementation.
- No change to existing `multi_commands.py` or its config defaults.
- No automatic migration of existing configs to the large-dataset command.

## Verification

The implementation is complete when:

- Unit tests prove active subset uniqueness, eviction constraints, min residence, and no duplicate pending/active ids.
- Unit tests prove active adaptive sampling uses only active ready motions but updates global bin ids.
- Unit tests prove distributed sync is a no-op without distributed initialization and uses all-reduce when distributed is available.
- Runner tests prove the optional timing hook prints and logs only when present.
- Existing adaptive sampling tests for `multi_commands.py` still pass.
