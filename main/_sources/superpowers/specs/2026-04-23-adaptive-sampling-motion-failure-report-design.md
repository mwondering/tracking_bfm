# Adaptive Sampling Motion Failure Reporting Design

## Goal

Make adaptive sampling behavior easier to interpret by exposing motion-level failure statistics in Weights & Biases.

For each motion in the local motion subset owned by the current rank, define:

- `motion_total_failures = sum(bin_failure_count[motion, :])`
- `motion_total_visits = sum(bin_episode_count[motion, :])`
- `motion_failure_rate = motion_total_failures / motion_total_visits`

The system should publish the highest-failure motions for `rank0` only, using rank-local statistics only. No cross-rank aggregation is required.

## Scope

In scope:

- Compute motion-level failure rates from existing adaptive sampling bin statistics.
- Preserve a stable mapping from motion index to human-readable motion name.
- Expose summary scalars for motion-level failure rates.
- Expose a top-10 worst-motion buffer to W&B without printing it in the terminal log.

Out of scope:

- Cross-rank or global motion aggregation.
- Changing adaptive sampling behavior or probabilities.
- Changing terminal logging format to include motion lists.
- Adding new viewer/debug visualization for these statistics.

## Current State

`MultiMotionCommand` already maintains adaptive sampling bookkeeping:

- `bin_episode_count[motion, bin]`
- `bin_failure_count[motion, bin]`

These are updated during adaptive sampling bookkeeping and are already used to derive per-bin failure rates for sampling.

The motion file ordering is stable because `_resolve_motion_files()` collects and sorts `.npz` files before constructing `MultiMotionLoader`. That ordering is the authoritative mapping used by `motion_idx`.

Training logging currently has two paths:

- `extras["log"]`, which is aggregated by the runner and sent to both terminal and W&B.
- direct writer logging in the runner via `writer.add_scalar(...)` or W&B-native calls.

Because the motion top-10 list should not clutter terminal output, it should not be routed through `extras["log"]`.

## Proposed Design

### 1. Motion identity and display names

`MultiMotionCommand` will retain the resolved motion file list used to initialize `MultiMotionLoader`.

It will derive a display name per motion:

- if `motion_path` is set, use the relative path from `motion_path` to the `.npz` file, with the `.npz` suffix removed
- otherwise, use the file basename without the `.npz` suffix

This keeps names stable and more collision-resistant than basename-only naming.

### 2. Motion-level aggregate statistics

`MultiMotionCommand` will expose helper methods that derive motion-level aggregates from existing bin-level tensors:

- total visits per motion
- total failures per motion
- failure rate per motion

Failure rate computation will clamp the denominator to avoid division by zero, and motions with zero visits will resolve to failure rate `0.0`.

These helpers will be pure derivations over the existing bookkeeping tensors. No new adaptive sampling state machine is needed.

### 3. Top-10 worst-motion snapshot

`MultiMotionCommand` will expose a snapshot builder that returns the top 10 motions ranked by:

1. descending motion failure rate
2. descending total failures
3. ascending motion index

Each row will contain:

- `motion_rank` within the top-10 list
- `motion_index`
- `motion_name`
- `failure_rate`
- `total_failures`
- `total_visits`

Only motions from the current process's local motion subset are considered.

### 4. W&B-only reporting path

The tracking runner will own reporting.

At training-log time, if all of the following are true:

- logger type is W&B
- logging is enabled
- current process is `rank0`
- the active `motion` command term is `MultiMotionCommand`

then the runner will:

- write scalar summaries with `writer.add_scalar(...)`
- publish the top-10 snapshot as a `wandb.Table`

This reporting will bypass `extras["log"]`, so the top-10 list does not appear in terminal output.

### 5. Logged outputs

Scalars:

- `Train/adaptive_sampling/motion_failure_rate_mean`
- `Train/adaptive_sampling/motion_failure_rate_max`
- `Train/adaptive_sampling/motion_failure_rate_top10_min`

Table:

- `Train/adaptive_sampling/top10_motion_failure_rate`

Recommended table columns:

- `rank`
- `motion_name`
- `motion_index`
- `failure_rate`
- `total_failures`
- `total_visits`

The `rank` column here means position inside the sorted top-10 table, not distributed rank.

## Data Flow

1. Adaptive sampling continues updating `bin_episode_count` and `bin_failure_count` inside `MultiMotionCommand`.
2. At runner logging time, the runner reads the active motion command term.
3. If the term is multi-motion adaptive sampling, the runner requests:
   - motion-level aggregate stats
   - top-10 worst-motion snapshot
4. The runner writes summary scalars and one W&B table for the current iteration.
5. Terminal log behavior remains unchanged.

## Error Handling

- If the active motion term is not `MultiMotionCommand`, the runner skips this reporting.
- If logger type is not W&B, the runner skips the table report entirely.
- If fewer than 10 motions exist locally, the table contains only the available motions.
- If all motions have zero visits, all failure rates remain `0.0`, and the table still reports a deterministic top subset.

## Testing

Add unit tests covering:

- motion-level aggregation from known bin counts
- top-10 ranking order
- stable mapping from motion index to motion display name
- zero-visit behavior produces finite failure rates
- runner-side payload construction for W&B table rows without requiring a real networked W&B session

Testing should avoid network calls and should assert payload shape and values rather than real upload behavior.

## Tradeoffs

Chosen design benefits:

- minimal change to adaptive sampling logic
- no cross-rank synchronization
- no terminal spam
- clear separation between statistics ownership (command term) and reporting ownership (runner)

Accepted limitations:

- reported top-10 is only for `rank0` local motions, not the global dataset
- table history may grow in W&B across iterations, which is acceptable for this debugging-oriented feature

## Implementation Notes

- Prefer adding read-only helper methods on `MultiMotionCommand` rather than mutating its reset/log pathways.
- Keep the W&B-specific code in the tracking runner so non-W&B loggers remain unaffected.
- Do not serialize the top-10 list into scalar tags; use a table for names plus rates.
