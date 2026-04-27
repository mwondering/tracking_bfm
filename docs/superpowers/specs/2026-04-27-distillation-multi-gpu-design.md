# Distillation Multi-GPU Design

## Goal

Enable true multi-GPU training for `src/mjlab/tasks/distillation` when launched through the existing multi-process training path. Each GPU should collect rollouts from its local environment shard, but all ranks must optimize the same student policy with synchronized gradients and a shared parameter state.

## Non-Goals

- Rewriting distillation to use PPO rollout storage.
- Refactoring distillation to inherit from `MjlabOnPolicyRunner`.
- Adding multi-node support beyond the existing single-host `torchrunx` launch path.
- Changing teacher policy behavior beyond what is required for rank-local inference.

## Current Problem

`src/mjlab/scripts/train.py` already launches multiple processes for multi-GPU runs, but `DistillationRunner` and `ActionDistillationAlgorithm` are single-process implementations. In a multi-rank launch today, each rank trains its own student copy independently and writes to the same log directory/checkpoint paths.

## Proposed Design

### Runner Responsibilities

`DistillationRunner` will gain a lightweight distributed configuration path modeled after `rsl_rl`:

- Detect `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`.
- Validate that the configured device matches `cuda:{LOCAL_RANK}` when distributed mode is active.
- Initialize:
  - `is_distributed`
  - `gpu_world_size`
  - `gpu_global_rank`
  - `gpu_local_rank`
  - `multi_gpu_cfg`
  - `disable_logs` for non-zero ranks
- Keep rank-local environment collection unchanged.
- Broadcast student parameters once before the training loop starts so all ranks begin from rank 0 state.
- Restrict TensorBoard/W&B/Neptune writes and checkpoint saves to rank 0.

### Algorithm Responsibilities

`ActionDistillationAlgorithm` will accept an optional `multi_gpu_cfg` and expose two distributed helpers:

- `broadcast_parameters()`
- `reduce_parameters()`

Update behavior per mini-batch:

1. Compute local loss on each rank from local rollout data.
2. Run `backward()` locally.
3. All-reduce gradients across ranks and divide by world size.
4. Clip gradients.
5. Run `optimizer.step()` identically on every rank.

This preserves the current optimizer structure while turning the effective batch into the union of all local rank batches.

### Metrics and Logging

- Non-zero ranks will not create writers or save checkpoints.
- Loss scalars and rollout mix metrics will be averaged across ranks before rank 0 logs them.
- Throughput logging will continue to report per-iteration collection/learn times from the active process. The displayed collection size should be scaled by `gpu_world_size` so total step throughput reflects all ranks.

### Checkpointing

- Only rank 0 writes model checkpoints.
- Checkpoint format remains unchanged.
- Resume continues to load the same checkpoint on each rank, then training start will broadcast rank 0 parameters so all workers are aligned.

## Error Handling

- Distributed mode must fail early if the runner device does not match `LOCAL_RANK`.
- Distributed reduction helpers must no-op cleanly in single-GPU/CPU mode.
- Gradient reduction must skip parameters with `grad is None`.

## Testing Strategy

### Unit Tests

Extend distillation tests to cover:

- Single-process behavior remains unchanged.
- `broadcast_parameters()` becomes a no-op in non-distributed mode.
- In distributed mode, the algorithm calls `broadcast_object_list` during sync and `all_reduce` on non-`None` gradients during updates.
- Non-zero ranks disable logging and checkpoint writes.
- Aggregated logging metrics are reduced across ranks before emission.

### Smoke Coverage

Keep the existing runner smoke path intact and adapt assertions only where distributed-aware fields are introduced.

## Risks

- Logging aggregation can silently drift if only some metrics are reduced; implementation must keep all logged scalar paths consistent.
- Per-rank observation normalization updates occur from local batches before gradient sync. This is acceptable for the first version because normalization state already lives with the local model and is synchronized via initial/load-time parameter alignment rather than per-step DDP buffers.

## Implementation Scope

The change is limited to:

- `src/mjlab/tasks/distillation/rl/runner.py`
- `src/mjlab/tasks/distillation/rl/algorithm.py`
- distillation tests that validate runner and algorithm behavior

No task config or launch CLI changes are required.
