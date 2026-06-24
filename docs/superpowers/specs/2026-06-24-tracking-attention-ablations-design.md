# Tracking Attention Ablations Design

## Goal

Add three attention-based actor variants for the G1 BFM TestOptimal tracking
task, while keeping the existing PPO algorithm, rollout storage, runner, critic,
reward design, and task behavior intact.

The purpose is to compare attention mechanisms for motion tracking under the
same optimality-probe task used by `scripts/train_test_optimal.sh`.

## Non-Goals

- Do not modify shared `src/mjlab/rl` configuration or runner code.
- Do not change PPO, rollout storage, reward computation, motion sampling, or
  environment stepping.
- Do not use future reference frames.
- Do not implement MaskedMimic-style masked inpainting, VAE training, scene
  point tokens, or PointNet scene encoders.
- Do not change the critic architecture for these attention ablations.

## Scope

Runtime implementation should stay under `src/mjlab/tasks/tracking`.
Tests may be added under `tests/`.

The minimal expected code footprint is:

- `src/mjlab/tasks/tracking/rl/attention_models.py`
- `src/mjlab/tasks/tracking/config/g1/attention_cfg.py`
- `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`
- `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
- `src/mjlab/tasks/tracking/config/g1/__init__.py`
- `tests/test_tracking_attention_models.py`
- focused task/script tests if needed

## Existing Entry Point

`scripts/train_test_optimal.sh` already forces:

```bash
--env.commands.motion.history_steps 0
--env.commands.motion.future_steps 1
```

This should remain the default for the attention ablations. Reference command
history and future windows from `MultiMotionCommand` are not used. Temporal
history comes from observation history buffers.

The new tasks are selected by overriding `TASK`, for example:

```bash
TASK=Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttn-NoRegNoDR \
  scripts/train_test_optimal.sh
```

## Task Variants

Register three new tasks:

- `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-FullObsCausalAttn-NoRegNoDR`
- `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-ProprioRefCrossAttn-NoRegNoDR`
- `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttn-NoRegNoDR`

Each task should start from
`unitree_g1_flat_tracking_bfm_test_optimal_env_cfg(disable_reg_and_dr=True)`.

Each task should preserve the TestOptimal choices:

- actor receives full, uncorrupted critic-style observations
- global body pose rewards are used
- regularization rewards are removed
- domain randomization is removed
- `motion.history_steps` is `0`
- `motion.future_steps` is `1`

## Critic

The critic must remain the ordinary MLP critic used by the current BFM PPO
configuration:

```python
RslRlModelCfg(
  hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
  activation="elu",
  obs_normalization=True,
)
```

Critic parameter count is not part of the 10M parameter target. The ablation
should isolate actor architecture differences.

## Actor Parameter Budget

Only the actor is budgeted. Each attention actor should be close to 10M
parameters without exceeding the test threshold.

Target:

- actor parameter count: approximately 10M
- accepted test window: `8_000_000 <= actor_params <= 10_500_000`

Default attention hyperparameters:

```python
d_model = 512
num_heads = 8
ffn_dim = 2048
history_layers = 2
cross_layers = 1
head_hidden_dims = (1536, 1024, 512, 256)
dropout = 0.0
activation = "gelu"
```

If one actor variant naturally falls below the budget because it has fewer
attention blocks, tune only that variant's MLP head or layer count enough to
keep capacity comparable.

## Observation Layout

The attention models are intentionally task-specific and will assume the current
TestOptimal full actor observation layout:

- `command`
- `motion_anchor_pos_b`
- `motion_anchor_ori_b`
- `body_pos`
- `body_ori`
- `base_lin_vel`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `actions`

The expected dimensions for the G1 BFM TestOptimal task are:

- `command`: `58`
- `motion_anchor_pos_b`: `3`
- `motion_anchor_ori_b`: `6`
- `body_pos`: `42`
- `body_ori`: `84`
- `base_lin_vel`: `3`
- `base_ang_vel`: `3`
- `joint_pos`: `29`
- `joint_vel`: `29`
- `actions`: `29`
- full frame: `286`

Models should validate the observation dimension at construction. A mismatch
should raise `ValueError` instead of silently slicing incorrectly.

## Observation History

Attention task env configs should use observation history for actor terms.

Use a default actor history length of `11`, representing ten previous frames
plus the current frame:

```text
obs_{t-10}, ..., obs_t
```

The history comes from `ObservationTermCfg.history_length` with flattened
history, so the actor still receives a single flat TensorDict observation from
RSL-RL. The custom actor reshapes this flat observation internally.

The observation manager flattens history in term-major order:

```text
[command_{t-H+1:t}, motion_anchor_pos_b_{t-H+1:t}, ..., actions_{t-H+1:t}]
```

The custom actor must reconstruct frame-major views internally when a model
needs full frames:

```text
[full_obs_{t-H+1}, ..., full_obs_t]
```

Reference command future is not used. The `command` term for each observation
frame is the command available at that environment step.

## Actor 1: FullObsCausalAttentionActor

Reference: Humanoid-GPT-style causal temporal sequence modeling.

Input:

```text
[full_obs_{t-H+1}, ..., full_obs_t]
```

Architecture:

```text
full obs frame projection
  -> add learned positional encoding
  -> causal Transformer encoder over history frames
  -> final current-frame token
  -> MLP head
  -> Gaussian policy distribution
```

The causal mask ensures each frame can only attend to itself and previous
frames. The actor output uses only the final token.

This variant tests whether treating the whole tracking observation as a
Humanoid-GPT-like temporal sequence improves tracking.

## Actor 2: ProprioRefCrossAttentionActor

Input:

- proprioception with history
- current reference command only

Proprioception terms:

- `base_lin_vel`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `actions`

Reference terms:

- `command`

Architecture:

```text
proprio history frames
  -> frame projection
  -> pooled proprio query

current command
  -> per-joint reference tokens [q_ref_i, qd_ref_i]
  -> command token projection

query attends to command tokens with cross-attention
  -> command embedding
  -> MLP head
  -> Gaussian policy distribution
```

The reference command should be tokenized as 29 per-joint tokens. Each token
contains the current reference joint position and velocity for one DoF. A learned
joint embedding should be added so the model can distinguish joints.

This variant tests whether cross-attention between proprioception and current
reference command is useful without a causal history encoder.

## Actor 3: HistProprioCrossAttentionActor

Reference: RoHM-style causal history encoder plus dynamics-conditioned command
aggregation.

Input:

- proprioception with history
- current reference command only
- current full observation for final MLP fusion

Architecture:

```text
proprio history frames
  -> frame projection
  -> causal self-attention
  -> dynamics embedding h_t

current command
  -> per-joint reference tokens [q_ref_i, qd_ref_i]
  -> command token projection

h_t as query attends to command tokens
  -> command embedding u_t

[current full obs, h_t, u_t]
  -> MLP head
  -> Gaussian policy distribution
```

This is the preferred main ablation because it directly tests the RoHM claim:
recent robot dynamics should condition how reference commands are aggregated.

## Model Integration

Use RSL-RL's existing `class_name` resolution with fully qualified class names:

```python
class_name="mjlab.tasks.tracking.rl.attention_models:HistProprioCrossAttentionActor"
```

The custom actor classes should match the RSL-RL model interface:

- constructor accepts `obs`, `obs_groups`, `obs_set`, `output_dim`, and actor
  config kwargs
- `forward(...)` returns deterministic output unless `stochastic_output=True`
- `distribution_cfg` is supported for Gaussian actions
- `output_mean`, `output_std`, `output_entropy`, log-probability, and KL methods
  are compatible with PPO
- `update_normalization(obs)` updates the observation normalizer if enabled
- `is_recurrent = False`

The implementation should reuse RSL-RL-compatible distribution and normalization
behavior. The model is not recurrent from RSL-RL's perspective because history is
provided as part of the flat observation.

## Config Integration

Because shared `src/mjlab/rl` config should not be modified, add a tracking-only
actor config dataclass in `attention_cfg.py` that subclasses `RslRlModelCfg`.

The dataclass should add these fields:

- `history_length`
- `frame_dim`
- `command_dim`
- `num_dofs`
- `d_model`
- `num_heads`
- `ffn_dim`
- `history_layers`
- `cross_layers`
- `head_hidden_dims`
- `dropout`
- `attention_activation`

`dataclasses.asdict()` in the training entry point will pass these fields to the
custom actor class. Baseline MLP tasks will not see these fields.

## Error Handling

The custom actors should fail early when:

- the actor observation group does not resolve to exactly one flat tensor group
- observation rank is not 2
- flattened dimension is not `history_length * frame_dim`
- `command_dim` is not `2 * num_dofs`
- `d_model` is not divisible by `num_heads`

Error messages should name the expected and actual dimensions.

## Testing

Add focused tests:

- the three attention tasks are registered
- attention tasks keep `motion.history_steps == 0` and `motion.future_steps == 1`
- actor observation history length is applied to all terms
- critic config remains identical to the baseline BFM MLP critic
- each actor can run a forward pass with dummy observations and produce
  `(batch, action_dim)` output
- each stochastic actor exposes PPO distribution methods
- actor parameter count is within `8_000_000 <= params <= 10_500_000`
- invalid observation dimensions raise `ValueError`

Run at least:

```bash
uv run pytest tests/test_tracking_attention_models.py tests/test_tracking_task.py
```

If implementation changes script behavior, also run the relevant script tests.

## Expected Training Commands

Baseline:

```bash
scripts/train_test_optimal.sh
```

Humanoid-GPT-style full-observation causal attention:

```bash
TASK=Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-FullObsCausalAttn-NoRegNoDR \
  RUN_NAME=test_optimal_full_obs_causal_attn_no_reg_no_dr \
  scripts/train_test_optimal.sh
```

Proprioception/reference cross-attention:

```bash
TASK=Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-ProprioRefCrossAttn-NoRegNoDR \
  RUN_NAME=test_optimal_proprio_ref_cross_attn_no_reg_no_dr \
  scripts/train_test_optimal.sh
```

RoHM-style causal proprioception plus cross-attention:

```bash
TASK=Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttn-NoRegNoDR \
  RUN_NAME=test_optimal_hist_proprio_cross_attn_no_reg_no_dr \
  scripts/train_test_optimal.sh
```

## Open Design Decisions Resolved

- Future reference frames are not used.
- Critic remains the existing large MLP.
- Actor parameter budget is approximately 10M.
- Runtime changes stay under `src/mjlab/tasks/tracking`.
- Tests may live under `tests/`.
- The first implementation is task-specific to G1 BFM TestOptimal rather than a
  general attention framework for all tasks.
