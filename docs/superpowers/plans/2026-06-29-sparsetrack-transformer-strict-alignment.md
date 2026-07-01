# SparseTrack Transformer Strict Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR` match SparseTrack's tracking transformer policy as closely as the current RSL-RL integration allows.

**Architecture:** Keep the existing full-ref `TestOptimal-NoRegNoDR` observation contract, but make the SparseTrack baseline use SparseTrack-style actor and critic transformer shapes: separate prop/action/task token preparation, SparseTrack `TaskEmbedder`, `HumanoidTransformer` with a single linear projection head, zero-initialized heads, and SparseTrack PPO-facing defaults. Add a tracking-local PPO subclass for SparseTrack actor/critic split learning rates while leaving other PPO tasks on the stock RSL-RL algorithm.

**Tech Stack:** PyTorch, RSL-RL `MLPModel`, `TensorDict`, pytest, ruff.

---

## Current Gaps

1. `SparseTrackFullRefAttentionActor` currently uses RSL-RL's MLP head with `head_hidden_dims=(512, 256)`. SparseTrack uses `HumanoidTransformer.projection_head = nn.Linear(embed_dim, output_dim)` directly.
2. Task embedding is currently an inline `nn.Linear(task_obs_dim, d_model)`. SparseTrack uses a separate `TaskEmbedder`, with support for `task_embedder_hidden_dims=[]` and `reduced_task_dim=None`.
3. Initial exploration std is `0.5`. SparseTrack tracking transformer config uses `init_noise_std=0.8` and `std_clamp_range=[0.001, 1.0]`.
4. PPO learning rate is a compromise single LR `5e-5`. SparseTrack uses actor LR `2e-5` and critic LR `1e-3`; strict alignment needs a tracking-local PPO subclass with separate optimizer parameter groups.
5. SparseTrack's tracking transformer uses `num_steps_per_env=32`, `num_learning_epochs=2`, `num_mini_batches=16`, `entropy_coef=0.005`. Current baseline already has epochs/minibatches aligned, but not rollout length or entropy.
6. Current tests allow the SparseTrack actor parameter budget to be `3M-3.5M`, which encodes the extra MLP head. Strict alignment should reduce this to roughly the SparseTrack actor shape: transformer + task embedder + one linear head.
7. The attention block implementation is already materially aligned: RMSNorm, RoPE, SwiGLU, self-attention, cross-attention, empty-token mask, residual output projection scaling, and zero-initialized mean output are all present.
8. Current critic observation history is not expanded for the SparseTrack baseline. The transformer critic needs the same full-ref history layout as the actor.

## File Structure

- Modify `src/mjlab/tasks/tracking/rl/attention_models.py`
  - Add `_SparseTrackTaskEmbedder`.
  - Replace the current SparseTrack actor's task projection with the embedder.
  - Replace the RSL-RL MLP head with a single `nn.Linear(d_model, num_dofs)` for the actor.
  - Add `SparseTrackFullRefAttentionCritic` with the same transformer/tokenization and a single `nn.Linear(d_model, 1)` value head.
- Add `src/mjlab/tasks/tracking/rl/ppo.py`
  - Add `SparseTrackSplitLrPPO` with actor and critic optimizer parameter groups and adaptive KL LR updates that preserve separate rates.
- Modify `src/mjlab/rl/config.py` and `src/mjlab/rl/runner.py`
  - Add optional `actor_learning_rate` / `critic_learning_rate` fields.
  - Strip None values before constructing stock PPO so existing tasks keep working.
- Modify `src/mjlab/tasks/tracking/config/g1/attention_cfg.py`
  - Add explicit config fields for `task_embedder_hidden_dims` and `reduced_task_dim`.
  - Change `init_std` to `0.8`.
- Modify `src/mjlab/tasks/tracking/config/g1/env_cfgs.py` and `src/mjlab/tasks/tracking/config/g1/__init__.py`
  - Allow critic history expansion and enable it only for the SparseTrack full-ref baseline.
- Modify `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`
  - Use the transformer critic, split-LR PPO, and SparseTrack defaults: `num_steps_per_env=32`, `entropy_coef=0.005`, actor LR `2e-5`, critic LR `1e-3`.
- Modify `tests/test_tracking_attention_models.py`
  - Add tests for TaskEmbedder module shape, linear-only output head, transformer critic shape, split-LR optimizer groups, and reduced parameter budget.
- Modify `tests/test_tracking_task.py`
  - Update PPO/std expectations to SparseTrack-aligned values and require transformer critic/split LRs.
- Modify `tests/test_train_test_optimal_script.py`
  - Expect the script's default no-reg/no-DR task to be the SparseTrack full-ref attention baseline.

---

### Task 1: Lock Down SparseTrack-Style Task Embedder Contract

**Files:**
- Modify: `tests/test_tracking_attention_models.py`
- Modify: `src/mjlab/tasks/tracking/rl/attention_models.py`

- [ ] **Step 1: Write failing tests**

Add tests after `test_sparsetrack_full_ref_attention_task_tokens_affect_latent`:

```python
def test_sparsetrack_full_ref_attention_uses_task_embedder_module() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)

  assert hasattr(actor, "task_embedder")
  assert not hasattr(actor, "task_projection")
  task_tokens = actor.task_embedder(torch.zeros(2, ACTOR_HISTORY_LENGTH, actor.task_obs_dim))

  assert task_tokens.shape == (2, ACTOR_HISTORY_LENGTH, actor.d_model)


def test_sparsetrack_task_embedder_linear_init_matches_reference_scale() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  task_projection = actor.task_embedder.task_projection

  assert isinstance(task_projection, torch.nn.Linear)
  expected_std = 1.0 / (actor.task_obs_dim**0.5)
  actual_std = task_projection.weight.std().item()

  assert actual_std == pytest.approx(expected_std, rel=0.2)
  assert torch.allclose(task_projection.bias, torch.zeros_like(task_projection.bias))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest \
  tests/test_tracking_attention_models.py::test_sparsetrack_full_ref_attention_uses_task_embedder_module \
  tests/test_tracking_attention_models.py::test_sparsetrack_task_embedder_linear_init_matches_reference_scale \
  -q
```

Expected: fail because `task_embedder` does not exist and `task_projection` is still directly on the actor.

- [ ] **Step 3: Add `_SparseTrackTaskEmbedder`**

In `src/mjlab/tasks/tracking/rl/attention_models.py`, add this class near `_SwiGLU`:

```python
class _SparseTrackTaskEmbedder(nn.Module):
  def __init__(
    self,
    task_obs_dim: int,
    embedding_dim: int,
    reduced_task_dim: int | None = None,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
  ) -> None:
    super().__init__()
    hidden_dims = tuple(hidden_dims or ())
    if reduced_task_dim is not None:
      self.task_projection = self._build_task_projection(
        task_obs_dim, reduced_task_dim, hidden_dims
      )
      matrix = torch.randn(embedding_dim, reduced_task_dim, dtype=torch.float)
      q_matrix, r_matrix = torch.linalg.qr(matrix, mode="reduced")
      diag = torch.sign(torch.diag(r_matrix))
      diag[diag == 0] = 1.0
      self.register_buffer("W", q_matrix * diag)
      self._forward_method = self._reduced_task_projection
    else:
      self.task_projection = self._build_task_projection(
        task_obs_dim, embedding_dim, hidden_dims
      )
      self._forward_method = self._normal_task_projection

  @staticmethod
  def _build_task_projection(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
  ) -> nn.Module:
    if len(hidden_dims) == 0:
      return nn.Linear(input_dim, output_dim)
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), nn.ELU()]
    for layer_index in range(len(hidden_dims) - 1):
      layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
      layers.append(nn.ELU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)

  def _reduced_task_projection(self, task_obs: torch.Tensor) -> torch.Tensor:
    task_embedding = self.task_projection(task_obs)
    task_embedding = task_embedding / (task_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(task_embedding, self.W.T)

  def _normal_task_projection(self, task_obs: torch.Tensor) -> torch.Tensor:
    return self.task_projection(task_obs)

  def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
    return self._forward_method(task_obs)

  @torch.no_grad()
  def init_weights(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        in_dim = module.weight.shape[1]
        std = 1.0 / math.sqrt(in_dim)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
```

- [ ] **Step 4: Route actor through task embedder**

Extend `_BaseTrackingAttentionActor.__init__` signature:

```python
    task_embedder_hidden_dims: tuple[int, ...] | list[int] | None = None,
    reduced_task_dim: int | None = None,
```

Store:

```python
    self.task_embedder_hidden_dims = tuple(task_embedder_hidden_dims or ())
    self.reduced_task_dim = reduced_task_dim
```

Replace in `SparseTrackFullRefAttentionActor.__init__`:

```python
self.task_projection = nn.Linear(self.task_obs_dim, self.d_model)
```

with:

```python
self.task_embedder = _SparseTrackTaskEmbedder(
  task_obs_dim=self.task_obs_dim,
  embedding_dim=self.d_model,
  reduced_task_dim=self.reduced_task_dim,
  hidden_dims=self.task_embedder_hidden_dims,
)
```

Replace in `_attention_latent_from_flat`:

```python
task_tokens = self.task_projection(task_obs)
```

with:

```python
task_tokens = self.task_embedder(task_obs)
```

Update `_init_sparsetrack_weights` by removing `self.task_projection` from the projection loop and adding:

```python
self.task_embedder.init_weights()
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_tracking_attention_models.py -q
```

Expected: all tests pass except parameter budget if Task 2 has not yet updated it.

---

### Task 2: Replace Extra MLP Head With SparseTrack Linear Projection Head Semantics

**Files:**
- Modify: `tests/test_tracking_attention_models.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/attention_cfg.py`

- [ ] **Step 1: Write failing tests**

Add after `test_sparsetrack_full_ref_attention_zero_initial_action_mean`:

```python
def test_sparsetrack_full_ref_attention_uses_linear_projection_head() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  linear_layers = [
    module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
  ]

  assert len(linear_layers) == 1
  assert linear_layers[0].in_features == actor.d_model
  assert linear_layers[0].out_features == NUM_DOFS
```

Update parameter budget:

```python
"sparsetrack_full_ref": (2_700_000, 3_000_000),
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest \
  tests/test_tracking_attention_models.py::test_sparsetrack_full_ref_attention_uses_linear_projection_head \
  tests/test_tracking_attention_models.py::test_tracking_attention_actor_parameter_budget \
  -q
```

Expected: fail because current head has two hidden layers and parameter count is about `3,240,762`.

- [ ] **Step 3: Add a linear-head override**

`rsl_rl.models.MLP` cannot represent an empty hidden-dim head because it indexes `hidden_dims[0]`. A one-step hidden dim of `-1` would still add an activation, so do not use it.

Instead replace `self.mlp` inside `SparseTrackFullRefAttentionActor.__init__` after `super().__init__` and before `_zero_output_head()`:

```python
self.mlp = nn.Linear(self.d_model, self.num_dofs)
```

Update `_zero_output_head` to support both direct `nn.Linear` and the older sequential `MLP` style:

```python
if isinstance(self.mlp, nn.Linear):
  nn.init.zeros_(self.mlp.weight)
  nn.init.zeros_(self.mlp.bias)
  return
```

Then keep the existing fallback for other `MLP`-style heads:

```python
last_linear = next(
  (module for module in reversed(self.mlp) if isinstance(module, nn.Linear)),
  None,
)
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_tracking_attention_models.py -q
```

Expected: all actor model tests pass, and the SparseTrack parameter budget is lower than before.

---

### Task 3: Align SparseTrack Config Defaults That Current PPO Supports

**Files:**
- Modify: `tests/test_tracking_task.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/attention_cfg.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`

- [ ] **Step 1: Write failing config test**

Update `test_sparsetrack_attention_test_optimal_uses_conservative_ppo_settings`:

```python
assert rl_cfg.actor.distribution_cfg == {
  "class_name": "GaussianDistribution",
  "init_std": 0.8,
  "std_type": "scalar",
  "std_range": (0.001, 1.0),
}
assert rl_cfg.num_steps_per_env == 32
assert rl_cfg.algorithm.learning_rate == 2.0e-5
assert rl_cfg.algorithm.num_learning_epochs == 2
assert rl_cfg.algorithm.num_mini_batches == 16
assert rl_cfg.algorithm.entropy_coef == 0.005
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_tracking_task.py::test_sparsetrack_attention_test_optimal_uses_conservative_ppo_settings -q
```

Expected: fail on `init_std`, `num_steps_per_env`, `learning_rate`, or `entropy_coef`.

- [ ] **Step 3: Update actor config**

In `tracking_attention_actor_cfg("sparsetrack_full_ref")`, change:

```python
init_std=0.8,
task_embedder_hidden_dims=(),
reduced_task_dim=None,
```

Add these fields to `TrackingAttentionModelCfg`:

```python
task_embedder_hidden_dims: tuple[int, ...] = field(default_factory=tuple)
reduced_task_dim: int | None = None
```

Add corresponding parameters to `_tracking_attention_actor_cfg`.

- [ ] **Step 4: Update RL defaults**

In `unitree_g1_trackingbfm_attention_ppo_runner_cfg`, for `sparsetrack_full_ref`:

```python
cfg.num_steps_per_env = 32
cfg.algorithm.learning_rate = 2.0e-5
cfg.algorithm.num_learning_epochs = 2
cfg.algorithm.num_mini_batches = 16
cfg.algorithm.entropy_coef = 0.005
```

Do not add `critic_learning_rate` here; current `RslRlPpoAlgorithmCfg` and installed `rsl_rl.algorithms.PPO` do not support it.

- [ ] **Step 5: Run config tests**

Run:

```bash
uv run pytest \
  tests/test_tracking_task.py::test_sparsetrack_attention_test_optimal_uses_conservative_ppo_settings \
  tests/test_tracking_task.py::test_tracking_attention_test_optimal_keeps_baseline_mlp_critic \
  -q
```

Expected: pass.

---

### Task 4: Add SparseTrack Transformer Critic

**Files:**
- Modify: `src/mjlab/tasks/tracking/rl/attention_models.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/__init__.py`
- Modify: `src/mjlab/tasks/tracking/config/g1/rl_cfg.py`
- Test: `tests/test_tracking_attention_models.py`
- Test: `tests/test_tracking_task.py`

- [ ] **Step 1: Write failing tests**

Add a model forward test:

```python
def test_sparsetrack_full_ref_attention_critic_forward_shape() -> None:
  critic = _make_sparsetrack_critic()
  obs = TensorDict(
    {"critic": torch.randn(3, ACTOR_HISTORY_LENGTH * FRAME_DIM)},
    batch_size=[3],
  )

  values = critic(obs)

  assert values.shape == (3, 1)
```

Add a config test:

```python
def test_sparsetrack_attention_test_optimal_uses_transformer_critic() -> None:
  rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg(
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR"
    ),
  )

  assert rl_cfg.critic.class_name == "mjlab.tasks.tracking.rl.attention_models:SparseTrackFullRefAttentionCritic"
```

- [ ] **Step 2: Add shared SparseTrack full-ref transformer implementation**

Factor `SparseTrackFullRefAttentionActor` internals into `_SparseTrackFullRefAttentionMixin` so actor and critic share prop/action/task tokenization, empty-token mask, transformer blocks, RMSNorm, TaskEmbedder, and SparseTrack weight initialization.

- [ ] **Step 3: Add `SparseTrackFullRefAttentionCritic`**

Subclass the same base/mixin, override `_expected_output_dim()` to return `1`, and initialize a single `nn.Linear(d_model, 1)` value head with zero weights/bias.

- [ ] **Step 4: Enable critic history only for SparseTrack baseline**

Add a `critic_history` flag to `unitree_g1_flat_tracking_bfm_attention_test_optimal_env_cfg()` and set critic term `history_length=ACTOR_HISTORY_LENGTH`, `flatten_history_dim=True` only when registering `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR`.

- [ ] **Step 5: Wire critic config**

In `unitree_g1_trackingbfm_attention_ppo_runner_cfg("sparsetrack_full_ref")`, set `cfg.critic` to `TrackingAttentionModelCfg(class_name="mjlab.tasks.tracking.rl.attention_models:SparseTrackFullRefAttentionCritic", ...)` with the actor's transformer dimensions.

- [ ] **Step 6: Verify**

Run:

```bash
uv run pytest \
  tests/test_tracking_attention_models.py::test_sparsetrack_full_ref_attention_critic_forward_shape \
  tests/test_tracking_task.py::test_sparsetrack_attention_test_optimal_uses_transformer_critic \
  -q
```

Expected: both pass.

---

### Task 5: Add PPO Extension For Actor/Critic LR Strictness

**Files:**
- Modify: `src/mjlab/rl/config.py`
- Modify: `src/mjlab/rl/runner.py`
- Create: `src/mjlab/tasks/tracking/rl/ppo.py`
- Test: add or modify `tests/test_tracking_task.py`
- Test: add or modify `tests/test_tracking_attention_models.py`

- [ ] **Step 1: Write config and optimizer tests**

Add a config test:

```python
def test_sparsetrack_attention_can_configure_actor_critic_learning_rates() -> None:
  rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg(
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR"
    ),
  )

  assert rl_cfg.algorithm.actor_learning_rate == 2.0e-5
  assert rl_cfg.algorithm.critic_learning_rate == 1.0e-3
```

Add an optimizer-param-group test:

```python
def test_sparsetrack_split_lr_ppo_uses_actor_critic_param_groups() -> None:
  actor = _make_actor("sparsetrack_full_ref", SparseTrackFullRefAttentionActor)
  critic = _make_sparsetrack_critic()
  obs = TensorDict(
    {
      "actor": torch.randn(4, ACTOR_HISTORY_LENGTH * FRAME_DIM),
      "critic": torch.randn(4, ACTOR_HISTORY_LENGTH * FRAME_DIM),
    },
    batch_size=[4],
  )
  storage = RolloutStorage("rl", 4, 2, obs, [NUM_DOFS], "cpu")

  alg = SparseTrackSplitLrPPO(
    actor,
    critic,
    storage,
    actor_learning_rate=2.0e-5,
    critic_learning_rate=1.0e-3,
  )

  assert [group["lr"] for group in alg.optimizer.param_groups] == [2.0e-5, 1.0e-3]
  assert alg.learning_rate == 2.0e-5
```

- [ ] **Step 2: Extend config dataclass**

In `src/mjlab/rl/config.py`, add:

```python
actor_learning_rate: float | None = None
critic_learning_rate: float | None = None
```

- [ ] **Step 3: Strip None split-LR fields for stock PPO**

In `src/mjlab/rl/runner.py`, remove `actor_learning_rate` and `critic_learning_rate` from the algorithm dict when their values are `None`, so existing stock PPO tasks do not receive unsupported keyword arguments.

- [ ] **Step 4: Implement optimizer param groups and adaptive LR preservation**

Create a tracking-local PPO subclass that, after base PPO construction, replaces the optimizer when both rates are configured:

```python
self.optimizer = resolve_optimizer(optimizer)(
  [
    {"params": self.actor.parameters(), "lr": actor_learning_rate},
    {"params": self.critic.parameters(), "lr": critic_learning_rate},
  ]
)
self.learning_rate = actor_learning_rate
```

Override `update()` only for the split-LR case so adaptive KL changes actor and critic rates independently, matching SparseTrack's behavior.

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest \
  tests/test_tracking_task.py::test_sparsetrack_attention_test_optimal_uses_actor_critic_learning_rates \
  tests/test_tracking_attention_models.py::test_sparsetrack_split_lr_ppo_uses_actor_critic_param_groups \
  tests/test_runner.py::test_runner_persists_common_step_counter \
  -q
```

Expected: pass, proving config fields are present, split optimizer groups are real, and stock PPO construction still works.

---

### Task 6: Final Verification

**Files:**
- Test only.

- [ ] **Step 1: Run actor tests**

Run:

```bash
uv run pytest tests/test_tracking_attention_models.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run focused task/config tests**

Run:

```bash
uv run pytest \
  tests/test_tracking_task.py::test_tracking_attention_test_optimal_tasks_are_registered \
  tests/test_tracking_task.py::test_tracking_attention_test_optimal_uses_no_future_ref_and_actor_history \
  tests/test_tracking_task.py::test_tracking_attention_test_optimal_keeps_baseline_mlp_critic \
  tests/test_tracking_task.py::test_sparsetrack_attention_test_optimal_uses_conservative_ppo_settings \
  tests/test_train_test_optimal_script.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run lint/format**

Run:

```bash
uv run ruff check \
  src/mjlab/tasks/tracking/rl/attention_models.py \
  src/mjlab/tasks/tracking/config/g1/attention_cfg.py \
  src/mjlab/tasks/tracking/config/g1/rl_cfg.py \
  tests/test_tracking_attention_models.py \
  tests/test_tracking_task.py

uv run ruff format --check \
  src/mjlab/tasks/tracking/rl/attention_models.py \
  src/mjlab/tasks/tracking/config/g1/attention_cfg.py \
  src/mjlab/tasks/tracking/config/g1/rl_cfg.py \
  tests/test_tracking_attention_models.py \
  tests/test_tracking_task.py
```

Expected: both pass.

- [ ] **Step 4: Record known unrelated failure**

Run:

```bash
uv run pytest tests/test_tracking_task.py -q
```

Expected in current workspace: one unrelated pre-existing failure may remain:

```text
test_g1_tracking_penalizes_waist_action_rate:
expected reward.weight == -0.05, actual -0.5
```

Do not fix it as part of this SparseTrack alignment task.

---

## Self-Review

- Spec coverage: covers actor architecture, task embedder, projection head, initialization, std, rollout length, PPO supported defaults, and actor/critic LR mismatch.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: all new config fields are named consistently as `task_embedder_hidden_dims` and `reduced_task_dim`.
- Known limitation: exact actor/critic LR split requires a PPO/runner extension because current installed RSL-RL PPO supports one shared `learning_rate`.
