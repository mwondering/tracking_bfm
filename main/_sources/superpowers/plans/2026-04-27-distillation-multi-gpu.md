# Distillation Multi-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable true multi-GPU distillation training so all ranks optimize one shared student policy with synchronized gradients, while only rank 0 writes logs and checkpoints.

**Architecture:** Keep the current custom distillation runner and pure action-distillation algorithm. Add a small distributed layer to the runner for rank detection and log gating, then add parameter broadcast and gradient all-reduce to the algorithm so each rank trains from local rollout data but applies identical optimizer steps.

**Tech Stack:** Python, PyTorch distributed, TensorDict, pytest, existing `torchrunx` launch path in `src/mjlab/scripts/train.py`

---

### Task 1: Add distributed algorithm coverage first

**Files:**
- Modify: `tests/test_distillation_algorithm.py`
- Modify: `src/mjlab/tasks/distillation/rl/algorithm.py`

- [ ] **Step 1: Write the failing test**

Add the following tests to `tests/test_distillation_algorithm.py` below the existing update test:

```python
from types import SimpleNamespace
from unittest.mock import patch


def test_action_distillation_algorithm_broadcast_parameters_noops_without_multi_gpu() -> None:
  obs = TensorDict({"student_actor": torch.randn(8, 5)}, batch_size=[8])
  model = build_student_model(
    obs=obs,
    student_obs_group="student_actor",
    action_dim=3,
    hidden_dims=(16, 16),
    activation="elu",
  )
  algorithm = ActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-3,
    max_grad_norm=1.0,
  )

  with patch("torch.distributed.broadcast_object_list") as broadcast:
    algorithm.broadcast_parameters()

  broadcast.assert_not_called()


def test_action_distillation_algorithm_multi_gpu_syncs_parameters_and_gradients() -> None:
  obs = TensorDict({"student_actor": torch.randn(16, 5)}, batch_size=[16])
  teacher_actions = obs["student_actor"][:, :3] * 0.25
  model = build_student_model(
    obs=obs,
    student_obs_group="student_actor",
    action_dim=3,
    hidden_dims=(16, 16),
    activation="elu",
  )
  algorithm = ActionDistillationAlgorithm(
    policy=model,
    learning_rate=1.0e-3,
    max_grad_norm=1.0,
    multi_gpu_cfg={"global_rank": 1, "local_rank": 1, "world_size": 2},
  )

  reduce_calls: list[torch.Tensor] = []

  def _record_all_reduce(tensor: torch.Tensor, op=None):
    reduce_calls.append(tensor.detach().clone())
    return None

  with (
    patch("torch.distributed.broadcast_object_list") as broadcast,
    patch("torch.distributed.all_reduce", side_effect=_record_all_reduce) as all_reduce,
  ):
    algorithm.broadcast_parameters()
    algorithm.update(
      student_obs=obs,
      teacher_actions=teacher_actions,
      num_learning_epochs=1,
      num_mini_batches=2,
    )

  broadcast.assert_called_once()
  assert all_reduce.call_count > 0
  assert reduce_calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_algorithm.py -v`
Expected: FAIL with `TypeError` because `ActionDistillationAlgorithm` does not accept `multi_gpu_cfg`, and/or `AttributeError` because `broadcast_parameters` does not exist.

- [ ] **Step 3: Write minimal implementation**

Update `src/mjlab/tasks/distillation/rl/algorithm.py` so the constructor accepts `multi_gpu_cfg`, and add the distributed helpers and gradient reduction path:

```python
class ActionDistillationAlgorithm:
  def __init__(
    self,
    policy: torch.nn.Module,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    multi_gpu_cfg: dict | None = None,
  ):
    self.policy = policy
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
    self.is_multi_gpu = multi_gpu_cfg is not None
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = int(multi_gpu_cfg["global_rank"])
      self.gpu_world_size = int(multi_gpu_cfg["world_size"])
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

  def broadcast_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    model_params = [self.policy.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)
    self.policy.load_state_dict(model_params[0])

  def reduce_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    for param in self.policy.parameters():
      if param.grad is None:
        continue
      torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
      param.grad.div_(self.gpu_world_size)
```

Then call the reduction inside `update()` after `mse_loss.backward()` and before gradient clipping:

```python
        self.optimizer.zero_grad(set_to_none=True)
        mse_loss.backward()
        self.reduce_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(
          self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation_algorithm.py -v`
Expected: PASS for all distillation algorithm tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_algorithm.py src/mjlab/tasks/distillation/rl/algorithm.py
git commit -m "feat: add distributed distillation algorithm sync"
```

### Task 2: Add runner distributed state and log gating

**Files:**
- Modify: `tests/test_distillation_runner_smoke.py`
- Modify: `src/mjlab/tasks/distillation/rl/runner.py`

- [ ] **Step 1: Write the failing test**

Add the following tests to `tests/test_distillation_runner_smoke.py`:

```python
from unittest.mock import MagicMock, patch


def test_distillation_runner_configures_multi_gpu_state_from_environment(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

  with patch("torch.distributed.init_process_group") as init_pg, patch("torch.cuda.set_device") as set_device:
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:1",
      teacher_adapter=TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25),
    )

  assert runner.is_distributed is True
  assert runner.gpu_world_size == 2
  assert runner.gpu_global_rank == 1
  assert runner.gpu_local_rank == 1
  assert runner.disable_logs is True
  init_pg.assert_called_once()
  set_device.assert_called_once_with(1)


def test_distillation_runner_rejects_mismatched_local_rank_device(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

  with pytest.raises(ValueError, match="does not match expected device"):
    DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:0",
      teacher_adapter=TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_runner_smoke.py::test_distillation_runner_configures_multi_gpu_state_from_environment tests/test_distillation_runner_smoke.py::test_distillation_runner_rejects_mismatched_local_rank_device -v`
Expected: FAIL because `DistillationRunner` does not define distributed fields or perform device/rank validation.

- [ ] **Step 3: Write minimal implementation**

In `src/mjlab/tasks/distillation/rl/runner.py`, add distributed initialization at runner startup before building the student policy:

```python
  def __init__(...):
    self.env = env
    self.cfg = train_cfg
    self.log_dir = log_dir
    self.device = torch.device(device)
    self._configure_multi_gpu()
    ...
    self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
```

Add the helper:

```python
  def _configure_multi_gpu(self) -> None:
    self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    self.is_distributed = self.gpu_world_size > 1
    if not self.is_distributed:
      self.gpu_local_rank = 0
      self.gpu_global_rank = 0
      self.multi_gpu_cfg = None
      return

    self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    self.gpu_global_rank = int(os.getenv("RANK", "0"))
    expected_device = f"cuda:{self.gpu_local_rank}"
    if str(self.device) != expected_device:
      raise ValueError(
        f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
      )

    self.multi_gpu_cfg = {
      "global_rank": self.gpu_global_rank,
      "local_rank": self.gpu_local_rank,
      "world_size": self.gpu_world_size,
    }
    torch.distributed.init_process_group(
      backend="nccl",
      rank=self.gpu_global_rank,
      world_size=self.gpu_world_size,
    )
    torch.cuda.set_device(self.gpu_local_rank)
```

Pass `multi_gpu_cfg` into the algorithm:

```python
    self.alg = ActionDistillationAlgorithm(
      policy=self.student_policy,
      learning_rate=float(self.cfg["learning_rate"]),
      max_grad_norm=1.0,
      multi_gpu_cfg=self.multi_gpu_cfg,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation_runner_smoke.py::test_distillation_runner_configures_multi_gpu_state_from_environment tests/test_distillation_runner_smoke.py::test_distillation_runner_rejects_mismatched_local_rank_device -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_runner_smoke.py src/mjlab/tasks/distillation/rl/runner.py
git commit -m "feat: configure distillation runner for distributed launch"
```

### Task 3: Add broadcast, metric reduction, and rank-0-only output behavior

**Files:**
- Modify: `tests/test_distillation_runner_smoke.py`
- Modify: `src/mjlab/tasks/distillation/rl/runner.py`

- [ ] **Step 1: Write the failing test**

Add the following tests to `tests/test_distillation_runner_smoke.py`:

```python
def test_distillation_runner_distributed_learn_broadcasts_and_skips_nonzero_rank_outputs(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(
    logger="tensorboard",
    upload_model=False,
    save_interval=1,
    num_steps_per_env=2,
    max_iterations=1,
    num_learning_epochs=1,
    num_mini_batches=1,
  )
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setenv("LOCAL_RANK", "1")

  with TemporaryDirectory() as tmpdir:
    with (
      patch("torch.distributed.init_process_group"),
      patch("torch.cuda.set_device"),
      patch.object(DistillationRunner, "_prepare_logging_writer") as prepare_writer,
      patch.object(DistillationRunner, "save") as save,
      patch.object(ActionDistillationAlgorithm, "broadcast_parameters") as broadcast,
    ):
      runner = DistillationRunner(
        env,
        asdict(cfg),
        log_dir=tmpdir,
        device="cuda:1",
        teacher_adapter=teacher_adapter,
      )
      runner.learn(num_learning_iterations=1)

  broadcast.assert_called_once()
  prepare_writer.assert_not_called()
  save.assert_not_called()


def test_distillation_runner_reduces_logged_scalars_across_ranks(monkeypatch) -> None:
  env = _DummyVecEnv()
  cfg = DistillationRunnerCfg(logger="tensorboard", upload_model=False)
  teacher_adapter = TeacherPolicyAdapter(lambda obs: obs["actor"][..., :3] * 0.25)

  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "0")
  monkeypatch.setenv("LOCAL_RANK", "0")

  with patch("torch.distributed.init_process_group"), patch("torch.cuda.set_device"):
    runner = DistillationRunner(
      env,
      asdict(cfg),
      log_dir=None,
      device="cuda:0",
      teacher_adapter=teacher_adapter,
    )

  reduced = []

  def _fake_all_reduce(tensor: torch.Tensor, op=None):
    reduced.append(float(tensor.item()))
    tensor.mul_(2.0)

  with patch("torch.distributed.all_reduce", side_effect=_fake_all_reduce):
    value = runner._distributed_mean(3.0)

  assert value == pytest.approx(3.0)
  assert reduced == [3.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation_runner_smoke.py::test_distillation_runner_distributed_learn_broadcasts_and_skips_nonzero_rank_outputs tests/test_distillation_runner_smoke.py::test_distillation_runner_reduces_logged_scalars_across_ranks -v`
Expected: FAIL because `learn()` always prepares logging, saves on every rank, and there is no scalar reduction helper.

- [ ] **Step 3: Write minimal implementation**

In `src/mjlab/tasks/distillation/rl/runner.py`, gate logging and saving on rank 0 and add distributed scalar reduction:

```python
  def _prepare_logging_writer(self) -> None:
    if self.disable_logs:
      return
    if self.log_dir is None or self.writer is not None:
      return
```

Add the helper:

```python
  def _distributed_mean(self, value: float) -> float:
    if not self.is_distributed:
      return value
    tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= self.gpu_world_size
    return float(tensor.item())
```

Update `learn()`:

```python
  def learn(...):
    self._prepare_logging_writer()
    teacher_adapter = self._get_teacher_adapter()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    ...
      self.last_loss_dict = {
        key: self._distributed_mean(value)
        for key, value in self.alg.update(...).items()
      }
      self.last_train_metrics = {
        "beta_teacher": self._distributed_mean(float(beta)),
        "teacher_action_ratio": self._distributed_mean(float(teacher_mask.float().mean().item())),
        "student_action_ratio": self._distributed_mean(float((~teacher_mask).float().mean().item())),
      }
      ...
      if self.log_dir is not None and not self.disable_logs:
        self._log_train_iteration(...)
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir is not None and not self.disable_logs:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
```

Update throughput sizing in `_log_train_iteration()`:

```python
    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation_runner_smoke.py::test_distillation_runner_distributed_learn_broadcasts_and_skips_nonzero_rank_outputs tests/test_distillation_runner_smoke.py::test_distillation_runner_reduces_logged_scalars_across_ranks -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_distillation_runner_smoke.py src/mjlab/tasks/distillation/rl/runner.py
git commit -m "feat: reduce distillation metrics and gate rank outputs"
```

### Task 4: Run regression coverage and clean up

**Files:**
- Modify: `tests/test_distillation_algorithm.py` if assertions need tightening
- Modify: `tests/test_distillation_runner_smoke.py` if smoke expectations need updates
- Modify: `src/mjlab/tasks/distillation/rl/runner.py` only if test output reveals minor correctness gaps

- [ ] **Step 1: Run focused regression tests**

Run: `pytest tests/test_distillation_algorithm.py tests/test_distillation_runner_smoke.py -v`
Expected: PASS for all distillation algorithm and runner smoke coverage.

- [ ] **Step 2: Run broader distillation regression**

Run: `pytest tests/test_distillation_task.py tests/test_play_distillation_script.py tests/test_distillation_student_play_viz.py -v`
Expected: PASS. If one test fails because it relies on exact logging side effects, adjust only the expectation, not the new distributed behavior.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff -- src/mjlab/tasks/distillation/rl/algorithm.py src/mjlab/tasks/distillation/rl/runner.py tests/test_distillation_algorithm.py tests/test_distillation_runner_smoke.py`
Expected: Diff is limited to distributed sync, metric reduction, and matching test updates.

- [ ] **Step 4: Commit final polish if needed**

```bash
git add src/mjlab/tasks/distillation/rl/algorithm.py src/mjlab/tasks/distillation/rl/runner.py tests/test_distillation_algorithm.py tests/test_distillation_runner_smoke.py
git commit -m "test: cover multi-gpu distillation runner behavior"
```

- [ ] **Step 5: Final verification note**

If GPU hardware is available, run one manual smoke launch after tests:

```bash
python -m mjlab.scripts.train Mjlab-Distillation-Flat-Unitree-G1 --gpu-ids 0 1 --agent.max-iterations 1 --agent.num-steps-per-env 2
```

Expected: one shared run directory, only rank 0 checkpoints, no device/rank mismatch error, and no duplicate writer output.
