# Adaptive Sampling Motion Failure Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rank-local motion failure-rate aggregation and W&B-only top-10 motion reporting for adaptive sampling, without printing the motion list in the terminal log.

**Architecture:** Extend `MultiMotionCommand` with read-only helpers that map motion indices to stable display names and derive motion-level failure summaries from existing per-bin adaptive sampling tensors. Then extend `MotionTrackingOnPolicyRunner` with a W&B-only logging hook that reads the motion report each training iteration on `rank0` and publishes summary scalars plus a `wandb.Table`.

**Tech Stack:** Python, PyTorch, rsl_rl `OnPolicyRunner`, Weights & Biases, pytest

---

### Task 1: Add command-level regression tests for motion failure aggregation

**Files:**
- Modify: `tests/test_multi_motion_command_sampling.py`
- Modify: `src/mjlab/tasks/tracking/mdp/multi_commands.py`
- Test: `tests/test_multi_motion_command_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `tests/test_multi_motion_command_sampling.py` after the existing adaptive sampling bookkeeping tests:

```python
def test_motion_failure_report_aggregates_bins_and_names() -> None:
  command = _make_command()
  command.motion_files = [
    "/dataset/acro/front_kick.npz",
    "/dataset/locomotion/side_step.npz",
  ]
  command.cfg.motion_path = "/dataset"
  command.bin_episode_count = torch.tensor(
    [[2.0, 2.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]], dtype=torch.float
  )
  command.bin_failure_count = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]], dtype=torch.float
  )
  command.bin_valid_mask = torch.tensor(
    [[True, True, False, False], [True, True, False, False]], dtype=torch.bool
  )

  report = command.get_motion_failure_report(top_k=10)

  assert report["mean_failure_rate"] == pytest.approx(0.5)
  assert report["max_failure_rate"] == pytest.approx(0.75)
  assert report["top10_min_failure_rate"] == pytest.approx(0.25)
  assert report["rows"] == [
    {
      "rank": 1,
      "motion_index": 1,
      "motion_name": "locomotion/side_step",
      "failure_rate": pytest.approx(0.75),
      "total_failures": pytest.approx(3.0),
      "total_visits": pytest.approx(4.0),
    },
    {
      "rank": 2,
      "motion_index": 0,
      "motion_name": "acro/front_kick",
      "failure_rate": pytest.approx(0.25),
      "total_failures": pytest.approx(1.0),
      "total_visits": pytest.approx(4.0),
    },
  ]


def test_motion_failure_report_handles_zero_visits_deterministically() -> None:
  command = _make_command()
  command.motion_files = [
    "/dataset/acro/front_kick.npz",
    "/dataset/locomotion/side_step.npz",
  ]
  command.cfg.motion_path = "/dataset"
  command.bin_episode_count.zero_()
  command.bin_failure_count.zero_()
  command.bin_valid_mask = torch.tensor(
    [[True, True, False, False], [True, True, False, False]], dtype=torch.bool
  )

  report = command.get_motion_failure_report(top_k=10)

  assert report["mean_failure_rate"] == pytest.approx(0.0)
  assert report["max_failure_rate"] == pytest.approx(0.0)
  assert report["top10_min_failure_rate"] == pytest.approx(0.0)
  assert report["rows"][0]["motion_index"] == 0
  assert report["rows"][0]["motion_name"] == "acro/front_kick"
  assert report["rows"][0]["failure_rate"] == pytest.approx(0.0)
  assert report["rows"][0]["total_visits"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_multi_motion_command_sampling.py -q
```

Expected:

- FAIL with `AttributeError` because `MultiMotionCommand` does not yet expose `get_motion_failure_report`

- [ ] **Step 3: Write the minimal implementation for motion names and report building**

In `src/mjlab/tasks/tracking/mdp/multi_commands.py`, make these edits:

1. In `MultiMotionCommand.__init__`, retain the resolved file list:

```python
    motion_files = self._resolve_motion_files()
    self.motion_files = list(motion_files)
```

2. Add a helper that maps the retained file list to display names:

```python
  def _motion_display_names(self) -> list[str]:
    motion_path = os.fspath(self.cfg.motion_path)
    display_names: list[str] = []
    for motion_file in self.motion_files:
      motion_path_obj = os.path.normpath(motion_file)
      if motion_path:
        name = os.path.relpath(motion_path_obj, motion_path)
      else:
        name = os.path.basename(motion_path_obj)
      stem, _ = os.path.splitext(name)
      display_names.append(stem)
    return display_names
```

3. Add a helper that derives motion-level totals and rates from existing tensors:

```python
  def _motion_failure_statistics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_visits = self.bin_episode_count.sum(dim=1)
    total_failures = self.bin_failure_count.sum(dim=1)
    failure_rate = total_failures / torch.clamp(total_visits, min=1.0e-12)
    failure_rate = torch.where(total_visits > 0.0, failure_rate, torch.zeros_like(failure_rate))
    return total_visits, total_failures, failure_rate
```

4. Add the report builder used by the runner:

```python
  def get_motion_failure_report(self, top_k: int = 10) -> dict[str, object]:
    total_visits, total_failures, failure_rate = self._motion_failure_statistics()
    display_names = self._motion_display_names()
    top_k = max(int(top_k), 0)

    ranked_indices = sorted(
      range(len(display_names)),
      key=lambda idx: (
        -float(failure_rate[idx].item()),
        -float(total_failures[idx].item()),
        idx,
      ),
    )
    top_indices = ranked_indices[:top_k]
    rows = []
    for rank, motion_index in enumerate(top_indices, start=1):
      rows.append(
        {
          "rank": rank,
          "motion_index": motion_index,
          "motion_name": display_names[motion_index],
          "failure_rate": float(failure_rate[motion_index].item()),
          "total_failures": float(total_failures[motion_index].item()),
          "total_visits": float(total_visits[motion_index].item()),
        }
      )

    top10_min = min((row["failure_rate"] for row in rows), default=0.0)
    return {
      "mean_failure_rate": float(failure_rate.mean().item()),
      "max_failure_rate": float(failure_rate.max().item()) if len(display_names) > 0 else 0.0,
      "top10_min_failure_rate": float(top10_min),
      "rows": rows,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/test_multi_motion_command_sampling.py -q
```

Expected:

- PASS with all tests green in `tests/test_multi_motion_command_sampling.py`

- [ ] **Step 5: Commit the command-layer change**

Run:

```bash
git add tests/test_multi_motion_command_sampling.py src/mjlab/tasks/tracking/mdp/multi_commands.py
git commit -m "feat: add motion failure report helpers"
```


### Task 2: Add runner-level tests for W&B-only adaptive sampling motion logging

**Files:**
- Create: `tests/test_tracking_runner_motion_failure_logging.py`
- Modify: `src/mjlab/tasks/tracking/rl/runner.py`
- Test: `tests/test_tracking_runner_motion_failure_logging.py`

- [ ] **Step 1: Write the failing runner tests**

Create `tests/test_tracking_runner_motion_failure_logging.py` with this content:

```python
from types import SimpleNamespace

from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class _FakeWriter:
  def __init__(self):
    self.scalars = []

  def add_scalar(self, tag, value, step):
    self.scalars.append((tag, float(value), step))


def _make_runner(command, logger_type="wandb", global_rank=0):
  runner = object.__new__(MotionTrackingOnPolicyRunner)
  runner.writer = _FakeWriter()
  runner.logger_type = logger_type
  runner.disable_logs = False
  runner.gpu_global_rank = global_rank
  runner.env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: command)
    )
  )
  return runner


def test_runner_logs_motion_failure_report_to_wandb(monkeypatch) -> None:
  command = object.__new__(MultiMotionCommand)
  command.get_motion_failure_report = lambda top_k=10: {
    "mean_failure_rate": 0.4,
    "max_failure_rate": 0.8,
    "top10_min_failure_rate": 0.2,
    "rows": [
      {
        "rank": 1,
        "motion_name": "locomotion/side_step",
        "motion_index": 3,
        "failure_rate": 0.8,
        "total_failures": 8.0,
        "total_visits": 10.0,
      }
    ],
  }

  logged = {}

  class _FakeTable:
    def __init__(self, columns, data):
      self.columns = columns
      self.data = data

  monkeypatch.setattr("mjlab.tasks.tracking.rl.runner.wandb.Table", _FakeTable)
  monkeypatch.setattr("mjlab.tasks.tracking.rl.runner.wandb.run", object())
  monkeypatch.setattr(
    "mjlab.tasks.tracking.rl.runner.wandb.log",
    lambda payload, step: logged.update({"payload": payload, "step": step}),
  )

  runner = _make_runner(command, logger_type="wandb", global_rank=0)
  runner._log_adaptive_sampling_motion_failure_report(it=12)

  assert runner.writer.scalars == [
    ("Train/adaptive_sampling/motion_failure_rate_mean", 0.4, 12),
    ("Train/adaptive_sampling/motion_failure_rate_max", 0.8, 12),
    ("Train/adaptive_sampling/motion_failure_rate_top10_min", 0.2, 12),
  ]
  assert logged["step"] == 12
  table = logged["payload"]["Train/adaptive_sampling/top10_motion_failure_rate"]
  assert table.columns == [
    "rank",
    "motion_name",
    "motion_index",
    "failure_rate",
    "total_failures",
    "total_visits",
  ]
  assert table.data == [[1, "locomotion/side_step", 3, 0.8, 8.0, 10.0]]


def test_runner_skips_motion_failure_report_off_rank0(monkeypatch) -> None:
  command = object.__new__(MultiMotionCommand)
  command.get_motion_failure_report = lambda top_k=10: {
    "mean_failure_rate": 0.4,
    "max_failure_rate": 0.8,
    "top10_min_failure_rate": 0.2,
    "rows": [],
  }

  logged = {"called": False}
  monkeypatch.setattr("mjlab.tasks.tracking.rl.runner.wandb.run", object())
  monkeypatch.setattr(
    "mjlab.tasks.tracking.rl.runner.wandb.log",
    lambda payload, step: logged.update({"called": True}),
  )

  runner = _make_runner(command, logger_type="wandb", global_rank=1)
  runner._log_adaptive_sampling_motion_failure_report(it=7)

  assert runner.writer.scalars == []
  assert not logged["called"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_tracking_runner_motion_failure_logging.py -q
```

Expected:

- FAIL with `AttributeError` because `MotionTrackingOnPolicyRunner` does not yet expose `_log_adaptive_sampling_motion_failure_report`

- [ ] **Step 3: Implement the runner-side W&B-only hook**

In `src/mjlab/tasks/tracking/rl/runner.py`, make these edits:

1. Import the multi-motion command type:

```python
from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand
```

2. Add a helper that safely retrieves the active multi-motion term:

```python
  def _get_multi_motion_command(self) -> MultiMotionCommand | None:
    motion_term = self.env.unwrapped.command_manager.get_term("motion")
    if isinstance(motion_term, MultiMotionCommand):
      return motion_term
    return None
```

3. Add the W&B-only logging hook:

```python
  def _log_adaptive_sampling_motion_failure_report(self, it: int) -> None:
    if self.writer is None or self.logger_type != "wandb" or self.disable_logs:
      return
    if getattr(self, "gpu_global_rank", 0) != 0:
      return
    if wandb.run is None:
      return

    motion_term = self._get_multi_motion_command()
    if motion_term is None:
      return

    report = motion_term.get_motion_failure_report(top_k=10)
    self.writer.add_scalar(
      "Train/adaptive_sampling/motion_failure_rate_mean",
      report["mean_failure_rate"],
      it,
    )
    self.writer.add_scalar(
      "Train/adaptive_sampling/motion_failure_rate_max",
      report["max_failure_rate"],
      it,
    )
    self.writer.add_scalar(
      "Train/adaptive_sampling/motion_failure_rate_top10_min",
      report["top10_min_failure_rate"],
      it,
    )

    rows = report["rows"]
    table = wandb.Table(
      columns=[
        "rank",
        "motion_name",
        "motion_index",
        "failure_rate",
        "total_failures",
        "total_visits",
      ],
      data=[
        [
          row["rank"],
          row["motion_name"],
          row["motion_index"],
          row["failure_rate"],
          row["total_failures"],
          row["total_visits"],
        ]
        for row in rows
      ],
    )
    wandb.log(
      {"Train/adaptive_sampling/top10_motion_failure_rate": table},
      step=it,
    )
```

4. Override `log()` to call the base implementation first, then emit the W&B-only report:

```python
  def log(self, locs: dict, width: int = 80, pad: int = 35):
    super().log(locs, width=width, pad=pad)
    self._log_adaptive_sampling_motion_failure_report(locs["it"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/test_tracking_runner_motion_failure_logging.py -q
```

Expected:

- PASS with both runner logging tests green

- [ ] **Step 5: Commit the runner logging change**

Run:

```bash
git add tests/test_tracking_runner_motion_failure_logging.py src/mjlab/tasks/tracking/rl/runner.py
git commit -m "feat: log adaptive sampling motion failures to wandb"
```


### Task 3: Run focused end-to-end verification for the new reporting path

**Files:**
- Modify: `src/mjlab/tasks/tracking/mdp/multi_commands.py`
- Modify: `src/mjlab/tasks/tracking/rl/runner.py`
- Modify: `tests/test_multi_motion_command_sampling.py`
- Create: `tests/test_tracking_runner_motion_failure_logging.py`
- Test: `tests/test_multi_motion_command_sampling.py`
- Test: `tests/test_tracking_runner_motion_failure_logging.py`

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
uv run python -m pytest \
  tests/test_multi_motion_command_sampling.py \
  tests/test_tracking_runner_motion_failure_logging.py \
  -q
```

Expected:

- PASS with all new and existing targeted tests green

- [ ] **Step 2: Run the existing tracking runner regression file**

Run:

```bash
uv run python -m pytest tests/test_runner.py -q
```

Expected:

- PASS, confirming the tracking runner changes did not regress ONNX export or checkpoint behavior

- [ ] **Step 3: Review the final diff for scope**

Run:

```bash
git diff -- \
  src/mjlab/tasks/tracking/mdp/multi_commands.py \
  src/mjlab/tasks/tracking/rl/runner.py \
  tests/test_multi_motion_command_sampling.py \
  tests/test_tracking_runner_motion_failure_logging.py
```

Expected:

- Only motion-report helpers, W&B-only runner logging, and focused tests are included

- [ ] **Step 4: Commit the verified implementation state**

Run:

```bash
git add \
  src/mjlab/tasks/tracking/mdp/multi_commands.py \
  src/mjlab/tasks/tracking/rl/runner.py \
  tests/test_multi_motion_command_sampling.py \
  tests/test_tracking_runner_motion_failure_logging.py
git commit -m "feat: report top adaptive sampling motion failures"
```
