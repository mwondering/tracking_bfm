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
