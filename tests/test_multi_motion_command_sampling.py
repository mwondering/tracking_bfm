"""Tests for multi-motion adaptive sampling bookkeeping."""

from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand


def _make_command() -> MultiMotionCommand:
  command = object.__new__(MultiMotionCommand)
  command.bin_width_steps = 5
  command.bin_count = 4
  command.motion = SimpleNamespace(num_files=2)
  command.motion_bin_counts = torch.tensor([4, 4], dtype=torch.long)
  command.bin_lengths = torch.full((2, 4), 5.0, dtype=torch.float)
  command.bin_episode_count = torch.zeros((2, 4), dtype=torch.float)
  command.bin_failure_count = torch.zeros((2, 4), dtype=torch.float)
  command.motion_idx = torch.tensor([1, 0], dtype=torch.long)
  command.time_steps = torch.tensor([7, 1], dtype=torch.long)
  command._adaptive_sampling_phase = "idle"
  command._skip_current_adaptive_episode_count = torch.zeros(
    2, dtype=torch.bool
  )
  command.cfg = SimpleNamespace(sampling_mode="adaptive")
  command._env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    termination_manager=SimpleNamespace(
      terminated=torch.tensor([True, False], dtype=torch.bool)
    ),
    episode_length_buf=torch.tensor([13, 0], dtype=torch.long),
  )
  return command


def test_stage_pre_resample_stats_records_old_failure_bin() -> None:
  """Pre-resample bookkeeping should use the old motion/bin state."""
  command = _make_command()

  command._stage_pre_resample_adaptive_stats(torch.tensor([0], dtype=torch.long))

  assert command.bin_episode_count[1, 1].item() == pytest.approx(0.2)
  assert command.bin_failure_count[1, 1].item() == pytest.approx(1.0)
  assert command._skip_current_adaptive_episode_count[0].item() is True


def test_current_step_stats_skip_envs_resampled_before_update() -> None:
  """A pre-resampled env should not be counted again on its new sampled bin."""
  command = _make_command()
  command._stage_pre_resample_adaptive_stats(torch.tensor([0], dtype=torch.long))

  command.motion_idx = torch.tensor([0, 0], dtype=torch.long)
  command.time_steps = torch.tensor([16, 6], dtype=torch.long)

  command._accumulate_current_adaptive_sampling_stats()

  assert command.bin_episode_count[1, 1].item() == pytest.approx(0.2)
  assert command.bin_episode_count[0, 1].item() == pytest.approx(0.2)
  assert command.bin_episode_count[0, 3].item() == pytest.approx(0.0)
  assert not command._skip_current_adaptive_episode_count.any()


def test_stage_pre_resample_stats_skips_initial_reset() -> None:
  """Initial resets should not create fake adaptive sampling counts."""
  command = _make_command()

  command._stage_pre_resample_adaptive_stats(torch.tensor([1], dtype=torch.long))

  assert torch.equal(command.bin_episode_count, torch.zeros_like(command.bin_episode_count))
  assert torch.equal(command.bin_failure_count, torch.zeros_like(command.bin_failure_count))
  assert not command._skip_current_adaptive_episode_count[1].item()


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
