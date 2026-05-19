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
  command.bin_valid_mask = torch.ones((2, 4), dtype=torch.bool)
  command.bin_lengths = torch.full((2, 4), 5.0, dtype=torch.float)
  command.bin_episode_count = torch.zeros((2, 4), dtype=torch.float)
  command.bin_failure_count = torch.zeros((2, 4), dtype=torch.float)
  command.motion_idx = torch.tensor([1, 0], dtype=torch.long)
  command.time_steps = torch.tensor([7, 1], dtype=torch.long)
  command._adaptive_sampling_phase = "idle"
  command._skip_current_adaptive_episode_count = torch.zeros(2, dtype=torch.bool)
  command.cfg = SimpleNamespace(
    sampling_mode="adaptive",
    adaptive_failure_rate_window_iterations=None,
    adaptive_failure_rate_window_chunks=40,
  )
  command._env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    termination_manager=SimpleNamespace(
      terminated=torch.tensor([True, False], dtype=torch.bool)
    ),
    episode_length_buf=torch.tensor([13, 0], dtype=torch.long),
  )
  return command


def _enable_window(
  command: MultiMotionCommand, *, iterations: int, chunks: int
) -> None:
  command.cfg.adaptive_failure_rate_window_iterations = iterations
  command.cfg.adaptive_failure_rate_window_chunks = chunks
  command._init_adaptive_sampling_window()


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

  assert torch.equal(
    command.bin_episode_count, torch.zeros_like(command.bin_episode_count)
  )
  assert torch.equal(
    command.bin_failure_count, torch.zeros_like(command.bin_failure_count)
  )
  assert not command._skip_current_adaptive_episode_count[1].item()


def test_adaptive_window_tracks_increments_in_current_iteration_chunk() -> None:
  """Window mode should store stats in both window totals and current chunk."""
  command = _make_command()
  _enable_window(command, iterations=4, chunks=2)
  command.begin_adaptive_sampling_iteration(10)

  command._stage_pre_resample_adaptive_stats(torch.tensor([0], dtype=torch.long))

  assert command.bin_episode_count[1, 1].item() == pytest.approx(0.2)
  assert command.bin_failure_count[1, 1].item() == pytest.approx(1.0)
  assert command._adaptive_window_episode_chunks[0, 1, 1].item() == pytest.approx(0.2)
  assert command._adaptive_window_failure_chunks[0, 1, 1].item() == pytest.approx(1.0)


def test_adaptive_window_expires_old_chunk_by_training_iteration() -> None:
  """Chunks should expire according to PPO iteration, not command update ticks."""
  command = _make_command()
  command.bin_episode_count[:] = 1.0
  command.bin_failure_count[:] = 1.0
  _enable_window(command, iterations=4, chunks=4)

  command.begin_adaptive_sampling_iteration(0)
  command.begin_adaptive_sampling_iteration(1)
  command.begin_adaptive_sampling_iteration(2)
  command.begin_adaptive_sampling_iteration(3)

  assert torch.equal(
    command.bin_episode_count, torch.ones_like(command.bin_episode_count)
  )
  assert torch.equal(
    command.bin_failure_count, torch.ones_like(command.bin_failure_count)
  )

  command.begin_adaptive_sampling_iteration(4)

  assert torch.equal(
    command.bin_episode_count, torch.zeros_like(command.bin_episode_count)
  )
  assert torch.equal(
    command.bin_failure_count, torch.zeros_like(command.bin_failure_count)
  )


def test_adaptive_window_caps_chunk_count_to_window_iterations() -> None:
  """A small iteration window should not be stretched by an oversized chunk count."""
  command = _make_command()

  _enable_window(command, iterations=3, chunks=10)

  assert command._adaptive_window_episode_chunks.shape[0] == 3
  assert command._adaptive_window_chunk_size == 1
