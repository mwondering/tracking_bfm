"""Tests for multi-motion adaptive sampling bookkeeping."""

from types import SimpleNamespace

import numpy as np
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


class _GhostVisualizer:
  env_idx = 0
  show_all_envs = False

  def __init__(self):
    self.ghosts = []

  def get_env_indices(self, num_envs: int):
    return [0] if num_envs else []

  def add_ghost_mesh(self, qpos, model, label=None):
    self.ghosts.append((qpos.copy(), model, label))


class _Scene(dict):
  def __init__(self, robot):
    super().__init__({"robot": robot})
    self.env_origins = torch.tensor([[10.0, 0.0, 0.0]])


def test_debug_vis_draws_extra_reference_motion_ghost() -> None:
  command = object.__new__(MultiMotionCommand)
  command.cfg = SimpleNamespace(
    viz=SimpleNamespace(mode="ghost"),
    entity_name="robot",
  )
  command.time_steps = torch.tensor([5], dtype=torch.long)
  command._ghost_color = np.array((0.5, 0.7, 0.5, 0.5), dtype=np.float32)
  command._ghost_model = None
  command._extra_reference_ghost_color = np.array(
    (1.0, 0.45, 0.1, 0.45), dtype=np.float32
  )
  command._extra_reference_ghost_model = None
  command.motion_idx = torch.tensor([0], dtype=torch.long)
  command.motion = SimpleNamespace(
    file_lengths=torch.tensor([8], dtype=torch.long),
    length_starts=torch.tensor([0], dtype=torch.long),
    body_pos_w=torch.tensor(
      [
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[1.0, 2.0, 3.0]],
      ],
      dtype=torch.float32,
    ),
    body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 6, dtype=torch.float32),
    joint_pos=torch.tensor(
      [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.1, 0.2]],
      dtype=torch.float32,
    ),
  )
  command.extra_reference_motion = SimpleNamespace(
    time_step_total=3,
    body_pos_w=torch.tensor([[[0.0, 0.0, 0.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]),
    body_quat_w=torch.tensor(
      [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
    ),
    joint_pos=torch.tensor([[0.0, 0.0], [0.3, 0.4], [0.5, 0.6]]),
  )

  model = SimpleNamespace(
    nq=9,
    geom_rgba=np.zeros((1, 4), dtype=np.float32),
  )
  robot = SimpleNamespace(
    indexing=SimpleNamespace(
      free_joint_q_adr=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
      joint_q_adr=torch.tensor([7, 8]),
    )
  )
  command._env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    sim=SimpleNamespace(mj_model=model),
    scene=_Scene(robot),
  )

  visualizer = _GhostVisualizer()
  command._debug_vis_impl(visualizer)

  assert [ghost[2] for ghost in visualizer.ghosts] == [
    "ghost_0",
    "extra_reference_ghost_0",
  ]
  np.testing.assert_allclose(visualizer.ghosts[1][0][0:3], np.array([17.0, 8.0, 9.0]))
  np.testing.assert_allclose(visualizer.ghosts[1][0][7:9], np.array([0.5, 0.6]))
  np.testing.assert_allclose(
    visualizer.ghosts[1][1].geom_rgba[0], np.array([1.0, 0.45, 0.1, 0.45])
  )
