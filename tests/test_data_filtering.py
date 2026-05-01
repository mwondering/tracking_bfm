"""Tests for deterministic motion filtering helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.scripts.data_filtering import (
  _build_filter_report,
  _configure_motion_command,
  _extract_failed_motion_paths,
  _prepare_filtering_env_cfg,
  motion_sequence_complete,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_bfm_env_cfg


def test_prepare_filtering_env_cfg_disables_randomization() -> None:
  """Filtering config should keep terminations but remove evaluation noise."""
  env_cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=False)

  filtered_cfg = _prepare_filtering_env_cfg(env_cfg)
  motion_cfg = filtered_cfg.commands["motion"]

  assert filtered_cfg.observations["actor"].enable_corruption is False
  assert filtered_cfg.observations["critic"].enable_corruption is False
  assert "push_robot" not in filtered_cfg.events
  assert "base_com" not in filtered_cfg.events
  assert "encoder_bias" not in filtered_cfg.events
  assert "foot_friction" not in filtered_cfg.events
  assert motion_cfg.pose_range == {}
  assert motion_cfg.velocity_range == {}
  assert motion_cfg.joint_position_range == (0.0, 0.0)
  assert filtered_cfg.episode_length_s == int(1e9)
  assert "time_out" not in filtered_cfg.terminations
  assert filtered_cfg.terminations["motion_complete"].time_out is True


def test_motion_sequence_complete_uses_per_env_motion_lengths() -> None:
  """Motion completion should depend on each env's assigned trajectory length."""
  env = SimpleNamespace(
    episode_length_buf=torch.tensor([5, 3, 7], dtype=torch.long),
    command_manager=SimpleNamespace(
      get_term=lambda name: SimpleNamespace(
        motion_length=torch.tensor([5, 4, 8], dtype=torch.long)
      )
    ),
  )

  result = motion_sequence_complete(env, command_name="motion")

  assert torch.equal(result, torch.tensor([True, False, False]))


def test_build_filter_report_counts_bad_motions() -> None:
  """The JSON report should summarize failed motions and ratios correctly."""
  records = [
    {
      "motion_index": 1,
      "path": "/tmp/b.npz",
      "completed_steps": 72,
      "total_steps": 100,
      "completion_ratio": 0.72,
      "terminated": True,
      "truncated": False,
    },
    {
      "motion_index": 0,
      "path": "/tmp/a.npz",
      "completed_steps": 100,
      "total_steps": 100,
      "completion_ratio": 1.0,
      "terminated": False,
      "truncated": True,
    },
  ]

  report = _build_filter_report(
    task_id="Unitree-G1-Tracking-BFM",
    motion_root="/dataset",
    checkpoint="/ckpt/model.pt",
    threshold=0.9,
    records=records,
  )

  assert report["total_motion_count"] == 2
  assert report["failed_motion_count"] == 1
  assert report["failed_motion_ratio"] == 0.5
  assert report["failed_motions"] == [records[0]]


def test_extract_failed_motion_paths_returns_unique_sorted_paths() -> None:
  """Delete mode should operate on the unique failed file list."""
  report = {
    "failed_motions": [
      {"path": str(Path("/tmp/z.npz"))},
      {"path": str(Path("/tmp/a.npz"))},
      {"path": str(Path("/tmp/z.npz"))},
    ]
  }

  paths = _extract_failed_motion_paths(report)

  assert paths == [Path("/tmp/a.npz"), Path("/tmp/z.npz")]


def test_configure_motion_command_applies_reference_window_overrides() -> None:
  """Filtering CLI should allow overriding motion history/future steps."""
  env_cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=False)
  motion_cfg = env_cfg.commands["motion"]

  _configure_motion_command(
    motion_cfg,
    motion_path="/dataset",
    motion_type="isaaclab",
    history_steps=0,
    future_steps=1,
  )

  assert motion_cfg.motion_path == "/dataset"
  assert motion_cfg.motion_type == "isaaclab"
  assert motion_cfg.history_steps == 0
  assert motion_cfg.future_steps == 1
