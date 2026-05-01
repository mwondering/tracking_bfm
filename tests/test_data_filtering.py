"""Tests for deterministic motion filtering helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.scripts.data_filtering import (
  EvaluateConfig,
  _build_filter_report,
  _configure_motion_command,
  _extract_failed_motion_paths,
  _merge_filter_reports,
  _prepare_filtering_env_cfg,
  _prepare_launch_cfg,
  _shard_motion_files,
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
    rank=0,
    world_size=1,
  )

  assert report["total_motion_count"] == 2
  assert report["failed_motion_count"] == 1
  assert report["failed_motion_ratio"] == 0.5
  assert report["rank"] == 0
  assert report["world_size"] == 1
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


def test_shard_motion_files_matches_rank_slicing() -> None:
  """Outer scheduling should shard motions the same way as the command loader."""
  motion_files = [Path(f"/dataset/{idx}.npz") for idx in range(6)]

  shard = _shard_motion_files(motion_files, world_size=3, rank=1)

  assert shard == [motion_files[1], motion_files[4]]


def test_prepare_launch_cfg_disables_viewer_when_gpu_ids_are_provided() -> None:
  """Any explicit gpu_ids launch should force non-viewer mode."""
  cfg = EvaluateConfig(viewer="viser", gpu_ids=[0, 1])

  prepared_cfg = _prepare_launch_cfg(cfg)

  assert prepared_cfg.viewer == "none"


def test_merge_filter_reports_combines_partial_reports(tmp_path: Path) -> None:
  """Multi-GPU runs should merge rank-local reports into one final report."""
  part_a = tmp_path / "part_a.json"
  part_b = tmp_path / "part_b.json"
  final_path = tmp_path / "merged.json"

  part_a.write_text(
    json.dumps(
      {
        "task_id": "Task",
        "motion_root": "/dataset",
        "checkpoint": "/ckpt.pt",
        "failure_threshold": 0.9,
        "rank": 0,
        "world_size": 2,
        "total_motion_count": 2,
        "failed_motion_count": 1,
        "failed_motion_ratio": 0.5,
        "failed_motions": [
          {
            "motion_index": 1,
            "path": "/dataset/b.npz",
            "completion_ratio": 0.7,
            "rank": 0,
          }
        ],
      }
    ),
    encoding="utf-8",
  )
  part_b.write_text(
    json.dumps(
      {
        "task_id": "Task",
        "motion_root": "/dataset",
        "checkpoint": "/ckpt.pt",
        "failure_threshold": 0.9,
        "rank": 1,
        "world_size": 2,
        "total_motion_count": 1,
        "failed_motion_count": 0,
        "failed_motion_ratio": 0.0,
        "failed_motions": [],
      }
    ),
    encoding="utf-8",
  )

  merged = _merge_filter_reports([part_a, part_b], final_path)

  assert merged["total_motion_count"] == 3
  assert merged["failed_motion_count"] == 1
  assert merged["failed_motion_ratio"] == 1 / 3
  assert merged["report_parts"] == 2
  assert merged["failed_motions"][0]["rank"] == 0
  assert final_path.exists()
