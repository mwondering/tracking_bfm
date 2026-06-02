"""Tests for deterministic motion filtering helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from mjlab.scripts.data_filtering import (
  EvaluateConfig,
  GenerateDatasetConfig,
  _build_filter_report,
  _build_generate_dataset_report,
  _capture_rollout_batch,
  _configure_motion_command,
  _extract_failed_motion_paths,
  _merge_filter_reports,
  _merge_generate_dataset_reports,
  _output_motion_path_for,
  _prepare_filtering_env_cfg,
  _prepare_launch_cfg,
  _save_rollout_motion,
  _shard_motion_files,
  motion_sequence_complete,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_bfm_env_cfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)


def test_prepare_filtering_env_cfg_disables_randomization() -> None:
  """Filtering config should keep terminations but remove evaluation noise."""
  env_cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=False)

  filtered_cfg = _prepare_filtering_env_cfg(env_cfg)
  motion_cfg = filtered_cfg.commands["motion"]

  assert filtered_cfg.observations["actor"].enable_corruption is False
  assert filtered_cfg.observations["critic"].enable_corruption is False
  assert "push_robot" not in filtered_cfg.events
  assert "base_com" not in filtered_cfg.events
  assert "base_inertia" not in filtered_cfg.events
  assert "body_inertia" not in filtered_cfg.events
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

  result = motion_sequence_complete(cast(Any, env), command_name="motion")

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
  assert isinstance(motion_cfg, MultiMotionCommandCfg)

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

  assert isinstance(prepared_cfg, EvaluateConfig)
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


def test_output_motion_path_for_preserves_relative_layout(tmp_path: Path) -> None:
  """Generated datasets should keep the source dataset's relative structure."""
  motion_root = tmp_path / "source"
  output_root = tmp_path / "generated"
  source_file = motion_root / "amass" / "clip" / "motion_001.npz"

  output_file = _output_motion_path_for(source_file, motion_root, output_root)

  assert output_file == output_root / "amass" / "clip" / "motion_001.npz"


def test_save_rollout_motion_writes_motion_npz_fields(tmp_path: Path) -> None:
  """Teacher rollout clips should be saved in the existing motion npz format."""
  output_file = tmp_path / "rollout.npz"
  rollout = {
    "fps": np.array([50.0]),
    "joint_pos": np.zeros((3, 29), dtype=np.float32),
    "joint_vel": np.ones((3, 29), dtype=np.float32),
    "body_pos_w": np.zeros((3, 30, 3), dtype=np.float32),
    "body_quat_w": np.zeros((3, 30, 4), dtype=np.float32),
    "body_lin_vel_w": np.zeros((3, 30, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((3, 30, 3), dtype=np.float32),
  }

  _save_rollout_motion(output_file, rollout)

  saved = np.load(output_file)
  assert set(saved.files) == set(rollout.keys())
  for key, expected in rollout.items():
    np.testing.assert_array_equal(saved[key], expected)


def test_capture_rollout_batch_removes_env_origin_from_body_positions() -> None:
  """Saved rollout body positions should be motion-local, not env-grid offset."""
  command = SimpleNamespace(
    robot_joint_pos=torch.zeros((2, 1)),
    robot_joint_vel=torch.zeros((2, 1)),
    robot_body_pos_w=torch.tensor(
      [
        [[10.0, 0.0, 1.0], [11.0, 0.0, 1.5]],
        [[20.0, 3.0, 1.0], [21.0, 3.0, 1.5]],
      ],
      dtype=torch.float32,
    ),
    robot_body_quat_w=torch.zeros((2, 2, 4)),
    robot_body_lin_vel_w=torch.zeros((2, 2, 3)),
    robot_body_ang_vel_w=torch.zeros((2, 2, 3)),
    _env=SimpleNamespace(
      scene=SimpleNamespace(
        env_origins=torch.tensor(
          [[10.0, 0.0, 0.0], [20.0, 3.0, 0.0]], dtype=torch.float32
        )
      )
    ),
  )

  batch = _capture_rollout_batch(command, torch.tensor([0, 1]))

  np.testing.assert_allclose(
    batch["body_pos_w"],
    np.array(
      [
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.5]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.5]],
      ],
      dtype=np.float32,
    ),
  )


def test_build_generate_dataset_report_counts_saved_and_failed() -> None:
  """Generate-dataset reports should summarize saved and failed rollout clips."""
  saved_record = {
    "motion_index": 0,
    "path": "/dataset/a.npz",
    "output_path": "/generated/a.npz",
    "completed_steps": 96,
    "total_steps": 100,
    "completion_ratio": 0.96,
  }
  failed_record = {
    "motion_index": 1,
    "path": "/dataset/b.npz",
    "completed_steps": 70,
    "total_steps": 100,
    "completion_ratio": 0.7,
  }

  report = _build_generate_dataset_report(
    task_id="Task",
    motion_root="/dataset",
    output_motion_root="/generated",
    checkpoint="/ckpt.pt",
    threshold=0.95,
    saved_records=[saved_record],
    failed_records=[failed_record],
    rank=0,
    world_size=1,
  )

  assert report["total_motion_count"] == 2
  assert report["saved_motion_count"] == 1
  assert report["failed_motion_count"] == 1
  assert report["saved_motion_ratio"] == 0.5
  assert report["completion_threshold"] == 0.95
  assert report["saved_motions"] == [saved_record]
  assert report["failed_motions"] == [failed_record]


def test_merge_generate_dataset_reports_combines_partial_reports(
  tmp_path: Path,
) -> None:
  """Multi-GPU generate-dataset runs should merge rank-local reports."""
  part_a = tmp_path / "generated.rank00-of-02.json"
  part_b = tmp_path / "generated.rank01-of-02.json"
  final_path = tmp_path / "generated.json"

  part_a.write_text(
    json.dumps(
      {
        "task_id": "Task",
        "motion_root": "/dataset",
        "output_motion_root": "/generated",
        "checkpoint": "/ckpt.pt",
        "completion_threshold": 0.95,
        "rank": 0,
        "world_size": 2,
        "total_motion_count": 2,
        "saved_motion_count": 1,
        "failed_motion_count": 1,
        "saved_motion_ratio": 0.5,
        "saved_motions": [{"motion_index": 0, "rank": 0}],
        "failed_motions": [{"motion_index": 1, "rank": 0}],
      }
    ),
    encoding="utf-8",
  )
  part_b.write_text(
    json.dumps(
      {
        "task_id": "Task",
        "motion_root": "/dataset",
        "output_motion_root": "/generated",
        "checkpoint": "/ckpt.pt",
        "completion_threshold": 0.95,
        "rank": 1,
        "world_size": 2,
        "total_motion_count": 1,
        "saved_motion_count": 1,
        "failed_motion_count": 0,
        "saved_motion_ratio": 1.0,
        "saved_motions": [{"motion_index": 2, "rank": 1}],
        "failed_motions": [],
      }
    ),
    encoding="utf-8",
  )

  merged = _merge_generate_dataset_reports([part_a, part_b], final_path)

  assert merged["total_motion_count"] == 3
  assert merged["saved_motion_count"] == 2
  assert merged["failed_motion_count"] == 1
  assert merged["saved_motion_ratio"] == 2 / 3
  assert merged["report_parts"] == 2
  assert [item["motion_index"] for item in merged["saved_motions"]] == [0, 2]
  assert [item["motion_index"] for item in merged["failed_motions"]] == [1]
  assert final_path.exists()


def test_prepare_launch_cfg_disables_viewer_only_for_evaluate() -> None:
  """GenerateDatasetConfig has no viewer and should keep launch settings intact."""
  cfg = GenerateDatasetConfig(gpu_ids=[0, 1], output_motion_path="/generated")

  prepared_cfg = _prepare_launch_cfg(cfg)

  assert prepared_cfg == cfg
