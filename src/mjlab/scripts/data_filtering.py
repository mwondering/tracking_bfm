"""Deterministic batch motion filtering for multi-motion tracking datasets."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import tyro
from tensordict import TensorDict
from tqdm import tqdm

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class EvaluateConfig:
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  motion_path: str | None = None
  motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"
  history_steps: int | None = None
  future_steps: int | None = None
  num_envs: int = 1024
  device: str | None = None
  failure_threshold: float = 0.9
  output_file: str = "filtered_motions.json"
  viewer: Literal["none", "auto", "native", "viser"] = "none"
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = None


@dataclass(frozen=True)
class DeleteConfig:
  report_file: str
  missing_ok: bool = True


@dataclass(frozen=True)
class GenerateDatasetConfig:
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  motion_path: str | None = None
  motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"
  history_steps: int | None = None
  future_steps: int | None = None
  num_envs: int = 1024
  device: str | None = None
  completion_threshold: float = 0.95
  output_motion_path: str = "generated_motions"
  output_file: str = "generated_motions_report.json"
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = None


_MOTION_NPZ_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def motion_sequence_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Terminate an evaluation episode exactly at the end of its assigned motion."""
  command = cast(Any, env.command_manager.get_term(command_name))
  return env.episode_length_buf >= command.motion_length


def _prepare_filtering_env_cfg(env_cfg):
  """Disable stochastic evaluation effects while preserving failure terminations."""
  env_cfg = deepcopy(env_cfg)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg)):
    raise ValueError("The selected task is not a tracking task with a motion command.")
  if not isinstance(motion_cmd, MultiMotionCommandCfg):
    raise ValueError("data_filtering.py requires a multi-motion tracking task.")

  motion_cmd.sampling_mode = "start"
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  motion_cmd.if_log_metrics = False

  if "actor" in env_cfg.observations:
    env_cfg.observations["actor"].enable_corruption = False
  if "critic" in env_cfg.observations:
    env_cfg.observations["critic"].enable_corruption = False

  for event_name in (
    "push_robot",
    "base_com",
    "base_inertia",
    "body_inertia",
    "encoder_bias",
    "foot_friction",
  ):
    env_cfg.events.pop(event_name, None)

  env_cfg.episode_length_s = int(1e9)
  env_cfg.terminations.pop("time_out", None)
  env_cfg.terminations["motion_complete"] = TerminationTermCfg(
    func=motion_sequence_complete,
    time_out=True,
    params={"command_name": "motion"},
  )
  return env_cfg


def _configure_motion_command(
  motion_cmd: MultiMotionCommandCfg,
  *,
  motion_path: str,
  motion_type: Literal["isaaclab", "mujoco"],
  history_steps: int | None,
  future_steps: int | None,
) -> None:
  """Apply CLI motion overrides to the filtering motion command."""
  motion_cmd.motion_path = motion_path
  motion_cmd.motion_file = ""
  motion_cmd.motion_type = motion_type
  if history_steps is not None:
    motion_cmd.history_steps = history_steps
  if future_steps is not None:
    motion_cmd.future_steps = future_steps


def _collect_motion_files(motion_root: str) -> list[Path]:
  motion_root_path = Path(motion_root)
  if not motion_root_path.exists():
    raise FileNotFoundError(f"Motion path not found: {motion_root}")
  if not motion_root_path.is_dir():
    raise ValueError(f"motion_path must be a directory: {motion_root}")

  motion_files = sorted(
    path
    for path in motion_root_path.rglob("*")
    if path.is_file() and path.suffix.lower() == ".npz"
  )
  if not motion_files:
    raise ValueError(f"No .npz motion files found under: {motion_root}")
  return motion_files


def _resolve_motion_root(cfg: EvaluateConfig | GenerateDatasetConfig) -> str:
  if cfg.motion_path is not None:
    return cfg.motion_path
  if cfg.wandb_run_path is None:
    raise ValueError(
      "Provide --motion-path explicitly, or provide --wandb-run-path so the motion "
      "artifact can be resolved."
    )

  import wandb

  api = wandb.Api()
  run = api.run(str(cfg.wandb_run_path))
  artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if artifact is None:
    raise RuntimeError("No motion artifact found in the W&B run.")
  return str(Path(artifact.download()))


def _resolve_checkpoint_path(
  task_id: str, cfg: EvaluateConfig | GenerateDatasetConfig
) -> tuple[Path, str]:
  agent_cfg = load_rl_cfg(task_id)
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()

  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path, str(checkpoint_path)

  if cfg.wandb_run_path is None:
    raise ValueError(
      "Provide --checkpoint-file or --wandb-run-path to resolve the evaluation checkpoint."
    )

  checkpoint_path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  return checkpoint_path, str(checkpoint_path)


def _update_relative_body_poses(command: Any) -> None:
  """Mirror the command's relative-body update without advancing time steps."""
  anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )
  anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )
  robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )
  robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )

  from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

  delta_pos_w = robot_anchor_pos_w_repeat.clone()
  delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
  delta_ori_w = yaw_quat(
    quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
  )

  command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
  command.body_pos_relative_w = delta_pos_w + quat_apply(
    delta_ori_w, command.body_pos_w - anchor_pos_w_repeat
  )


def _recompute_observations(env: ManagerBasedRlEnv, command: Any) -> None:
  env.scene.write_data_to_sim()
  env.sim.forward()
  _update_relative_body_poses(command)
  env.sim.sense()
  env.observation_manager._obs_buffer = None
  env.obs_buf = env.observation_manager.compute(update_history=False)


def _assign_motion_indices(
  env: ManagerBasedRlEnv,
  command: Any,
  env_ids: torch.Tensor,
  motion_indices: torch.Tensor,
) -> None:
  if env_ids.numel() == 0:
    return

  command.motion_idx[env_ids] = motion_indices
  command.motion_length[env_ids] = command.motion.file_lengths[motion_indices]
  command.time_steps[env_ids] = 0

  root_pos = command.body_pos_w[env_ids, 0].clone()
  root_ori = command.body_quat_w[env_ids, 0].clone()
  root_lin_vel = command.body_lin_vel_w[env_ids, 0].clone()
  root_ang_vel = command.body_ang_vel_w[env_ids, 0].clone()
  joint_pos = command.joint_pos[env_ids].clone()
  joint_vel = command.joint_vel[env_ids].clone()

  soft_joint_pos_limits = command.robot.data.soft_joint_pos_limits[env_ids]
  joint_pos = torch.clip(
    joint_pos,
    soft_joint_pos_limits[:, :, 0],
    soft_joint_pos_limits[:, :, 1],
  )

  command.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  command.robot.write_root_state_to_sim(
    torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
    env_ids=env_ids,
  )
  command.robot.clear_state(env_ids=env_ids)
  _recompute_observations(env, command)


def _build_filter_report(
  *,
  task_id: str,
  motion_root: str,
  checkpoint: str,
  threshold: float,
  records: list[dict[str, Any]],
  rank: int,
  world_size: int,
) -> dict[str, Any]:
  sorted_records = sorted(records, key=lambda item: item["motion_index"])
  failed_records = [
    record for record in sorted_records if record["completion_ratio"] < threshold
  ]
  total_motion_count = len(sorted_records)
  failed_motion_count = len(failed_records)
  failed_motion_ratio = (
    failed_motion_count / total_motion_count if total_motion_count > 0 else 0.0
  )

  return {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": task_id,
    "motion_root": motion_root,
    "checkpoint": checkpoint,
    "failure_threshold": threshold,
    "rank": rank,
    "world_size": world_size,
    "total_motion_count": total_motion_count,
    "failed_motion_count": failed_motion_count,
    "failed_motion_ratio": failed_motion_ratio,
    "failed_motions": failed_records,
  }


def _output_motion_path_for(
  source_motion_path: Path,
  motion_root: Path | str,
  output_motion_root: Path | str,
) -> Path:
  """Map a source motion path into the generated dataset while preserving layout."""
  source_path = Path(source_motion_path).resolve()
  root_path = Path(motion_root).resolve()
  output_root_path = Path(output_motion_root)
  relative_path = source_path.relative_to(root_path)
  return output_root_path / relative_path


def _save_rollout_motion(output_path: Path, rollout: dict[str, np.ndarray]) -> None:
  """Save one teacher rollout clip in the standard motion ``.npz`` format."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **cast(dict[str, Any], rollout))


def _build_generate_dataset_report(
  *,
  task_id: str,
  motion_root: str,
  output_motion_root: str,
  checkpoint: str,
  threshold: float,
  saved_records: list[dict[str, Any]],
  failed_records: list[dict[str, Any]],
  rank: int,
  world_size: int,
) -> dict[str, Any]:
  sorted_saved = sorted(saved_records, key=lambda item: item["motion_index"])
  sorted_failed = sorted(failed_records, key=lambda item: item["motion_index"])
  total_motion_count = len(sorted_saved) + len(sorted_failed)
  saved_motion_count = len(sorted_saved)
  failed_motion_count = len(sorted_failed)
  saved_motion_ratio = (
    saved_motion_count / total_motion_count if total_motion_count > 0 else 0.0
  )

  return {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": task_id,
    "motion_root": motion_root,
    "output_motion_root": output_motion_root,
    "checkpoint": checkpoint,
    "completion_threshold": threshold,
    "rank": rank,
    "world_size": world_size,
    "total_motion_count": total_motion_count,
    "saved_motion_count": saved_motion_count,
    "failed_motion_count": failed_motion_count,
    "saved_motion_ratio": saved_motion_ratio,
    "saved_motions": sorted_saved,
    "failed_motions": sorted_failed,
  }


def _extract_failed_motion_paths(report: dict[str, Any]) -> list[Path]:
  failed_paths = {
    Path(entry["path"]).resolve()
    for entry in report.get("failed_motions", [])
    if isinstance(entry, dict) and "path" in entry
  }
  return sorted(failed_paths)


def _load_policy(
  task_id: str, env: RslRlVecEnvWrapper, device: str, checkpoint_path: Path
):
  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def _shard_motion_files(
  motion_files: list[Path], world_size: int, rank: int
) -> list[Path]:
  if world_size <= 1:
    return motion_files
  return motion_files[rank::world_size]


def _runtime_rank_context(
  cfg: EvaluateConfig | GenerateDatasetConfig,
) -> tuple[str, int, int]:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))

  if cuda_visible == "":
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  device = f"cuda:{local_rank}"
  return device, rank, world_size


def _rank_output_path(output_file: str, rank: int, world_size: int) -> Path:
  output_path = Path(output_file)
  if world_size <= 1:
    return output_path
  return output_path.with_name(
    f"{output_path.stem}.rank{rank:02d}-of-{world_size:02d}{output_path.suffix or '.json'}"
  )


def _prepare_launch_cfg(
  cfg: EvaluateConfig | GenerateDatasetConfig,
) -> EvaluateConfig | GenerateDatasetConfig:
  if (
    isinstance(cfg, EvaluateConfig) and cfg.gpu_ids is not None and cfg.viewer != "none"
  ):
    print(
      "[INFO] gpu_ids provided; forcing viewer=none to avoid multi-process viewer conflicts."
    )
    return replace(cfg, viewer="none")
  return cfg


def _merge_filter_reports(
  report_paths: list[Path], output_path: Path
) -> dict[str, Any]:
  reports = []
  for report_path in sorted(report_paths):
    with report_path.open("r", encoding="utf-8") as file:
      reports.append(json.load(file))
  if not reports:
    raise ValueError("No partial reports found to merge.")

  merged_failed_motions: list[dict[str, Any]] = []
  total_motion_count = 0
  failed_motion_count = 0

  for report in reports:
    total_motion_count += int(report["total_motion_count"])
    failed_motion_count += int(report["failed_motion_count"])
    merged_failed_motions.extend(report.get("failed_motions", []))

  merged_failed_motions.sort(
    key=lambda item: (item.get("rank", -1), item.get("motion_index", -1))
  )
  merged_report = {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": reports[0]["task_id"],
    "motion_root": reports[0]["motion_root"],
    "checkpoint": reports[0]["checkpoint"],
    "failure_threshold": reports[0]["failure_threshold"],
    "world_size": max(int(report.get("world_size", 1)) for report in reports),
    "report_parts": len(reports),
    "total_motion_count": total_motion_count,
    "failed_motion_count": failed_motion_count,
    "failed_motion_ratio": (
      failed_motion_count / total_motion_count if total_motion_count > 0 else 0.0
    ),
    "failed_motions": merged_failed_motions,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(merged_report, file, indent=2)
  return merged_report


def _merge_generate_dataset_reports(
  report_paths: list[Path], output_path: Path
) -> dict[str, Any]:
  reports = []
  for report_path in sorted(report_paths):
    with report_path.open("r", encoding="utf-8") as file:
      reports.append(json.load(file))
  if not reports:
    raise ValueError("No partial reports found to merge.")

  merged_saved_motions: list[dict[str, Any]] = []
  merged_failed_motions: list[dict[str, Any]] = []
  total_motion_count = 0
  saved_motion_count = 0
  failed_motion_count = 0

  for report in reports:
    total_motion_count += int(report["total_motion_count"])
    saved_motion_count += int(report["saved_motion_count"])
    failed_motion_count += int(report["failed_motion_count"])
    merged_saved_motions.extend(report.get("saved_motions", []))
    merged_failed_motions.extend(report.get("failed_motions", []))

  merged_saved_motions.sort(
    key=lambda item: (item.get("rank", -1), item.get("motion_index", -1))
  )
  merged_failed_motions.sort(
    key=lambda item: (item.get("rank", -1), item.get("motion_index", -1))
  )
  merged_report = {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": reports[0]["task_id"],
    "motion_root": reports[0]["motion_root"],
    "output_motion_root": reports[0]["output_motion_root"],
    "checkpoint": reports[0]["checkpoint"],
    "completion_threshold": reports[0]["completion_threshold"],
    "world_size": max(int(report.get("world_size", 1)) for report in reports),
    "report_parts": len(reports),
    "total_motion_count": total_motion_count,
    "saved_motion_count": saved_motion_count,
    "failed_motion_count": failed_motion_count,
    "saved_motion_ratio": (
      saved_motion_count / total_motion_count if total_motion_count > 0 else 0.0
    ),
    "saved_motions": merged_saved_motions,
    "failed_motions": merged_failed_motions,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(merged_report, file, indent=2)
  return merged_report


class FilteringEvalEnv:
  """VecEnv wrapper that records completed motions and refills idle env slots."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    motion_files: list[Path],
    command: Any,
    rank: int,
  ) -> None:
    self._env = env
    self._motion_files = motion_files
    self._command = command
    self._rank = rank
    self.records: list[dict[str, Any]] = []
    self.finished = False
    self._next_motion_index = 0
    self._active_mask = torch.zeros(
      self.unwrapped.num_envs, dtype=torch.bool, device=self.unwrapped.device
    )
    self._assigned_motion_ids = torch.full(
      (self.unwrapped.num_envs,),
      -1,
      dtype=torch.long,
      device=self.unwrapped.device,
    )
    self._assigned_motion_lengths = torch.zeros(
      self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
    )
    env_ids = torch.arange(
      self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
    )
    self._assign_available(env_ids)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._env, name)

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._env.unwrapped

  @property
  def cfg(self):
    return self._env.cfg

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  def reset(self) -> tuple[TensorDict, dict]:
    obs, extras = self._env.reset()
    self.records.clear()
    self.finished = False
    self._next_motion_index = 0
    self._active_mask.zero_()
    self._assigned_motion_ids.fill_(-1)
    self._assigned_motion_lengths.zero_()
    env_ids = torch.arange(
      self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
    )
    self._assign_available(env_ids)
    return TensorDict(self.unwrapped.obs_buf, batch_size=[self.num_envs]), extras

  def get_observations(self) -> TensorDict:
    return TensorDict(self.unwrapped.obs_buf, batch_size=[self.num_envs])

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    pre_episode_lengths = self.unwrapped.episode_length_buf.clone()
    pre_motion_ids = self._assigned_motion_ids.clone()
    pre_motion_lengths = self._assigned_motion_lengths.clone()
    pre_active_mask = self._active_mask.clone()

    obs, rew, dones, extras = self._env.step(actions)

    done_env_ids = torch.where(dones.bool() & pre_active_mask)[0]
    if done_env_ids.numel() == 0:
      self.finished = bool(
        self._next_motion_index >= len(self._motion_files)
        and not self._active_mask.any()
      )
      return obs, rew, dones, extras

    terminated = self.unwrapped.reset_terminated[done_env_ids].detach().cpu()
    truncated = self.unwrapped.reset_time_outs[done_env_ids].detach().cpu()

    for idx, env_id in enumerate(done_env_ids.tolist()):
      motion_index = int(pre_motion_ids[env_id].item())
      motion_length = int(pre_motion_lengths[env_id].item())
      completed_steps = min(int(pre_episode_lengths[env_id].item()) + 1, motion_length)
      completion_ratio = completed_steps / float(max(motion_length, 1))
      self.records.append(
        {
          "motion_index": motion_index,
          "path": str(self._motion_files[motion_index].resolve()),
          "rank": self._rank,
          "completed_steps": completed_steps,
          "total_steps": motion_length,
          "completion_ratio": completion_ratio,
          "terminated": bool(terminated[idx].item()),
          "truncated": bool(truncated[idx].item()),
        }
      )

    self._assign_available(done_env_ids.to(device=self.unwrapped.device))
    self.finished = bool(
      self._next_motion_index >= len(self._motion_files) and not self._active_mask.any()
    )

    return obs, rew, dones, extras

  def close(self) -> None:
    self._env.close()

  def _assign_available(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    remaining = len(self._motion_files) - self._next_motion_index
    if remaining <= 0:
      self._active_mask[env_ids] = False
      self._assigned_motion_ids[env_ids] = -1
      self._assigned_motion_lengths[env_ids] = 0
      return

    assign_count = min(int(env_ids.numel()), remaining)
    assign_env_ids = env_ids[:assign_count]
    motion_indices = torch.arange(
      self._next_motion_index,
      self._next_motion_index + assign_count,
      device=self.unwrapped.device,
      dtype=torch.long,
    )
    self._next_motion_index += assign_count

    _assign_motion_indices(
      self.unwrapped, self._command, assign_env_ids, motion_indices
    )
    self._active_mask[assign_env_ids] = True
    self._assigned_motion_ids[assign_env_ids] = motion_indices
    self._assigned_motion_lengths[assign_env_ids] = self._command.motion.file_lengths[
      motion_indices
    ]

    if assign_count < int(env_ids.numel()):
      idle_env_ids = env_ids[assign_count:]
      self._active_mask[idle_env_ids] = False
      self._assigned_motion_ids[idle_env_ids] = -1
      self._assigned_motion_lengths[idle_env_ids] = 0


def _resolve_viewer_backend(
  viewer: Literal["none", "auto", "native", "viser"],
) -> Literal["native", "viser"] | None:
  if viewer == "none":
    return None
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"
  return viewer


def _run_viewer_loop(viewer: NativeMujocoViewer | ViserPlayViewer) -> None:
  viewer._interrupted = False
  viewer.setup()
  now = time.perf_counter()
  viewer._stats_last_time = now
  viewer._last_tick_time = now
  try:
    while (
      viewer.is_running()
      and not cast(Any, viewer.env).finished
      and not viewer._interrupted
    ):
      if not viewer.tick():
        time.sleep(0.001)
      viewer._update_stats()
  finally:
    viewer.close()


def _run_viewer_evaluate(
  task_id: str,
  cfg: EvaluateConfig,
  motion_files: list[Path],
  checkpoint_path: Path,
  checkpoint_label: str,
  motion_root: str,
) -> dict[str, Any]:
  device, rank, world_size = _runtime_rank_context(cfg)
  env_cfg = _prepare_filtering_env_cfg(load_env_cfg(task_id, play=False))
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  _configure_motion_command(
    motion_cmd,
    motion_path=motion_root,
    motion_type=cfg.motion_type,
    history_steps=cfg.history_steps,
    future_steps=cfg.future_steps,
  )
  env_cfg.scene.num_envs = min(cfg.num_envs, len(motion_files))

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  policy = _load_policy(task_id, vec_env, device, checkpoint_path)
  command = cast(Any, env.command_manager.get_term("motion"))
  filtering_env = FilteringEvalEnv(vec_env, motion_files, command, rank=rank)

  viewer_backend = _resolve_viewer_backend(cfg.viewer)
  assert viewer_backend is not None
  viewer = (
    NativeMujocoViewer(filtering_env, policy)
    if viewer_backend == "native"
    else ViserPlayViewer(filtering_env, policy)
  )
  _run_viewer_loop(viewer)

  report = _build_filter_report(
    task_id=task_id,
    motion_root=motion_root,
    checkpoint=checkpoint_label,
    threshold=cfg.failure_threshold,
    records=filtering_env.records,
    rank=rank,
    world_size=world_size,
  )
  output_path = _rank_output_path(cfg.output_file, rank, world_size)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

  print(
    f"[INFO] Evaluated {report['total_motion_count']} motions. "
    f"Failed: {report['failed_motion_count']} "
    f"({report['failed_motion_ratio']:.2%})."
  )
  print(f"[INFO] Report saved to {output_path.resolve()}")
  return report


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, Any]:
  configure_torch_backends()
  device, rank, world_size = _runtime_rank_context(cfg)
  motion_root = _resolve_motion_root(cfg)
  motion_files = _shard_motion_files(
    _collect_motion_files(motion_root), world_size, rank
  )
  checkpoint_path, checkpoint_label = _resolve_checkpoint_path(task_id, cfg)
  if cfg.viewer != "none":
    return _run_viewer_evaluate(
      task_id, cfg, motion_files, checkpoint_path, checkpoint_label, motion_root
    )

  env_cfg = _prepare_filtering_env_cfg(load_env_cfg(task_id, play=False))
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  _configure_motion_command(
    motion_cmd,
    motion_path=motion_root,
    motion_type=cfg.motion_type,
    history_steps=cfg.history_steps,
    future_steps=cfg.future_steps,
  )
  env_cfg.scene.num_envs = min(cfg.num_envs, len(motion_files))

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  policy = _load_policy(task_id, vec_env, device, checkpoint_path)

  command = cast(Any, env.command_manager.get_term("motion"))
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  assigned_motion_ids = torch.full(
    (env.num_envs,), -1, dtype=torch.long, device=env.device
  )
  assigned_motion_lengths = torch.zeros(
    env.num_envs, dtype=torch.long, device=env.device
  )

  next_motion_index = 0

  def assign_available(target_env_ids: torch.Tensor) -> None:
    nonlocal next_motion_index
    if target_env_ids.numel() == 0:
      return

    remaining = len(motion_files) - next_motion_index
    if remaining <= 0:
      active_mask[target_env_ids] = False
      assigned_motion_ids[target_env_ids] = -1
      assigned_motion_lengths[target_env_ids] = 0
      return

    assign_count = min(target_env_ids.numel(), remaining)
    assign_env_ids = target_env_ids[:assign_count]
    motion_indices = torch.arange(
      next_motion_index,
      next_motion_index + assign_count,
      device=env.device,
      dtype=torch.long,
    )
    next_motion_index += assign_count

    _assign_motion_indices(env, command, assign_env_ids, motion_indices)

    active_mask[assign_env_ids] = True
    assigned_motion_ids[assign_env_ids] = motion_indices
    assigned_motion_lengths[assign_env_ids] = command.motion.file_lengths[
      motion_indices
    ]

    if assign_count < target_env_ids.numel():
      idle_env_ids = target_env_ids[assign_count:]
      active_mask[idle_env_ids] = False
      assigned_motion_ids[idle_env_ids] = -1
      assigned_motion_lengths[idle_env_ids] = 0

  assign_available(env_ids)
  obs = TensorDict(env.obs_buf, batch_size=[env.num_envs])

  records: list[dict[str, Any]] = []
  progress = tqdm(total=len(motion_files), desc="Filtering motions", unit="motion")
  completed_motion_count = 0

  while completed_motion_count < len(motion_files):
    pre_episode_lengths = env.episode_length_buf.clone()
    pre_motion_ids = assigned_motion_ids.clone()
    pre_motion_lengths = assigned_motion_lengths.clone()
    pre_active_mask = active_mask.clone()

    with torch.no_grad():
      actions = policy(obs)

    obs, _, dones, _ = vec_env.step(actions)

    done_env_ids = torch.where(dones.bool() & pre_active_mask)[0]
    if done_env_ids.numel() == 0:
      continue

    terminated = env.reset_terminated[done_env_ids].detach().cpu()
    truncated = env.reset_time_outs[done_env_ids].detach().cpu()

    for idx, env_id in enumerate(done_env_ids.tolist()):
      motion_index = int(pre_motion_ids[env_id].item())
      total_steps = int(pre_motion_lengths[env_id].item())
      completed_steps = min(int(pre_episode_lengths[env_id].item()) + 1, total_steps)
      completion_ratio = completed_steps / float(max(total_steps, 1))
      records.append(
        {
          "motion_index": motion_index,
          "path": str(motion_files[motion_index].resolve()),
          "rank": rank,
          "completed_steps": completed_steps,
          "total_steps": total_steps,
          "completion_ratio": completion_ratio,
          "terminated": bool(terminated[idx].item()),
          "truncated": bool(truncated[idx].item()),
        }
      )

    completed_motion_count += int(done_env_ids.numel())
    progress.update(int(done_env_ids.numel()))

    assign_available(done_env_ids.to(device=env.device))
    obs = TensorDict(env.obs_buf, batch_size=[env.num_envs])

  progress.close()
  env.close()

  report = _build_filter_report(
    task_id=task_id,
    motion_root=motion_root,
    checkpoint=checkpoint_label,
    threshold=cfg.failure_threshold,
    records=records,
    rank=rank,
    world_size=world_size,
  )
  output_path = _rank_output_path(cfg.output_file, rank, world_size)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

  print(
    f"[INFO] Evaluated {report['total_motion_count']} motions. "
    f"Failed: {report['failed_motion_count']} "
    f"({report['failed_motion_ratio']:.2%})."
  )
  print(f"[INFO] Report saved to {output_path.resolve()}")
  return report


def launch_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, Any]:
  cfg = cast(EvaluateConfig, _prepare_launch_cfg(cfg))

  if cfg.gpu_ids is None:
    return run_evaluate(task_id, cfg)

  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    return run_evaluate(task_id, cfg)

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if cfg.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = cfg.torchrunx_log_dir
    else:
      output_path = Path(cfg.output_file)
      os.environ["TORCHRUNX_LOG_DIR"] = str(
        output_path.parent / f"{output_path.stem}_torchrunx"
      )

  print(f"[INFO] Launching data filtering with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_evaluate, task_id, cfg)

  rank_report_paths = [
    _rank_output_path(cfg.output_file, rank=rank, world_size=num_gpus)
    for rank in range(num_gpus)
  ]
  merged_report = _merge_filter_reports(rank_report_paths, Path(cfg.output_file))
  print(
    f"[INFO] Merged {len(rank_report_paths)} partial reports into "
    f"{Path(cfg.output_file).resolve()}"
  )
  return merged_report


def _empty_rollout_buffer() -> dict[str, list[np.ndarray]]:
  return {field: [] for field in _MOTION_NPZ_FIELDS}


def _capture_rollout_batch(
  command: Any, env_ids: torch.Tensor
) -> dict[str, np.ndarray]:
  """Capture current robot state for a batch of envs as CPU numpy arrays."""
  body_pos_w = command.robot_body_pos_w[env_ids]
  body_pos_w = body_pos_w - command._env.scene.env_origins[env_ids, None, :]
  return {
    "joint_pos": command.robot_joint_pos[env_ids].detach().cpu().numpy().copy(),
    "joint_vel": command.robot_joint_vel[env_ids].detach().cpu().numpy().copy(),
    "body_pos_w": body_pos_w.detach().cpu().numpy().copy(),
    "body_quat_w": command.robot_body_quat_w[env_ids].detach().cpu().numpy().copy(),
    "body_lin_vel_w": command.robot_body_lin_vel_w[env_ids]
    .detach()
    .cpu()
    .numpy()
    .copy(),
    "body_ang_vel_w": command.robot_body_ang_vel_w[env_ids]
    .detach()
    .cpu()
    .numpy()
    .copy(),
  }


def _append_rollout_batch(
  rollout_buffers: dict[int, dict[str, list[np.ndarray]]],
  command: Any,
  env_ids: torch.Tensor,
) -> None:
  if env_ids.numel() == 0:
    return

  batch = _capture_rollout_batch(command, env_ids)
  for batch_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
    buffer = rollout_buffers.setdefault(int(env_id), _empty_rollout_buffer())
    for field in _MOTION_NPZ_FIELDS:
      buffer[field].append(batch[field][batch_index])


def _stack_rollout_buffer(
  buffer: dict[str, list[np.ndarray]],
  *,
  fps: Any,
  frame_count: int,
) -> dict[str, np.ndarray]:
  rollout = {"fps": np.asarray(fps)}
  for field in _MOTION_NPZ_FIELDS:
    if len(buffer[field]) < frame_count:
      raise ValueError(
        f"Rollout field '{field}' has {len(buffer[field])} frames, "
        f"expected at least {frame_count}."
      )
    frames = buffer[field][:frame_count]
    if not frames:
      raise ValueError(f"Cannot save rollout with no frames for field '{field}'.")
    rollout[field] = np.stack(frames, axis=0)
  return rollout


def run_generate_dataset(task_id: str, cfg: GenerateDatasetConfig) -> dict[str, Any]:
  configure_torch_backends()
  device, rank, world_size = _runtime_rank_context(cfg)
  motion_root = _resolve_motion_root(cfg)
  motion_files = _shard_motion_files(
    _collect_motion_files(motion_root), world_size, rank
  )
  checkpoint_path, checkpoint_label = _resolve_checkpoint_path(task_id, cfg)

  env_cfg = _prepare_filtering_env_cfg(load_env_cfg(task_id, play=False))
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  _configure_motion_command(
    motion_cmd,
    motion_path=motion_root,
    motion_type=cfg.motion_type,
    history_steps=cfg.history_steps,
    future_steps=cfg.future_steps,
  )
  env_cfg.scene.num_envs = min(cfg.num_envs, len(motion_files))

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  policy = _load_policy(task_id, vec_env, device, checkpoint_path)

  command = cast(Any, env.command_manager.get_term("motion"))
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  assigned_motion_ids = torch.full(
    (env.num_envs,), -1, dtype=torch.long, device=env.device
  )
  assigned_motion_lengths = torch.zeros(
    env.num_envs, dtype=torch.long, device=env.device
  )
  rollout_buffers: dict[int, dict[str, list[np.ndarray]]] = {}

  next_motion_index = 0

  def assign_available(target_env_ids: torch.Tensor) -> None:
    nonlocal next_motion_index
    if target_env_ids.numel() == 0:
      return

    remaining = len(motion_files) - next_motion_index
    if remaining <= 0:
      active_mask[target_env_ids] = False
      assigned_motion_ids[target_env_ids] = -1
      assigned_motion_lengths[target_env_ids] = 0
      for env_id in target_env_ids.detach().cpu().tolist():
        rollout_buffers.pop(int(env_id), None)
      return

    assign_count = min(target_env_ids.numel(), remaining)
    assign_env_ids = target_env_ids[:assign_count]
    motion_indices = torch.arange(
      next_motion_index,
      next_motion_index + assign_count,
      device=env.device,
      dtype=torch.long,
    )
    next_motion_index += assign_count

    _assign_motion_indices(env, command, assign_env_ids, motion_indices)

    active_mask[assign_env_ids] = True
    assigned_motion_ids[assign_env_ids] = motion_indices
    assigned_motion_lengths[assign_env_ids] = command.motion.file_lengths[
      motion_indices
    ]
    for env_id in assign_env_ids.detach().cpu().tolist():
      rollout_buffers[int(env_id)] = _empty_rollout_buffer()

    if assign_count < target_env_ids.numel():
      idle_env_ids = target_env_ids[assign_count:]
      active_mask[idle_env_ids] = False
      assigned_motion_ids[idle_env_ids] = -1
      assigned_motion_lengths[idle_env_ids] = 0
      for env_id in idle_env_ids.detach().cpu().tolist():
        rollout_buffers.pop(int(env_id), None)

  assign_available(env_ids)
  obs = TensorDict(env.obs_buf, batch_size=[env.num_envs])

  saved_records: list[dict[str, Any]] = []
  failed_records: list[dict[str, Any]] = []
  progress = tqdm(total=len(motion_files), desc="Generating motions", unit="motion")
  completed_motion_count = 0
  output_motion_root = Path(cfg.output_motion_path)

  while completed_motion_count < len(motion_files):
    active_env_ids = torch.where(active_mask)[0]
    _append_rollout_batch(rollout_buffers, command, active_env_ids)

    pre_episode_lengths = env.episode_length_buf.clone()
    pre_motion_ids = assigned_motion_ids.clone()
    pre_motion_lengths = assigned_motion_lengths.clone()
    pre_active_mask = active_mask.clone()

    with torch.no_grad():
      actions = policy(obs)

    obs, _, dones, _ = vec_env.step(actions)

    done_env_ids = torch.where(dones.bool() & pre_active_mask)[0]
    if done_env_ids.numel() == 0:
      continue

    terminated = env.reset_terminated[done_env_ids].detach().cpu()
    truncated = env.reset_time_outs[done_env_ids].detach().cpu()

    for idx, env_id in enumerate(done_env_ids.tolist()):
      motion_index = int(pre_motion_ids[env_id].item())
      total_steps = int(pre_motion_lengths[env_id].item())
      completed_steps = min(int(pre_episode_lengths[env_id].item()) + 1, total_steps)
      completion_ratio = completed_steps / float(max(total_steps, 1))
      source_path = motion_files[motion_index]
      base_record = {
        "motion_index": motion_index,
        "path": str(source_path.resolve()),
        "rank": rank,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "completion_ratio": completion_ratio,
        "terminated": bool(terminated[idx].item()),
        "truncated": bool(truncated[idx].item()),
      }

      if completion_ratio >= cfg.completion_threshold:
        output_path = _output_motion_path_for(
          source_path, motion_root, output_motion_root
        )
        rollout = _stack_rollout_buffer(
          rollout_buffers[int(env_id)],
          fps=command.motion.fps_list[motion_index],
          frame_count=completed_steps,
        )
        _save_rollout_motion(output_path, rollout)
        saved_records.append(
          {
            **base_record,
            "output_path": str(output_path.resolve()),
          }
        )
      else:
        failed_records.append(base_record)

      rollout_buffers.pop(int(env_id), None)

    completed_motion_count += int(done_env_ids.numel())
    progress.update(int(done_env_ids.numel()))

    assign_available(done_env_ids.to(device=env.device))
    obs = TensorDict(env.obs_buf, batch_size=[env.num_envs])

  progress.close()
  env.close()

  report = _build_generate_dataset_report(
    task_id=task_id,
    motion_root=motion_root,
    output_motion_root=str(output_motion_root),
    checkpoint=checkpoint_label,
    threshold=cfg.completion_threshold,
    saved_records=saved_records,
    failed_records=failed_records,
    rank=rank,
    world_size=world_size,
  )
  output_path = _rank_output_path(cfg.output_file, rank, world_size)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

  print(
    f"[INFO] Evaluated {report['total_motion_count']} motions. "
    f"Saved: {report['saved_motion_count']} "
    f"({report['saved_motion_ratio']:.2%})."
  )
  print(f"[INFO] Generated dataset root: {output_motion_root.resolve()}")
  print(f"[INFO] Report saved to {output_path.resolve()}")
  return report


def launch_generate_dataset(task_id: str, cfg: GenerateDatasetConfig) -> dict[str, Any]:
  cfg = cast(GenerateDatasetConfig, _prepare_launch_cfg(cfg))

  if cfg.gpu_ids is None:
    return run_generate_dataset(task_id, cfg)

  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    return run_generate_dataset(task_id, cfg)

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if cfg.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = cfg.torchrunx_log_dir
    else:
      output_path = Path(cfg.output_file)
      os.environ["TORCHRUNX_LOG_DIR"] = str(
        output_path.parent / f"{output_path.stem}_torchrunx"
      )

  print(f"[INFO] Launching dataset generation with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_generate_dataset, task_id, cfg)

  rank_report_paths = [
    _rank_output_path(cfg.output_file, rank=rank, world_size=num_gpus)
    for rank in range(num_gpus)
  ]
  merged_report = _merge_generate_dataset_reports(
    rank_report_paths, Path(cfg.output_file)
  )
  print(
    f"[INFO] Merged {len(rank_report_paths)} partial reports into "
    f"{Path(cfg.output_file).resolve()}"
  )
  return merged_report


def run_delete(cfg: DeleteConfig) -> None:
  report_path = Path(cfg.report_file)
  if not report_path.exists():
    raise FileNotFoundError(f"Report file not found: {report_path}")

  with report_path.open("r", encoding="utf-8") as file:
    report = json.load(file)

  failed_paths = _extract_failed_motion_paths(report)
  deleted_count = 0
  missing_paths: list[Path] = []

  for motion_path in failed_paths:
    if motion_path.exists():
      motion_path.unlink()
      deleted_count += 1
    elif not cfg.missing_ok:
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    else:
      missing_paths.append(motion_path)

  print(f"[INFO] Deleted {deleted_count} motion files from {report_path.resolve()}.")
  if missing_paths:
    print(f"[INFO] Skipped {len(missing_paths)} missing files.")


def _print_usage() -> None:
  print("usage: data-filtering evaluate <TASK> [OPTIONS]")
  print("       data-filtering generate-dataset <TASK> [OPTIONS]")
  print("       data-filtering delete [OPTIONS]")
  print()
  print("Run 'data-filtering evaluate <TASK> --help' for evaluation options.")
  print(
    "Run 'data-filtering generate-dataset <TASK> --help' for dataset generation options."
  )
  print("Run 'data-filtering delete --help' for deletion options.")
  print("Run 'uv run list-envs' to list available tasks.")


def main() -> None:
  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    _print_usage()
    sys.exit(0)

  command = sys.argv[1]

  if command == "evaluate":
    if len(sys.argv) < 3 or sys.argv[2] in ("-h", "--help"):
      print("usage: data-filtering evaluate <TASK> [OPTIONS]")
      print("Run 'uv run list-envs' to list available tasks.")
      sys.exit(0)

    import mjlab.tasks as _mjlab_tasks  # noqa: F401

    tracking_tasks = [task for task in list_tasks() if "Tracking" in task]
    chosen_task, remaining_args = tyro.cli(
      tyro.extras.literal_type_from_choices(tracking_tasks),
      args=sys.argv[2:],
      add_help=False,
      return_unknown_args=True,
      config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
      EvaluateConfig,
      args=remaining_args,
      prog=sys.argv[0] + f" evaluate {chosen_task}",
      config=mjlab.TYRO_FLAGS,
    )
    launch_evaluate(chosen_task, args)
    return

  if command == "generate-dataset":
    if len(sys.argv) < 3 or sys.argv[2] in ("-h", "--help"):
      print("usage: data-filtering generate-dataset <TASK> [OPTIONS]")
      print("Run 'uv run list-envs' to list available tasks.")
      sys.exit(0)

    import mjlab.tasks as _mjlab_tasks  # noqa: F401

    tracking_tasks = [task for task in list_tasks() if "Tracking" in task]
    chosen_task, remaining_args = tyro.cli(
      tyro.extras.literal_type_from_choices(tracking_tasks),
      args=sys.argv[2:],
      add_help=False,
      return_unknown_args=True,
      config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
      GenerateDatasetConfig,
      args=remaining_args,
      prog=sys.argv[0] + f" generate-dataset {chosen_task}",
      config=mjlab.TYRO_FLAGS,
    )
    launch_generate_dataset(chosen_task, args)
    return

  if command == "delete":
    args = tyro.cli(
      DeleteConfig,
      args=sys.argv[2:],
      prog=sys.argv[0] + " delete",
      config=mjlab.TYRO_FLAGS,
    )
    run_delete(args)
    return

  print(f"Unknown command: {command}")
  _print_usage()
  sys.exit(1)


if __name__ == "__main__":
  main()
