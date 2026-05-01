"""Deterministic batch motion filtering for multi-motion tracking datasets."""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

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


@dataclass(frozen=True)
class DeleteConfig:
  report_file: str
  missing_ok: bool = True


def motion_sequence_complete(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
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

  for event_name in ("push_robot", "base_com", "encoder_bias", "foot_friction"):
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
    path for path in motion_root_path.rglob("*") if path.is_file() and path.suffix.lower() == ".npz"
  )
  if not motion_files:
    raise ValueError(f"No .npz motion files found under: {motion_root}")
  return motion_files


def _resolve_motion_root(cfg: EvaluateConfig) -> str:
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


def _resolve_checkpoint_path(task_id: str, cfg: EvaluateConfig) -> tuple[Path, str]:
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
    "total_motion_count": total_motion_count,
    "failed_motion_count": failed_motion_count,
    "failed_motion_ratio": failed_motion_ratio,
    "failed_motions": failed_records,
  }


def _extract_failed_motion_paths(report: dict[str, Any]) -> list[Path]:
  failed_paths = {
    Path(entry["path"]).resolve()
    for entry in report.get("failed_motions", [])
    if isinstance(entry, dict) and "path" in entry
  }
  return sorted(failed_paths)


def _load_policy(task_id: str, env: RslRlVecEnvWrapper, device: str, checkpoint_path: Path):
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


class FilteringEvalEnv:
  """VecEnv wrapper that records completed motions and refills idle env slots."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    motion_files: list[Path],
    command: Any,
  ) -> None:
    self._env = env
    self._motion_files = motion_files
    self._command = command
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
        self._next_motion_index >= len(self._motion_files) and not self._active_mask.any()
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
    has_display = bool(
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")  # type: ignore[name-defined]
    )
    return "native" if has_display else "viser"
  return viewer


def _run_viewer_loop(viewer: NativeMujocoViewer | ViserPlayViewer) -> None:
  viewer._interrupted = False
  viewer.setup()
  now = time.perf_counter()
  viewer._stats_last_time = now
  viewer._last_tick_time = now
  try:
    while viewer.is_running() and not viewer.env.finished and not viewer._interrupted:
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

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  policy = _load_policy(task_id, vec_env, env.device, checkpoint_path)
  command = cast(Any, env.command_manager.get_term("motion"))
  filtering_env = FilteringEvalEnv(vec_env, motion_files, command)

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
  )
  output_path = Path(cfg.output_file)
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
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  motion_root = _resolve_motion_root(cfg)
  motion_files = _collect_motion_files(motion_root)
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
    assigned_motion_lengths[assign_env_ids] = command.motion.file_lengths[motion_indices]

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
  )
  output_path = Path(cfg.output_file)
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
  print("       data-filtering delete [OPTIONS]")
  print()
  print("Run 'data-filtering evaluate <TASK> --help' for evaluation options.")
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
    run_evaluate(chosen_task, args)
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
