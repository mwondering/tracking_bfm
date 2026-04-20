"""Replay a tracking motion stored in ``.npz`` with native MuJoCo rendering.

This script reuses the motion ordering logic from
``mjlab.tasks.tracking.mdp.multi_commands`` so it can play back motions saved in
either IsaacLab joint/body order or MuJoCo joint/body order.

Examples:
  uv run python -m mjlab.scripts.replay_npz /path/to/motion.npz --motion-type isaaclab
  uv run python -m mjlab.scripts.replay_npz /path/to/motion.npz --motion-type mujoco
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
import torch
import tyro

import mjlab
from mjlab.scene import Scene
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionLoader,
  _ISAACLAB_BODY_NAMES,
  _ISAACLAB_JOINT_NAMES,
  _MUJOCO_BODY_NAMES,
  _MUJOCO_JOINT_NAMES,
)

_ROOT_BODY_INDEX = torch.tensor([0], dtype=torch.long)
_REQUIRED_MOTION_KEYS = (
  "fps",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


@dataclass(frozen=True)
class ReplayConfig:
  motion_file: tyro.conf.Positional[str]
  motion_type: Literal["auto", "isaaclab", "mujoco"] = "auto"
  fps: float | None = None
  loop: bool = True
  start_frame: int = 0
  end_frame: int | None = None
  realtime_rate: float = 1.0
  track_body: str = "torso_link"
  distance: float = 2.8
  elevation: float = -5.0
  azimuth: float = 120.0


def _normalize_names(raw: np.ndarray) -> list[str]:
  names: list[str] = []
  for item in raw.tolist():
    if isinstance(item, bytes):
      names.append(item.decode())
    else:
      names.append(str(item))
  return names


def _infer_motion_type(data: np.lib.npyio.NpzFile) -> Literal["isaaclab", "mujoco"]:
  if "motion_type" in data.files:
    motion_type = str(np.asarray(data["motion_type"]).reshape(-1)[0]).lower()
    if motion_type in {"isaaclab", "mujoco"}:
      return motion_type  # type: ignore[return-value]

  if "joint_names" in data.files:
    joint_names = _normalize_names(np.asarray(data["joint_names"]))
    if joint_names == _ISAACLAB_JOINT_NAMES:
      return "isaaclab"
    if joint_names == _MUJOCO_JOINT_NAMES:
      return "mujoco"

  if "body_names" in data.files:
    body_names = _normalize_names(np.asarray(data["body_names"]))
    if body_names == _ISAACLAB_BODY_NAMES:
      return "isaaclab"
    if body_names == _MUJOCO_BODY_NAMES:
      return "mujoco"

  raise ValueError(
    "Unable to infer motion ordering from the npz file. "
    "Pass `--motion-type isaaclab` or `--motion-type mujoco` explicitly."
  )


def _resolve_motion_type(
  motion_file: Path,
  motion_type: Literal["auto", "isaaclab", "mujoco"],
) -> Literal["isaaclab", "mujoco"]:
  if motion_type != "auto":
    return motion_type

  with np.load(motion_file) as data:
    return _infer_motion_type(data)


def _validate_motion_file(motion_file: Path) -> None:
  if not motion_file.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")

  with np.load(motion_file) as data:
    missing = [key for key in _REQUIRED_MOTION_KEYS if key not in data.files]
    if missing:
      raise KeyError(f"Missing keys in motion npz: {missing}")


def _find_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
  for candidate in (name, f"robot/{name}"):
    obj_id = mujoco.mj_name2id(model, obj_type, candidate)
    if obj_id != -1:
      return obj_id
  raise ValueError(f"Failed to resolve {obj_type.name} named '{name}' in model.")


def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = quat_wxyz
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
      [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
      [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def _world_to_body_ang_vel(
  root_quat_wxyz: np.ndarray, ang_vel_world: np.ndarray
) -> np.ndarray:
  rot = _quat_to_rotmat(root_quat_wxyz)
  return rot.T @ ang_vel_world


def _build_model() -> mujoco.MjModel:
  env_cfg = unitree_g1_flat_tracking_env_cfg(play=True)
  scene = Scene(env_cfg.scene, device="cpu")
  return scene.compile()


def _reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
  else:
    mujoco.mj_resetData(model, data)


def _configure_camera(
  viewer: mujoco.viewer.Handle,
  model: mujoco.MjModel,
  cfg: ReplayConfig,
) -> None:
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
  viewer.cam.trackbodyid = _find_name(model, mujoco.mjtObj.mjOBJ_BODY, cfg.track_body)
  viewer.cam.fixedcamid = -1
  viewer.cam.distance = cfg.distance
  viewer.cam.elevation = cfg.elevation
  viewer.cam.azimuth = cfg.azimuth


def _frame_indices(total_frames: int, cfg: ReplayConfig) -> np.ndarray:
  start = cfg.start_frame
  end = total_frames if cfg.end_frame is None else cfg.end_frame
  if start < 0 or start >= total_frames:
    raise ValueError(f"start_frame must be in [0, {total_frames - 1}], got {start}")
  if end <= start or end > total_frames:
    raise ValueError(
      f"end_frame must be in ({start}, {total_frames}], got {end}"
    )
  return np.arange(start, end, dtype=np.int64)


def _write_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  root_qpos_adr: int,
  root_dof_adr: int,
  joint_qpos_adrs: np.ndarray,
  joint_dof_adrs: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  root_lin_vel: np.ndarray,
  root_ang_vel_world: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
) -> None:
  _reset_to_keyframe(model, data)

  data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos
  data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_quat
  data.qpos[joint_qpos_adrs] = joint_pos

  data.qvel[:] = 0.0
  data.qvel[root_dof_adr : root_dof_adr + 3] = root_lin_vel
  data.qvel[root_dof_adr + 3 : root_dof_adr + 6] = _world_to_body_ang_vel(
    root_quat, root_ang_vel_world
  )
  data.qvel[joint_dof_adrs] = joint_vel

  if model.nu > 0:
    data.ctrl[:] = 0.0

  mujoco.mj_forward(model, data)


def run_replay(cfg: ReplayConfig) -> None:
  motion_file = Path(cfg.motion_file).expanduser().resolve()
  _validate_motion_file(motion_file)
  resolved_motion_type = _resolve_motion_type(motion_file, cfg.motion_type)

  motion = MotionLoader(
    motion_file=str(motion_file),
    body_indexes=_ROOT_BODY_INDEX,
    motion_type=resolved_motion_type,
    device="cpu",
  )

  model = _build_model()
  data = mujoco.MjData(model)

  free_joint_ids = np.flatnonzero(
    model.jnt_type == mujoco.mjtJoint.mjJNT_FREE.value
  )
  if len(free_joint_ids) != 1:
    raise RuntimeError(
      f"Expected exactly one free joint in the replay model, got {len(free_joint_ids)}."
    )
  free_joint_id = int(free_joint_ids[0])
  root_qpos_adr = int(model.jnt_qposadr[free_joint_id])
  root_dof_adr = int(model.jnt_dofadr[free_joint_id])

  joint_qpos_adrs = np.array(
    [
      model.jnt_qposadr[_find_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
      for joint_name in _MUJOCO_JOINT_NAMES
    ],
    dtype=np.int64,
  )
  joint_dof_adrs = np.array(
    [
      model.jnt_dofadr[_find_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
      for joint_name in _MUJOCO_JOINT_NAMES
    ],
    dtype=np.int64,
  )

  joint_pos = motion.joint_pos.cpu().numpy()
  joint_vel = motion.joint_vel.cpu().numpy()
  root_pos = motion.body_pos_w[:, 0].cpu().numpy()
  root_quat = motion.body_quat_w[:, 0].cpu().numpy()
  root_lin_vel = motion.body_lin_vel_w[:, 0].cpu().numpy()
  root_ang_vel = motion.body_ang_vel_w[:, 0].cpu().numpy()

  frame_ids = _frame_indices(motion.time_step_total, cfg)
  motion_fps = float(np.asarray(motion.fps).reshape(-1)[0])
  replay_fps = cfg.fps if cfg.fps is not None else motion_fps
  if replay_fps <= 0.0:
    raise ValueError(f"fps must be positive, got {replay_fps}")
  if cfg.realtime_rate <= 0.0:
    raise ValueError(
      f"realtime_rate must be positive, got {cfg.realtime_rate}"
    )

  frame_dt = 1.0 / (replay_fps * cfg.realtime_rate)

  print(f"[INFO] Motion file: {motion_file}")
  print(f"[INFO] Motion type: {resolved_motion_type}")
  print(
    f"[INFO] Replay frames: {frame_ids[0]}..{frame_ids[-1]} "
    f"({len(frame_ids)} frames)"
  )
  print(f"[INFO] Motion fps: {motion_fps:.3f}")
  print(
    f"[INFO] Replay fps: {replay_fps:.3f} "
    f"(realtime_rate={cfg.realtime_rate:.3f}x)"
  )
  print("[INFO] Close the MuJoCo window or press Ctrl+C to exit.")

  frame_cursor = 0
  next_tick = time.perf_counter()

  with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
  ) as viewer:
    _configure_camera(viewer, model, cfg)

    while viewer.is_running():
      frame_idx = int(frame_ids[frame_cursor])
      _write_frame(
        model=model,
        data=data,
        root_qpos_adr=root_qpos_adr,
        root_dof_adr=root_dof_adr,
        joint_qpos_adrs=joint_qpos_adrs,
        joint_dof_adrs=joint_dof_adrs,
        root_pos=root_pos[frame_idx],
        root_quat=root_quat[frame_idx],
        root_lin_vel=root_lin_vel[frame_idx],
        root_ang_vel_world=root_ang_vel[frame_idx],
        joint_pos=joint_pos[frame_idx],
        joint_vel=joint_vel[frame_idx],
      )
      viewer.sync()

      frame_cursor += 1
      if frame_cursor >= len(frame_ids):
        if not cfg.loop:
          break
        frame_cursor = 0

      next_tick += frame_dt
      sleep_time = next_tick - time.perf_counter()
      if sleep_time > 0.0:
        time.sleep(sleep_time)
      else:
        next_tick = time.perf_counter()


def main() -> None:
  cfg = tyro.cli(ReplayConfig, description=__doc__, config=mjlab.TYRO_FLAGS)
  run_replay(cfg)


if __name__ == "__main__":
  main()
