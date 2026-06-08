"""Minimal deployment ONNX export for tracking_bfm actor checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch

import mjlab.tasks  # noqa: F401  # Trigger task registration via import side effects.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

CheckpointFamily = Literal["tracking", "distillation"]
_MINIMAL_METADATA_KEYS = ("task_id", "obs_group", "checkpoint_family", "robot_name")


def detect_checkpoint_family(checkpoint: dict) -> CheckpointFamily:
  """Infer checkpoint family from the saved state keys."""
  if "actor_state_dict" in checkpoint:
    return "tracking"
  if "policy_state_dict" in checkpoint:
    return "distillation"
  raise ValueError(
    "Unsupported checkpoint format: expected `actor_state_dict` or "
    "`policy_state_dict`."
  )


def resolve_deploy_onnx_path(
  checkpoint_path: str | Path, output_name: str | None = None
) -> Path:
  """Resolve the output ONNX path beside the source checkpoint."""
  checkpoint_path = Path(checkpoint_path)
  if output_name is None:
    output_name = f"deploy_{checkpoint_path.stem}.onnx"
  elif not output_name.endswith(".onnx"):
    output_name = f"{output_name}.onnx"
  return checkpoint_path.parent / output_name


def _minimal_metadata(
  *,
  task_id: str,
  obs_group: str,
  checkpoint_family: CheckpointFamily,
  robot_name: str | None = None,
) -> dict[str, str]:
  metadata = {
    "task_id": task_id,
    "obs_group": obs_group,
    "checkpoint_family": checkpoint_family,
    "robot_name": robot_name,
  }
  return {key: value for key, value in metadata.items() if value is not None}


def export_actor_model_to_onnx(
  *,
  actor,
  checkpoint_path: str | Path,
  task_id: str,
  checkpoint_family: CheckpointFamily,
  obs_group: str,
  output_name: str | None = None,
  robot_name: str | None = None,
  verbose: bool = False,
) -> Path:
  """Export a rebuilt actor module to a deployable ONNX file."""
  onnx_model = actor.as_onnx(verbose=verbose)
  onnx_model.to("cpu")
  onnx_model.eval()

  onnx_path = resolve_deploy_onnx_path(checkpoint_path, output_name=output_name)
  onnx_path.parent.mkdir(parents=True, exist_ok=True)

  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),  # type: ignore[operator]
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=onnx_model.input_names,  # type: ignore[arg-type]
    output_names=onnx_model.output_names,  # type: ignore[arg-type]
    dynamic_axes={},
    dynamo=False,
  )

  metadata = _minimal_metadata(
    task_id=task_id,
    obs_group=obs_group,
    checkpoint_family=checkpoint_family,
    robot_name=robot_name,
  )
  attach_metadata_to_onnx(str(onnx_path), metadata)
  return onnx_path


def _apply_motion_source(
  env_cfg: Any,
  *,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
) -> None:
  """Apply an optional local motion source to tracking task env config."""
  if motion_path is not None and motion_file is not None:
    raise ValueError("Provide only one of `motion_path` or `motion_file`.")

  motion_cfg = getattr(env_cfg, "commands", {}).get("motion")
  if motion_cfg is None or (motion_path is None and motion_file is None):
    return

  if motion_path is not None:
    if not hasattr(motion_cfg, "motion_path"):
      raise ValueError("This task motion command does not support `motion_path`.")
    motion_cfg.motion_path = str(motion_path)
    if hasattr(motion_cfg, "motion_file"):
      motion_cfg.motion_file = ""
    return

  if motion_file is not None:
    if not hasattr(motion_cfg, "motion_file"):
      raise ValueError("This task motion command does not support `motion_file`.")
    motion_cfg.motion_file = str(motion_file)
    if hasattr(motion_cfg, "motion_path"):
      motion_cfg.motion_path = ""


def _apply_distillation_student_obs_overrides(
  env_cfg: Any,
  *,
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
) -> None:
  """Apply student observation shape overrides used by distillation checkpoints."""
  if (
    student_history_steps is None
    and student_future_steps is None
    and student_robot_history_steps is None
  ):
    return

  motion_cfg = getattr(env_cfg, "commands", {}).get("motion")
  if motion_cfg is not None:
    if student_history_steps is not None and hasattr(motion_cfg, "history_steps"):
      motion_cfg.history_steps = int(student_history_steps)
    if student_future_steps is not None and hasattr(motion_cfg, "future_steps"):
      motion_cfg.future_steps = int(student_future_steps)

  observations = getattr(env_cfg, "observations", {})
  student_actor = observations.get("student_actor")
  if student_actor is None:
    return

  terms = getattr(student_actor, "terms", {})
  command_terms = ("ee_pose", "base_lin_vel_b", "base_ang_vel_b", "anchor_height_w")
  for term_name in command_terms:
    term = terms.get(term_name)
    if term is None:
      continue
    if student_history_steps is not None:
      term.params["history_steps"] = int(student_history_steps)
    if student_future_steps is not None:
      term.params["future_steps"] = int(student_future_steps)

  if student_robot_history_steps is None:
    return

  robot_state_terms = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  for term_name in robot_state_terms:
    term = terms.get(term_name)
    if term is not None:
      term.history_length = int(student_robot_history_steps)


def _apply_tracking_actor_obs_overrides(
  env_cfg: Any,
  *,
  obs_group: str = "actor",
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
) -> None:
  """Apply tracking actor observation overrides used by wbteleop checkpoints."""
  if (
    student_history_steps is None
    and student_future_steps is None
    and student_robot_history_steps is None
  ):
    return

  observations = getattr(env_cfg, "observations", {})
  actor = observations.get(obs_group)
  if actor is None:
    return

  terms = getattr(actor, "terms", {})
  ref_limb_ee_pose = terms.get("ref_limb_ee_pose_b")
  if ref_limb_ee_pose is not None:
    if student_history_steps is not None:
      ref_limb_ee_pose.params["history_steps"] = int(student_history_steps)
    if student_future_steps is not None:
      ref_limb_ee_pose.params["future_steps"] = int(student_future_steps)

  if student_robot_history_steps is None:
    return

  robot_state_terms = (
    "robot_limb_ee_pose_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  for term_name in robot_state_terms:
    term = terms.get(term_name)
    if term is not None:
      term.history_length = int(student_robot_history_steps)


def _build_actor_from_checkpoint(
  *,
  checkpoint_path: str | Path,
  task_id: str,
  checkpoint_family: CheckpointFamily,
  obs_group: str | None = None,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
  device: str = "cpu",
):
  """Rebuild and load the correct actor module for deployment export."""
  env = None
  try:
    env_cfg = load_env_cfg(task_id, play=True)
    _apply_motion_source(
      env_cfg,
      motion_path=motion_path,
      motion_file=motion_file,
    )
    if checkpoint_family == "distillation":
      _apply_distillation_student_obs_overrides(
        env_cfg,
        student_history_steps=student_history_steps,
        student_future_steps=student_future_steps,
        student_robot_history_steps=student_robot_history_steps,
      )
    else:
      _apply_tracking_actor_obs_overrides(
        env_cfg,
        obs_group=obs_group or "actor",
        student_history_steps=student_history_steps,
        student_future_steps=student_future_steps,
        student_robot_history_steps=student_robot_history_steps,
      )
    runner_cfg = asdict(load_rl_cfg(task_id))
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env)
    runner = runner_cls(wrapped_env, runner_cfg, log_dir=None, device=device)
    runner.load(str(checkpoint_path), map_location=device)

    if checkpoint_family == "distillation":
      actor = runner.student_policy
      resolved_obs_group = obs_group or load_rl_cfg(task_id).student_obs_group
    else:
      actor = runner.alg.get_policy()
      resolved_obs_group = obs_group or "actor"

    actor.to("cpu")
    actor.eval()
    return actor, resolved_obs_group
  finally:
    if env is not None:
      env.close()


def export_checkpoint_to_onnx(
  *,
  checkpoint_path: str | Path,
  task_id: str,
  checkpoint_family: CheckpointFamily | Literal["auto"] = "auto",
  obs_group: str | None = None,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
  output_name: str | None = None,
  robot_name: str | None = None,
  device: str = "cpu",
  verbose: bool = False,
) -> Path:
  """Export a tracking_bfm checkpoint to a minimal deployment ONNX."""
  checkpoint_path = Path(checkpoint_path)
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  resolved_family = (
    detect_checkpoint_family(checkpoint)
    if checkpoint_family == "auto"
    else checkpoint_family
  )
  actor, resolved_obs_group = _build_actor_from_checkpoint(
    checkpoint_path=checkpoint_path,
    task_id=task_id,
    checkpoint_family=resolved_family,
    obs_group=obs_group,
    motion_path=motion_path,
    motion_file=motion_file,
    student_history_steps=student_history_steps,
    student_future_steps=student_future_steps,
    student_robot_history_steps=student_robot_history_steps,
    device=device,
  )
  return export_actor_model_to_onnx(
    actor=actor,
    checkpoint_path=checkpoint_path,
    task_id=task_id,
    checkpoint_family=resolved_family,
    obs_group=resolved_obs_group,
    output_name=output_name,
    robot_name=robot_name,
    verbose=verbose,
  )
