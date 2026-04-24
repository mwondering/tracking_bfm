import os
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def _get_multi_motion_command(self) -> MultiMotionCommand | None:
    motion_term = self.env.unwrapped.command_manager.get_term("motion")
    if isinstance(motion_term, MultiMotionCommand):
      return motion_term
    return None

  def _log_adaptive_sampling_motion_failure_report(self, it: int) -> None:
    if self.writer is None or self.logger_type != "wandb" or self.disable_logs:
      return
    if getattr(self, "gpu_global_rank", 0) != 0:
      return
    if wandb.run is None:
      return

    motion_term = self._get_multi_motion_command()
    if motion_term is None:
      return

    report = motion_term.get_motion_failure_report(top_k=10)
    self.writer.add_scalar(
      "Train/adaptive_sampling/motion_failure_rate_mean",
      report["mean_failure_rate"],
      it,
    )
    self.writer.add_scalar(
      "Train/adaptive_sampling/motion_failure_rate_max",
      report["max_failure_rate"],
      it,
    )
    self.writer.add_scalar(
      "Train/adaptive_sampling/motion_failure_rate_top10_min",
      report["top10_min_failure_rate"],
      it,
    )

    rows = report["rows"]
    table = wandb.Table(
      columns=[
        "rank",
        "motion_name",
        "motion_index",
        "failure_rate",
        "total_failures",
        "total_visits",
      ],
      data=[
        [
          row["rank"],
          row["motion_name"],
          row["motion_index"],
          row["failure_rate"],
          row["total_failures"],
          row["total_visits"],
        ]
        for row in rows
      ],
    )
    wandb.log(
      {"Train/adaptive_sampling/top10_motion_failure_rate": table},
      step=it,
    )

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    super().log(locs, width=width, pad=pad)
    self._log_adaptive_sampling_motion_failure_report(locs["it"])
