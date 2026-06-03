"""Runner for G1 BFM wbteleop tracking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import os

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.distillation.rl.teacher import TeacherPolicyAdapter
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class WbTeleopTrackingRunner(MotionTrackingOnPolicyRunner):
  """Tracking runner for wbteleop PPO plus teacher-action BC."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device, registry_name=registry_name)
    self.teacher_adapter: TeacherPolicyAdapter | None = None

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    if self.teacher_adapter is None:
      self.teacher_adapter = self._build_teacher_adapter()
    self.alg.set_teacher_adapter(self.teacher_adapter)
    return super().learn(num_learning_iterations, init_at_random_ep_len)

  def _begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    if hasattr(self.alg, "set_learning_iteration"):
      self.alg.set_learning_iteration(iteration)
    super()._begin_adaptive_sampling_iteration(iteration)

  def _build_teacher_adapter(self) -> TeacherPolicyAdapter:
    algorithm_cfg = self.cfg.get("algorithm", {})
    checkpoint_path = algorithm_cfg.get("teacher_checkpoint_path", "")
    if not checkpoint_path:
      raise ValueError(
        "teacher_checkpoint_path must be provided for wbteleop training"
      )

    teacher_task_id = algorithm_cfg.get(
      "teacher_task_id",
      "Mjlab-Trackingbfm-Flat-Unitree-G1",
    )
    teacher_obs_group = algorithm_cfg.get("teacher_obs_group", "teacher_actor")
    teacher_runner_cls = load_runner_cls(teacher_task_id) or MjlabOnPolicyRunner
    teacher_cfg = asdict(load_rl_cfg(teacher_task_id))
    teacher_cfg["obs_groups"]["actor"] = (teacher_obs_group,)

    common_step_counter = getattr(self.env.unwrapped, "common_step_counter", None)
    with self._suppress_distributed_env_for_nested_runner():
      teacher_runner = teacher_runner_cls(
        self.env,
        teacher_cfg,
        log_dir=None,
        device=str(self.device),
      )
    teacher_runner.load(checkpoint_path, map_location=str(self.device))
    if common_step_counter is not None:
      self.env.unwrapped.common_step_counter = common_step_counter

    return TeacherPolicyAdapter(
      teacher_runner.get_inference_policy(device=self.device),
      obs_group=teacher_obs_group,
      policy_input_key=teacher_obs_group,
    )

  @contextmanager
  def _suppress_distributed_env_for_nested_runner(self):
    keys = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    old_values = {key: os.environ.get(key) for key in keys}
    for key in keys:
      os.environ.pop(key, None)
    try:
      yield
    finally:
      for key, value in old_values.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value
