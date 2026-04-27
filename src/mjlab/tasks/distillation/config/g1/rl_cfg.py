"""RL configuration for the Unitree G1 distillation task."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  teacher_task_id: str = "Mjlab-Trackingbfm-Flat-Unitree-G1"
  teacher_checkpoint_path: str = ""
  teacher_obs_group: str = "teacher_actor"
  student_obs_group: str = "student_actor"
  beta_schedule: Literal["linear"] = "linear"
  beta_start: float = 1.0
  beta_end: float = 0.0
  beta_decay_steps: int = 2_00
  student_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (4096,2048,2048,1024, 1024, 1024, 512, 256, 128))
  student_activation: str = "elu"
  learning_rate: float = 1.0e-3
  num_learning_epochs: int = 5
  num_mini_batches: int = 4


def unitree_g1_distillation_runner_cfg() -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    experiment_name="g1_distillation",
    save_interval=5000,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
