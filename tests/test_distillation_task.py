"""Tests specific to distillation tasks."""

import mjlab.tasks.distillation.config.g1  # noqa: F401

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls


def test_distillation_task_is_registered() -> None:
  assert "Mjlab-Distillation-Flat-Unitree-G1" in list_tasks()


def test_distillation_task_loads_cfgs() -> None:
  task_id = "Mjlab-Distillation-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)

  assert "teacher_actor" in env_cfg.observations
  assert "student_actor" in env_cfg.observations
  assert runner_cls is not None
  assert rl_cfg.class_name == "DistillationRunner"
