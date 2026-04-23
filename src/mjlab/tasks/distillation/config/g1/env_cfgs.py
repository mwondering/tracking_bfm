"""Unitree G1 distillation environment configurations."""

from mjlab.tasks.distillation.distillation_env_cfg import make_distillation_env_cfg


def unitree_g1_flat_distillation_env_cfg(play: bool = False):
  return make_distillation_env_cfg(play=play)
