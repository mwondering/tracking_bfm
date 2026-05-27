from mjlab.tasks.latentvelocity.rl import LatentVelocityOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_flat_latent_rl_env_cfg,
  unitree_g1_rough_latent_rl_env_cfg,
)
from .rl_cfg import unitree_g1_latent_velocity_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-LatentRL-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_latent_rl_env_cfg(),
  play_env_cfg=unitree_g1_flat_latent_rl_env_cfg(play=True),
  rl_cfg=unitree_g1_latent_velocity_ppo_runner_cfg(),
  runner_cls=LatentVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-LatentRL-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_latent_rl_env_cfg(),
  play_env_cfg=unitree_g1_rough_latent_rl_env_cfg(play=True),
  rl_cfg=unitree_g1_latent_velocity_ppo_runner_cfg(),
  runner_cls=LatentVelocityOnPolicyRunner,
)
