"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import (
  dump_yaml,
  get_checkpoint_path,
  get_wandb_checkpoint_path,
  get_wandb_run_file_path,
  load_yaml,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  registry_name: str | None = None
  debug: bool = False
  """Disable W&B logging/upload while keeping normal local training behavior."""
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


@dataclass(frozen=True)
class _TrainBootstrapConfig:
  wandb_run_path: str | None = None


def _merge_mapping_into_cfg(target, overrides) -> None:
  """Recursively merge a mapping into a config object."""
  if overrides is None:
    return
  if isinstance(target, dict):
    for key, value in overrides.items():
      if key in target and (
        is_dataclass(target[key]) or isinstance(target[key], dict)
      ) and isinstance(value, dict):
        _merge_mapping_into_cfg(target[key], value)
      else:
        target[key] = value
    return

  if not is_dataclass(target):
    raise TypeError(f"Cannot merge mapping into non-dataclass config: {type(target)}")

  for key, value in overrides.items():
    current_value = getattr(target, key)
    if isinstance(current_value, dict) and isinstance(value, dict):
      _merge_mapping_into_cfg(current_value, value)
    elif is_dataclass(current_value) and isinstance(value, dict):
      _merge_mapping_into_cfg(current_value, value)
    else:
      setattr(target, key, value)


def _apply_wandb_run_configs(default_cfg: TrainConfig, run_path: Path) -> None:
  """Load env/agent config files from W&B and merge them into the defaults."""
  cache_root = Path("logs") / "rsl_rl"
  env_cfg_path, env_cached = get_wandb_run_file_path(
    cache_root, run_path, "params/env.yaml"
  )
  agent_cfg_path, agent_cached = get_wandb_run_file_path(
    cache_root, run_path, "params/agent.yaml"
  )

  _merge_mapping_into_cfg(default_cfg.env, load_yaml(env_cfg_path))
  _merge_mapping_into_cfg(default_cfg.agent, load_yaml(agent_cfg_path))

  cache_msg = "cached" if env_cached and agent_cached else "downloaded"
  print(
    f"[INFO] Loaded env/agent config from W&B run {run_path} ({cache_msg}).",
    flush=True,
  )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
  if cfg.debug:
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    if cfg.agent.run_name:
      if not cfg.agent.run_name.endswith("debug"):
        cfg.agent.run_name = f"{cfg.agent.run_name}_debug"
    else:
      cfg.agent.run_name = "debug"
    if rank == 0:
      print("[INFO] Debug mode enabled: using local logger only, W&B disabled.")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and (isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  ) or isinstance(
    cfg.env.commands["motion"], MultiMotionCommandCfg
  ))

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg))

    if isinstance(motion_cmd, MotionCommandCfg):
      motion_label = "motion file"
      motion_arg = "--env.commands.motion.motion-file /path/to/motion.npz"
      has_local_motion = bool(motion_cmd.motion_file) and Path(
        motion_cmd.motion_file
      ).exists()
    else:
      motion_label = "motion path"
      motion_arg = "--env.commands.motion.motion-path /path/to/motions_dir"
      has_local_motion = bool(motion_cmd.motion_path) and Path(
        motion_cmd.motion_path
      ).is_dir()

    if has_local_motion:
      if isinstance(motion_cmd, MotionCommandCfg):
        print(f"[INFO] Using local {motion_label}: {motion_cmd.motion_file}")
      else:
        print(f"[INFO] Using local {motion_label}: {motion_cmd.motion_path}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      artifact_dir = Path(artifact.download())
      if isinstance(motion_cmd, MotionCommandCfg):
        motion_cmd.motion_file = str(artifact_dir / "motion.npz")
      else:
        motion_cmd.motion_path = str(artifact_dir)
    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\n"
        f"  {motion_arg} (local {motion_label})"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(
    env,
    agent_cfg,
    str(log_dir),
    device,
    **runner_kwargs,
  )

  if not cfg.debug:
    add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  maybe_print_top_level_help("train")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  default_args = TrainConfig.from_task(chosen_task)
  bootstrap_args = tyro.cli(
    _TrainBootstrapConfig,
    args=remaining_args,
    default=_TrainBootstrapConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  if bootstrap_args.wandb_run_path is not None:
    _apply_wandb_run_configs(default_args, Path(bootstrap_args.wandb_run_path))

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=default_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, bootstrap_args, default_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
