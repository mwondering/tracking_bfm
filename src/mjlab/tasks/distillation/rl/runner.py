"""Distillation runner with mixed teacher/student rollout."""

from __future__ import annotations

from contextlib import contextmanager
import os
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any

import torch
from tensordict import TensorDict

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from .algorithm import ActionDistillationAlgorithm
from .models import build_student_model
from .schedules import LinearTeacherMixSchedule
from .teacher import TeacherPolicyAdapter


def mix_rollout_actions(
  student_actions: torch.Tensor,
  teacher_actions: torch.Tensor,
  beta: float,
  generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Select rollout actions via Bernoulli switching."""
  if beta <= 0.0:
    teacher_mask = torch.zeros(
      student_actions.shape[0], dtype=torch.bool, device=student_actions.device
    )
  elif beta >= 1.0:
    teacher_mask = torch.ones(
      student_actions.shape[0], dtype=torch.bool, device=student_actions.device
    )
  else:
    probs = torch.full(
      (student_actions.shape[0],), beta, dtype=torch.float32, device=student_actions.device
    )
    teacher_mask = torch.bernoulli(probs, generator=generator).to(dtype=torch.bool)

  rollout_actions = torch.where(
    teacher_mask.unsqueeze(-1), teacher_actions, student_actions
  )
  return rollout_actions, teacher_mask


class DistillationRunner:
  """Standalone runner for pure action distillation."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    teacher_adapter: TeacherPolicyAdapter | None = None,
    registry_name: str | None = None,
  ):
    self.env = env
    self.cfg = train_cfg
    self.log_dir = log_dir
    self.device = torch.device(device)
    self._configure_multi_gpu()
    self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
    self.save_interval = int(self.cfg["save_interval"])
    self.student_obs_group = self.cfg["student_obs_group"]
    self.teacher_obs_group = self.cfg["teacher_obs_group"]

    self.git_status_repos: list[str] = []
    self.writer = None
    self.logger = None
    self.logger_type = self.cfg.get("logger", "tensorboard").lower()
    self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

    self.current_learning_iteration = 0
    self.tot_timesteps = 0
    self.tot_time = 0.0
    self.last_loss_dict: dict[str, float] = {}
    self.last_train_metrics: dict[str, float] = {}

    obs = self.env.get_observations().to(self.device)
    self.student_policy = build_student_model(
      obs=obs,
      student_obs_group=self.student_obs_group,
      action_dim=self.env.num_actions,
      hidden_dims=tuple(self.cfg["student_hidden_dims"]),
      activation=self.cfg["student_activation"],
      obs_normalization=True,
    ).to(self.device)
    self.alg = ActionDistillationAlgorithm(
      policy=self.student_policy,
      learning_rate=float(self.cfg["learning_rate"]),
      max_grad_norm=1.0,
      multi_gpu_cfg=self.multi_gpu_cfg,
    )
    self.mix_schedule = LinearTeacherMixSchedule(
      beta_start=float(self.cfg["beta_start"]),
      beta_end=float(self.cfg["beta_end"]),
      decay_steps=int(self.cfg["beta_decay_steps"]),
    )
    self.teacher_adapter = teacher_adapter
    self.registry_name = registry_name

  def _configure_multi_gpu(self) -> None:
    self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    self.is_distributed = self.gpu_world_size > 1

    if not self.is_distributed:
      self.gpu_local_rank = 0
      self.gpu_global_rank = 0
      self.multi_gpu_cfg = None
      return

    self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    self.gpu_global_rank = int(os.getenv("RANK", "0"))
    expected_device = f"cuda:{self.gpu_local_rank}"
    if str(self.device) != expected_device:
      raise ValueError(
        f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
      )

    self.multi_gpu_cfg = {
      "global_rank": self.gpu_global_rank,
      "local_rank": self.gpu_local_rank,
      "world_size": self.gpu_world_size,
    }
    torch.distributed.init_process_group(
      backend="nccl",
      rank=self.gpu_global_rank,
      world_size=self.gpu_world_size,
    )
    torch.cuda.set_device(self.gpu_local_rank)

  def add_git_repo_to_log(self, repo_file_path):
    self.git_status_repos.append(repo_file_path)

  def train_mode(self) -> None:
    self.student_policy.train()

  def eval_mode(self) -> None:
    self.student_policy.eval()

  def get_inference_policy(self, device=None):
    self.eval_mode()
    if device is not None:
      self.student_policy.to(device)
    return self.student_policy

  def save(self, path: str, infos=None) -> None:
    env_state = {
      "common_step_counter": getattr(self.env.unwrapped, "common_step_counter", 0)
    }
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
    torch.save(saved_dict, path)
    if (
      self.logger_type in {"wandb", "neptune"}
      and self.writer is not None
      and self.cfg.get("upload_model", True)
      and hasattr(self.writer, "save_model")
    ):
      self.writer.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ):
    checkpoint = torch.load(path, weights_only=False, map_location=map_location)
    if "policy_state_dict" not in checkpoint:
      if "actor_state_dict" in checkpoint:
        raise ValueError(
          "Checkpoint is not a distillation student checkpoint: found "
          "`actor_state_dict` but missing `policy_state_dict`. "
          "You likely passed a tracking/teacher checkpoint to the "
          "distillation play path."
        )
      raise ValueError(
        "Checkpoint is not a distillation student checkpoint: missing "
        "`policy_state_dict`."
      )
    load_actor = True if load_cfg is None else bool(load_cfg.get("actor", False))
    if load_actor:
      self.student_policy.load_state_dict(
        checkpoint["policy_state_dict"],
        strict=strict,
      )
    if load_cfg is None and "optimizer_state_dict" in checkpoint:
      self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.current_learning_iteration = checkpoint.get("iter", 0)
    infos = checkpoint.get("infos")
    if infos and "env_state" in infos and hasattr(self.env.unwrapped, "common_step_counter"):
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if not self.disable_logs:
      self._prepare_logging_writer()
    teacher_adapter = self._get_teacher_adapter()
    if self.is_distributed:
      self.alg.broadcast_parameters()

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.train_mode()

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float32, device=self.device
    )

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start_time = time.time()
      ep_infos: list[dict[str, Any]] = []
      student_rollout: list[torch.Tensor] = []
      teacher_rollout: list[torch.Tensor] = []
      teacher_masks: list[torch.Tensor] = []

      with torch.no_grad():
        beta = self.mix_schedule(it)
        for _ in range(self.num_steps_per_env):
          student_obs = self._student_obs_from(obs)
          student_actions = self.student_policy(student_obs)
          teacher_actions = teacher_adapter.act_mean(obs)
          rollout_actions, teacher_mask = mix_rollout_actions(
            student_actions, teacher_actions, beta
          )

          next_obs, rewards, dones, extras = self.env.step(
            rollout_actions.to(self.env.device)
          )

          next_obs = next_obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)

          student_rollout.append(student_obs[self.student_obs_group].detach().clone())
          teacher_rollout.append(teacher_actions.detach().clone())
          teacher_masks.append(teacher_mask.detach().clone())

          episode_log = extras.get("episode") or extras.get("log")
          if episode_log is not None:
            ep_infos.append(episode_log)
          cur_reward_sum += rewards
          cur_episode_length += 1

          done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
          if done_ids.numel() > 0:
            rewbuffer.extend(cur_reward_sum[done_ids].cpu().tolist())
            lenbuffer.extend(cur_episode_length[done_ids].cpu().tolist())
            cur_reward_sum[done_ids] = 0.0
            cur_episode_length[done_ids] = 0.0

          obs = next_obs

      collection_time = time.time() - start_time
      learn_start = time.time()

      student_batch = TensorDict(
        {
          self.student_obs_group: torch.cat(student_rollout, dim=0),
        },
        batch_size=[self.env.num_envs * self.num_steps_per_env],
        device=self.device,
      )
      teacher_batch = torch.cat(teacher_rollout, dim=0)
      self.last_loss_dict = {
        key: self._distributed_mean(value)
        for key, value in self.alg.update(
        student_obs=student_batch,
        teacher_actions=teacher_batch,
        num_learning_epochs=int(self.cfg["num_learning_epochs"]),
        num_mini_batches=int(self.cfg["num_mini_batches"]),
        ).items()
      }
      learn_time = time.time() - learn_start

      teacher_mask = torch.cat(teacher_masks, dim=0)
      self.last_train_metrics = {
        "beta_teacher": self._distributed_mean(float(beta)),
        "teacher_action_ratio": self._distributed_mean(
          float(teacher_mask.float().mean().item())
        ),
        "student_action_ratio": self._distributed_mean(
          float((~teacher_mask).float().mean().item())
        ),
      }
      env_metrics, aggregated_ep_info = self._collect_distributed_log_data(
        ep_infos=ep_infos,
        rewbuffer=rewbuffer,
        lenbuffer=lenbuffer,
      )
      self.current_learning_iteration = it

      if self.log_dir is not None and not self.disable_logs:
        self._log_train_iteration(
          it=it,
          total_iterations=tot_iter,
          collection_time=collection_time,
          learn_time=learn_time,
          env_metrics=env_metrics,
          aggregated_ep_info=aggregated_ep_info,
        )
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir is not None and not self.disable_logs:
      self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

  def _prepare_logging_writer(self) -> None:
    if self.disable_logs:
      return
    if self.log_dir is None or self.writer is not None:
      return

    if self.logger_type == "wandb":
      from rsl_rl.utils.wandb_utils import WandbSummaryWriter

      self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
    elif self.logger_type == "neptune":
      from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

      self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
    elif self.logger_type == "tensorboard":
      from torch.utils.tensorboard import SummaryWriter

      self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    else:
      raise ValueError(f"Unsupported logger type: {self.logger_type}")
    self.logger = self.writer

  def _build_teacher_adapter(self) -> TeacherPolicyAdapter:
    checkpoint_path = self.cfg.get("teacher_checkpoint_path", "")
    if not checkpoint_path:
      raise ValueError(
        "teacher_checkpoint_path must be provided when teacher_adapter is not injected"
      )

    teacher_task_id = self.cfg["teacher_task_id"]
    teacher_runner_cls = load_runner_cls(teacher_task_id) or MjlabOnPolicyRunner
    teacher_cfg = asdict(load_rl_cfg(teacher_task_id))

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
      obs_group=self.teacher_obs_group,
      policy_input_key="actor",
    )

  def _get_teacher_adapter(self) -> TeacherPolicyAdapter:
    if self.teacher_adapter is None:
      self.teacher_adapter = self._build_teacher_adapter()
    return self.teacher_adapter

  @contextmanager
  def _suppress_distributed_env_for_nested_runner(self):
    env_keys = ("WORLD_SIZE", "RANK", "LOCAL_RANK")
    original_env = {key: os.environ.get(key) for key in env_keys}
    try:
      for key in env_keys:
        os.environ.pop(key, None)
      yield
    finally:
      for key, value in original_env.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value

  def _distributed_mean(self, value: float) -> float:
    if not self.is_distributed:
      return value
    tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= self.gpu_world_size
    return float(tensor.item())

  def _collect_distributed_log_data(
    self,
    *,
    ep_infos: list[dict[str, Any]],
    rewbuffer: deque,
    lenbuffer: deque,
  ) -> tuple[dict[str, float], dict[str, float]]:
    env_metrics: dict[str, float] = {}
    if len(rewbuffer) > 0:
      env_metrics["mean_reward"] = self._distributed_mean(statistics.mean(rewbuffer))
      env_metrics["mean_episode_length"] = self._distributed_mean(
        statistics.mean(lenbuffer)
      )

    aggregated_ep_info: dict[str, float] = {}
    if ep_infos:
      aggregated: dict[str, list[float]] = defaultdict(list)
      for ep_info in ep_infos:
        for key, value in ep_info.items():
          aggregated[key].append(self._to_float(value))
      aggregated_ep_info = {
        key: self._distributed_mean(sum(values) / len(values))
        for key, values in aggregated.items()
      }

    return env_metrics, aggregated_ep_info

  def _log_train_iteration(
    self,
    *,
    it: int,
    total_iterations: int,
    collection_time: float,
    learn_time: float,
    env_metrics: dict[str, float],
    aggregated_ep_info: dict[str, float],
  ) -> None:
    if self.writer is None:
      return

    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
    self.tot_timesteps += collection_size
    self.tot_time += collection_time + learn_time
    fps = int(collection_size / max(collection_time + learn_time, 1.0e-6))

    for key, value in self.last_loss_dict.items():
      self.writer.add_scalar(f"Train/distill/{key}", value, it)
    for key, value in self.last_train_metrics.items():
      self.writer.add_scalar(f"Train/distill/{key}", value, it)

    if "mean_reward" in env_metrics:
      self.writer.add_scalar("Train/env/mean_reward", env_metrics["mean_reward"], it)
      self.writer.add_scalar(
        "Train/env/mean_episode_length",
        env_metrics["mean_episode_length"],
        it,
      )

    for key, value in aggregated_ep_info.items():
      self.writer.add_scalar(f"Train/env/{key}", value, it)

    self.writer.add_scalar("Perf/total_fps", fps, it)
    self.writer.add_scalar("Perf/collection_time", collection_time, it)
    self.writer.add_scalar("Perf/learning_time", learn_time, it)
    print(
      self._format_terminal_log(
        it=it,
        total_iterations=total_iterations,
        collection_time=collection_time,
        learn_time=learn_time,
        fps=fps,
        aggregated_ep_info=aggregated_ep_info,
        env_metrics=env_metrics,
      )
    )

  def _format_terminal_log(
    self,
    *,
    it: int,
    total_iterations: int,
    collection_time: float,
    learn_time: float,
    fps: int,
    aggregated_ep_info: dict[str, float],
    env_metrics: dict[str, float],
  ) -> str:
    width = 80
    pad = 35
    iteration_time = collection_time + learn_time
    lines = [
      "#" * width,
      f" Learning iteration {it}/{total_iterations} ".center(width, " "),
      "",
      f"{'Computation:':>{pad}} {fps:.0f} steps/s "
      f"(collection: {collection_time:.3f}s, learning {learn_time:.3f}s)",
    ]

    for key, value in self.last_loss_dict.items():
      lines.append(f"{f'{key}:':>{pad}} {value:.4f}")
    for key, value in self.last_train_metrics.items():
      lines.append(f"{f'{key}:':>{pad}} {value:.4f}")

    if "mean_reward" in env_metrics:
      lines.append(f"{'Mean reward:':>{pad}} {env_metrics['mean_reward']:.4f}")
      lines.append(
        f"{'Mean episode length:':>{pad}} {env_metrics['mean_episode_length']:.4f}"
      )

    for key, value in aggregated_ep_info.items():
      lines.append(f"{f'{key}:':>{pad}} {value:.4f}")

    lines.extend(
      [
        "-" * width,
        f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
        f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
        f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
      ]
    )
    return "\n".join(lines)

  def _student_obs_from(self, obs: TensorDict) -> TensorDict:
    return TensorDict(
      {self.student_obs_group: obs[self.student_obs_group]},
      batch_size=list(obs.batch_size),
      device=obs.device,
    )

  @staticmethod
  def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
      if value.numel() == 1:
        return float(value.item())
      return float(value.float().mean().item())
    return float(value)
