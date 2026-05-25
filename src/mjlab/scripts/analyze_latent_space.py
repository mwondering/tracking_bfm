"""Collect and visualize latent distributions from latent distillation checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class LatentSpaceAnalysisConfig:
  checkpoint_file: str
  output_dir: str
  motion_path: str | None = None
  num_envs: int = 512
  num_points: int = 50_000
  device: str | None = None
  deterministic: bool = True
  sampling_mode: str | None = "uniform"
  motion_history_steps: int | None = None
  motion_future_steps: int | None = None
  proprio_history_length: int | None = None
  max_plot_points: int = 10_000
  plot_range: tuple[float, float] | None = (-20.0, 20.0)
  """Fixed plotting range for latent visualizations; set to None for autoscale."""


def _latent_policy_obs(policy, obs: TensorDict) -> TensorDict:
  return TensorDict(
    {
      policy.encoder_obs_group: obs[policy.encoder_obs_group],
      policy.decoder_obs_group: obs[policy.decoder_obs_group],
    },
    batch_size=list(obs.batch_size),
    device=obs.device,
  )


def _apply_latent_analysis_overrides(env_cfg, cfg: LatentSpaceAnalysisConfig) -> None:
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  motion_cfg = env_cfg.commands.get("motion")
  motion_overrides = (
    cfg.motion_path is not None
    or cfg.sampling_mode is not None
    or cfg.motion_history_steps is not None
    or cfg.motion_future_steps is not None
  )
  if motion_overrides and motion_cfg is None:
    raise ValueError("Latent analysis expects an env command named 'motion'.")
  if motion_cfg is not None:
    if cfg.motion_path is not None:
      motion_cfg.motion_path = cfg.motion_path
    if cfg.sampling_mode is not None:
      motion_cfg.sampling_mode = cfg.sampling_mode
    if cfg.motion_history_steps is not None:
      motion_cfg.history_steps = cfg.motion_history_steps
    if cfg.motion_future_steps is not None:
      motion_cfg.future_steps = cfg.motion_future_steps

  if cfg.proprio_history_length is not None:
    proprio_group = env_cfg.observations.get("proprio_actor")
    if proprio_group is None:
      raise ValueError("Latent analysis expects an observation group named 'proprio_actor'.")
    for term_name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
      term = proprio_group.terms.get(term_name)
      if term is None:
        raise ValueError(f"proprio_actor is missing term {term_name!r}.")
      term.history_length = cfg.proprio_history_length


@torch.no_grad()
def collect_latent_batches(
  *,
  env,
  policy,
  num_points: int,
  deterministic: bool,
  device: torch.device,
) -> dict[str, torch.Tensor]:
  """Collect latent samples while stepping parallel auto-reset environments."""
  policy.eval()
  obs = env.get_observations().to(device)
  collected: dict[str, list[torch.Tensor]] = {
    "mu": [],
    "log_std": [],
    "z": [],
    "dones": [],
    "step": [],
  }
  collected_points = 0
  step = 0

  while collected_points < num_points:
    policy_obs = _latent_policy_obs(policy, obs)
    mu, log_std = policy.encode(policy_obs)
    z = mu if deterministic else mu + torch.randn_like(mu) * torch.exp(log_std)
    if hasattr(policy, "decode"):
      actions = policy.decode(policy_obs, z)
    else:
      actions = policy.act(policy_obs, deterministic=deterministic)
    next_obs, _, dones, _ = env.step(actions.to(env.device))

    remaining = num_points - collected_points
    take = min(mu.shape[0], remaining)
    collected["mu"].append(mu[:take].detach().cpu())
    collected["log_std"].append(log_std[:take].detach().cpu())
    collected["z"].append(z[:take].detach().cpu())
    collected["dones"].append(dones[:take].detach().cpu().bool())
    collected["step"].append(torch.full((take,), step, dtype=torch.long))

    collected_points += take
    step += 1
    obs = next_obs.to(device)

  return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def _pca_2d(samples: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
  centered = samples - samples.mean(dim=0, keepdim=True)
  _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
  coords = centered @ vh[:2].T
  variance = singular_values.square()
  explained = variance[: min(2, variance.numel())] / torch.clamp(variance.sum(), min=1.0e-12)
  return coords.cpu().numpy(), explained.cpu().numpy()


def _effective_rank(eigenvalues: torch.Tensor) -> float:
  eigenvalues = torch.clamp(eigenvalues, min=0.0)
  total = eigenvalues.sum()
  if total <= 0:
    return 0.0
  probs = eigenvalues / total
  entropy = -(probs * torch.log(probs + 1.0e-12)).sum()
  return float(torch.exp(entropy).item())


def summarize_latents(latents: dict[str, torch.Tensor]) -> dict[str, Any]:
  z = latents["z"].float()
  mu = latents["mu"].float()
  log_std = latents["log_std"].float()
  radius = z.norm(dim=-1)
  centered = z - z.mean(dim=0, keepdim=True)
  cov = centered.T @ centered / max(z.shape[0] - 1, 1)
  diag_mask = torch.eye(cov.shape[0], dtype=torch.bool)
  offdiag = cov[~diag_mask]
  eigenvalues = torch.linalg.eigvalsh(cov).flip(0)
  prior = torch.randn_like(z)
  return {
    "num_points": int(z.shape[0]),
    "latent_dim": int(z.shape[-1]),
    "mu_mean_norm": float(mu.mean(dim=0).norm().item()),
    "z_mean_norm": float(z.mean(dim=0).norm().item()),
    "z_std_mean": float(z.std(dim=0, unbiased=False).mean().item()),
    "z_std_min": float(z.std(dim=0, unbiased=False).min().item()),
    "z_std_max": float(z.std(dim=0, unbiased=False).max().item()),
    "latent_std_mean": float(torch.exp(log_std).mean().item()),
    "cov_offdiag_mean_abs": float(offdiag.abs().mean().item()) if offdiag.numel() else 0.0,
    "radius_mean": float(radius.mean().item()),
    "radius_std": float(radius.std(unbiased=False).item()),
    "radius_q05": float(torch.quantile(radius, 0.05).item()),
    "radius_q50": float(torch.quantile(radius, 0.50).item()),
    "radius_q95": float(torch.quantile(radius, 0.95).item()),
    "effective_rank": _effective_rank(eigenvalues),
    "prior_radius_mean": float(prior.norm(dim=-1).mean().item()),
    "done_ratio": float(latents["dones"].float().mean().item()),
  }


def _coerce_plot_range(
  plot_range: tuple[float, float] | None,
) -> tuple[float, float] | None:
  if plot_range is None:
    return None
  lo, hi = float(plot_range[0]), float(plot_range[1])
  if not lo < hi:
    raise ValueError(f"plot_range must be increasing, got {plot_range}")
  return lo, hi


def save_latent_plots(
  latents: dict[str, torch.Tensor],
  output_dir: Path,
  max_plot_points: int,
  plot_range: tuple[float, float] | None = (-20.0, 20.0),
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output_dir.mkdir(parents=True, exist_ok=True)
  z = latents["z"].float()
  mu = latents["mu"].float()
  axis_range = _coerce_plot_range(plot_range)
  plot_count = min(max_plot_points, z.shape[0])
  plot_idx = torch.linspace(0, z.shape[0] - 1, plot_count).long()

  z_2d, z_explained = _pca_2d(z[plot_idx])
  prior = torch.randn_like(z[plot_idx])
  prior_2d, _ = _pca_2d(prior)

  plt.figure(figsize=(7, 6))
  plt.scatter(prior_2d[:, 0], prior_2d[:, 1], s=2, alpha=0.25, label="N(0,I)")
  plt.scatter(z_2d[:, 0], z_2d[:, 1], s=2, alpha=0.35, label="encoder z")
  plt.xlabel(f"PC1 ({z_explained[0] * 100:.1f}%)")
  plt.ylabel(f"PC2 ({z_explained[1] * 100:.1f}%)" if z_explained.size > 1 else "PC2")
  if axis_range is not None:
    plt.xlim(axis_range)
    plt.ylim(axis_range)
  plt.legend(markerscale=4)
  plt.tight_layout()
  plt.savefig(output_dir / "pca_z_vs_prior.png", dpi=180)
  plt.close()

  mu_2d, mu_explained = _pca_2d(mu[plot_idx])
  plt.figure(figsize=(7, 6))
  plt.scatter(mu_2d[:, 0], mu_2d[:, 1], s=2, alpha=0.35)
  plt.xlabel(f"PC1 ({mu_explained[0] * 100:.1f}%)")
  plt.ylabel(f"PC2 ({mu_explained[1] * 100:.1f}%)" if mu_explained.size > 1 else "PC2")
  if axis_range is not None:
    plt.xlim(axis_range)
    plt.ylim(axis_range)
  plt.tight_layout()
  plt.savefig(output_dir / "pca_mu.png", dpi=180)
  plt.close()

  radius = z.norm(dim=-1).cpu().numpy()
  prior_radius = torch.randn_like(z).norm(dim=-1).cpu().numpy()
  plt.figure(figsize=(7, 4))
  plt.hist(prior_radius, bins=80, alpha=0.45, density=True, label="N(0,I)")
  plt.hist(radius, bins=80, alpha=0.45, density=True, label="encoder z")
  plt.xlabel("||z||")
  plt.ylabel("density")
  if axis_range is not None:
    plt.xlim(axis_range)
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_dir / "radius_hist.png", dpi=180)
  plt.close()

  centered = z - z.mean(dim=0, keepdim=True)
  cov = centered.T @ centered / max(z.shape[0] - 1, 1)
  plt.figure(figsize=(6, 5))
  imshow_kwargs = {"cmap": "coolwarm"}
  if axis_range is not None:
    imshow_kwargs |= {"vmin": axis_range[0], "vmax": axis_range[1]}
  plt.imshow(cov.cpu().numpy(), **imshow_kwargs)
  plt.colorbar(label="covariance")
  plt.tight_layout()
  plt.savefig(output_dir / "cov_heatmap.png", dpi=180)
  plt.close()

  eigenvalues = torch.linalg.eigvalsh(cov).flip(0).cpu().numpy()
  plt.figure(figsize=(7, 4))
  plt.plot(eigenvalues)
  plt.xlabel("principal component")
  plt.ylabel("eigenvalue")
  if axis_range is not None:
    plt.ylim(axis_range)
  plt.tight_layout()
  plt.savefig(output_dir / "pca_spectrum.png", dpi=180)
  plt.close()


def run_analysis(task_id: str, cfg: LatentSpaceAnalysisConfig) -> Path:
  configure_torch_backends()
  device = torch.device(cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  _apply_latent_analysis_overrides(env_cfg, cfg)
  motion_cfg = env_cfg.commands.get("motion")
  if cfg.motion_path is not None and not isinstance(motion_cfg, MultiMotionCommandCfg):
    raise ValueError("Latent analysis currently expects a multi-motion tracking task.")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    raise ValueError(f"Task {task_id} does not define a runner class")
  runner = runner_cls(wrapped_env, asdict(agent_cfg), log_dir=None, device=str(device))
  runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=str(device))
  policy = runner.student_policy
  if not all(hasattr(policy, name) for name in ("encode", "decode", "encoder_obs_group", "decoder_obs_group")):
    raise ValueError("Checkpoint policy is not a latent distillation policy")

  try:
    latents = collect_latent_batches(
      env=wrapped_env,
      policy=policy,
      num_points=cfg.num_points,
      deterministic=cfg.deterministic,
      device=device,
    )
  finally:
    env.close()

  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_dir / "latents.npz", **{k: v.numpy() for k, v in latents.items()})
  summary = summarize_latents(latents)
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  save_latent_plots(
    latents,
    output_dir,
    max_plot_points=cfg.max_plot_points,
    plot_range=cfg.plot_range,
  )
  print(json.dumps(summary, indent=2))
  print(f"[INFO] Latent analysis written to {output_dir}")
  return output_dir


def main() -> None:
  maybe_print_top_level_help("analyze-latent-space")
  import mjlab.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
  )
  cfg = tyro.cli(
    LatentSpaceAnalysisConfig,
    args=remaining_args,
    prog=f"analyze-latent-space {task_id}",
  )
  if cfg.device is not None:
    os.environ.setdefault("MUJOCO_GL", "egl")
  run_analysis(task_id, cfg)


if __name__ == "__main__":
  main()
