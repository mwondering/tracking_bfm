"""Tests for latent velocity RL task configuration."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def test_latent_velocity_task_is_registered() -> None:
  from mjlab.tasks.latentvelocity.rl import LatentVelocityOnPolicyRunner

  task_id = "Mjlab-LatentRL-Flat-Unitree-G1"

  assert task_id in list_tasks()
  assert load_runner_cls(task_id) is LatentVelocityOnPolicyRunner
  assert load_rl_cfg(task_id).latent_dim == 64
  assert load_rl_cfg(task_id).latent_action_clip == 6.0


def test_latent_velocity_env_removes_gait_specific_rewards() -> None:
  baseline = unitree_g1_flat_env_cfg()
  cfg = load_env_cfg("Mjlab-LatentRL-Flat-Unitree-G1")

  removed_rewards = {
    "upright",
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
  }
  assert set(cfg.rewards.keys()) == (
    set(baseline.rewards.keys()) - removed_rewards | {"waist_joint_vel_l2"}
  )

  for name, reward in baseline.rewards.items():
    if name in removed_rewards:
      continue
    latent_reward = cfg.rewards[name]
    assert latent_reward.func is reward.func
    expected_weight = (
      3.0
      if name in {"track_linear_velocity", "track_angular_velocity"}
      else reward.weight
    )
    assert latent_reward.weight == expected_weight
    expected_params = dict(reward.params)
    if name == "track_linear_velocity":
      expected_params["penalize_z_velocity"] = False
    elif name == "track_angular_velocity":
      expected_params["penalize_xy_angular_velocity"] = False
    assert latent_reward.params == expected_params

  assert "penalize_z_velocity" not in baseline.rewards["track_linear_velocity"].params
  assert (
    "penalize_xy_angular_velocity"
    not in baseline.rewards["track_angular_velocity"].params
  )
  assert cfg.rewards["track_linear_velocity"].params["penalize_z_velocity"] is False
  assert (
    cfg.rewards["track_angular_velocity"].params["penalize_xy_angular_velocity"]
    is False
  )
  waist_reward = cfg.rewards["waist_joint_vel_l2"]
  assert waist_reward.func is mdp.joint_vel_l2
  assert waist_reward.weight == -0.05
  assert waist_reward.params["asset_cfg"].joint_names == (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  )
  assert "waist_joint_vel_l2" not in baseline.rewards

  assert cfg.commands.keys() == baseline.commands.keys()
  assert cfg.terminations.keys() == baseline.terminations.keys()
  assert cfg.curriculum.keys() == baseline.curriculum.keys()


def test_latent_velocity_proprio_actor_matches_first_stage_proprio_terms() -> None:
  cfg = load_env_cfg("Mjlab-LatentRL-Flat-Unitree-G1")

  assert "proprio_actor" in cfg.observations
  terms = cfg.observations["proprio_actor"].terms
  assert tuple(terms.keys()) == (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  for term in terms.values():
    assert term.history_length == 0


def test_latent_velocity_train_help_exposes_decoder_flags() -> None:
  result = subprocess.run(
    ["uv", "run", "train", "Mjlab-LatentRL-Flat-Unitree-G1", "--help"],
    check=True,
    capture_output=True,
    text=True,
  )

  assert "--agent.latent-decoder-checkpoint-path" in result.stdout
  assert "--agent.latent-action-clip" in result.stdout
  assert "--agent.proprio-obs-group" in result.stdout


def test_latent_velocity_play_help_exposes_rl_decoder_flags() -> None:
  result = subprocess.run(
    ["uv", "run", "play", "Mjlab-LatentRL-Flat-Unitree-G1", "--help"],
    check=True,
    capture_output=True,
    text=True,
  )

  assert "--agent {zero,random,trained}" in result.stdout
  assert "--stochastic-policy" in result.stdout
  assert "--rl.latent-decoder-checkpoint-path" in result.stdout
  assert "--rl.latent-action-clip" in result.stdout
  assert "--rl.proprio-obs-group" in result.stdout


def test_tracking_play_help_keeps_agent_mode_cli() -> None:
  result = subprocess.run(
    ["uv", "run", "play", "Mjlab-Trackingbfm-Flat-Unitree-G1", "--help"],
    check=True,
    capture_output=True,
    text=True,
  )

  assert "--agent {zero,random,trained}" in result.stdout
  assert "--motion-file" in result.stdout
  assert "--rl.experiment-name" in result.stdout


def test_play_uses_runner_wrapped_env_for_viewer(monkeypatch, tmp_path: Path) -> None:
  from mjlab.scripts import play as play_script

  checkpoint = tmp_path / "model.pt"
  checkpoint.write_text("dummy")
  captured = {}

  @dataclass
  class TestRlCfg(RslRlBaseRunnerCfg):
    experiment_name: str = "test_play"
    clip_actions: float | None = None

  class FakeRawEnv:
    render_mode = None

    def close(self) -> None:
      captured["closed"] = True

  class FakeEnvCfg:
    commands = {}
    terminations = {}

  class FakeRslEnv:
    def __init__(self, env, clip_actions=None) -> None:
      self.env = env
      self.clip_actions = clip_actions
      self.unwrapped = env

    def close(self) -> None:
      self.env.close()

  class FakeRunnerWrappedEnv:
    def __init__(self, env) -> None:
      self.env = env
      self.unwrapped = env.unwrapped

    def close(self) -> None:
      self.env.env.close()

  class FakeRunner:
    def __init__(self, env, train_cfg, device="cpu") -> None:
      del train_cfg, device
      self.env = FakeRunnerWrappedEnv(env)
      captured["runner_env"] = self.env

    def load(self, path, load_cfg=None, strict=True, map_location=None):
      del path, load_cfg, strict, map_location
      return {}

    def get_inference_policy(self, device="cpu"):
      del device
      return object()

  class FakeViewer:
    def __init__(self, env, policy, checkpoint_manager=None) -> None:
      del policy, checkpoint_manager
      captured["viewer_env"] = env

    def run(self) -> None:
      captured["viewer_ran"] = True

  monkeypatch.setattr(play_script, "load_env_cfg", lambda task_id, play=True: FakeEnvCfg())
  monkeypatch.setattr(play_script, "load_rl_cfg", lambda task_id: TestRlCfg())
  monkeypatch.setattr(play_script, "ManagerBasedRlEnv", lambda **kwargs: FakeRawEnv())
  monkeypatch.setattr(play_script, "RslRlVecEnvWrapper", FakeRslEnv)
  monkeypatch.setattr(play_script, "load_runner_cls", lambda task_id: FakeRunner)
  monkeypatch.setattr(play_script, "ViserPlayViewer", FakeViewer)

  cfg = play_script.PlayConfig(
    checkpoint_file=str(checkpoint),
    device="cpu",
    viewer="viser",
  )

  play_script.run_play("Mjlab-LatentRL-Flat-Unitree-G1", cfg)

  assert captured["viewer_env"] is captured["runner_env"]
  assert captured["viewer_ran"] is True
  assert captured["closed"] is True


def test_play_can_use_stochastic_trained_policy(monkeypatch, tmp_path: Path) -> None:
  from mjlab.scripts import play as play_script

  checkpoint = tmp_path / "model.pt"
  checkpoint.write_text("dummy")
  captured = {}

  @dataclass
  class TestRlCfg(RslRlBaseRunnerCfg):
    experiment_name: str = "test_play"
    clip_actions: float | None = None

  class FakeRawEnv:
    render_mode = None

    def close(self) -> None:
      captured["closed"] = True

  class FakeEnvCfg:
    commands = {}
    terminations = {}

  class FakeRslEnv:
    def __init__(self, env, clip_actions=None) -> None:
      del clip_actions
      self.env = env
      self.unwrapped = env

    def close(self) -> None:
      self.env.close()

  class FakeActor:
    def __call__(self, obs, stochastic_output=False):
      captured["policy_obs"] = obs
      captured["stochastic_output"] = stochastic_output
      return "sampled_action"

  class FakeAlg:
    def get_policy(self):
      return FakeActor()

  class FakeRunner:
    def __init__(self, env, train_cfg, device="cpu") -> None:
      del train_cfg, device
      self.env = env
      self.alg = FakeAlg()

    def load(self, path, load_cfg=None, strict=True, map_location=None):
      del path, load_cfg, strict, map_location
      return {}

    def get_inference_policy(self, device="cpu"):
      del device
      raise AssertionError("stochastic play should not use deterministic inference")

  class FakeViewer:
    def __init__(self, env, policy, checkpoint_manager=None) -> None:
      del env, checkpoint_manager
      captured["policy_result"] = policy("obs")

    def run(self) -> None:
      captured["viewer_ran"] = True

  monkeypatch.setattr(play_script, "load_env_cfg", lambda task_id, play=True: FakeEnvCfg())
  monkeypatch.setattr(play_script, "load_rl_cfg", lambda task_id: TestRlCfg())
  monkeypatch.setattr(play_script, "ManagerBasedRlEnv", lambda **kwargs: FakeRawEnv())
  monkeypatch.setattr(play_script, "RslRlVecEnvWrapper", FakeRslEnv)
  monkeypatch.setattr(play_script, "load_runner_cls", lambda task_id: FakeRunner)
  monkeypatch.setattr(play_script, "ViserPlayViewer", FakeViewer)

  cfg = play_script.PlayConfig(
    checkpoint_file=str(checkpoint),
    device="cpu",
    viewer="viser",
    stochastic_policy=True,
  )

  play_script.run_play("Mjlab-LatentRL-Flat-Unitree-G1", cfg)

  assert captured["policy_obs"] == "obs"
  assert captured["policy_result"] == "sampled_action"
  assert captured["stochastic_output"] is True
  assert captured["viewer_ran"] is True
  assert captured["closed"] is True
