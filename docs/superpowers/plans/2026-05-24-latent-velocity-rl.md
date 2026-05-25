# Latent Velocity RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second-stage direct latent RL walking task for Unitree G1 that trains a high-level PPO policy in the learned latent space and decodes latents through the frozen first-stage latent distillation decoder.

**Architecture:** The new `latentvelocity` task package imports the flat G1 velocity env factory as its baseline, then adds only the `proprio_actor` observation group and latent RL runner/config in its own directory. RSL-RL sees a 64D latent action space; a custom VecEnv wrapper clips latent actions to `[-6, 6]`, decodes them with a frozen decoder checkpoint using `proprio_actor` observations, then steps the original velocity env with decoded joint actions.

**Tech Stack:** Python, PyTorch, TensorDict, RSL-RL PPO, `ManagerBasedRlEnv`, `RslRlVecEnvWrapper`, existing latent distillation decoder.

---

## Scope And Boundaries

**Must change:**
- Add a new task id: `Mjlab-LatentRL-Flat-Unitree-G1`.
- Add a new isolated task package under `src/mjlab/tasks/latentvelocity/`.
- Add a latent velocity env variant in the new package that imports and extends the flat G1 velocity env with a `proprio_actor` observation group for the frozen decoder.
- Add a latent RL runner and VecEnv wrapper under `src/mjlab/tasks/latentvelocity/rl/`.
- Add latent RL config under `src/mjlab/tasks/latentvelocity/config/g1/rl_cfg.py`.
- Add a training script `scripts/train_latent_velocity_rl.sh`.
- Add focused tests for registration, reward parity, wrapper decode behavior, checkpoint validation, and script defaults.

**Must not change:**
- Do not modify velocity reward functions or weights.
- Do not modify velocity command sampling, termination rules, curriculum, or randomization.
- Do not modify `Mjlab-Velocity-Flat-Unitree-G1` baseline behavior.
- Do not add latent RL classes, wrappers, configs, or tests under `src/mjlab/tasks/velocity/`.
- Do not use tracking motion commands or tracking rewards.
- Do not use the first-stage teacher encoder during second-stage RL.
- Do not train or update the decoder during second-stage RL.

## File Structure

- Create `src/mjlab/tasks/latentvelocity/__init__.py`
  - Empty package marker. `mjlab.tasks` auto-imports task packages.

- Create `src/mjlab/tasks/latentvelocity/config/__init__.py`
  - Empty package marker.

- Create `src/mjlab/tasks/latentvelocity/config/g1/env_cfgs.py`
  - Add `unitree_g1_flat_latent_rl_env_cfg(play: bool = False)`.
  - It imports `unitree_g1_flat_env_cfg(play=play)` from `mjlab.tasks.velocity.config.g1.env_cfgs` and adds `proprio_actor` observations matching first-stage latent distillation.

- Create `src/mjlab/tasks/latentvelocity/config/g1/rl_cfg.py`
  - Add `LatentVelocityPpoRunnerCfg`, extending `RslRlOnPolicyRunnerCfg` with latent decoder fields.
  - Add `unitree_g1_latent_velocity_ppo_runner_cfg()`.

- Create `src/mjlab/tasks/latentvelocity/config/g1/__init__.py`
  - Register `Mjlab-LatentRL-Flat-Unitree-G1` with `LatentVelocityOnPolicyRunner`.

- Create `src/mjlab/tasks/latentvelocity/rl/latent_decoder_wrapper.py`
  - Owns frozen decoder checkpoint loading, latent clipping, decode, and diagnostics.

- Create `src/mjlab/tasks/latentvelocity/rl/runner.py`
  - Add `LatentVelocityOnPolicyRunner`.
  - It can subclass `mjlab.tasks.velocity.rl.VelocityOnPolicyRunner` only for save/export behavior, but all latent-specific logic lives in `latentvelocity`.

- Create `src/mjlab/tasks/latentvelocity/rl/__init__.py`
  - Export `LatentVelocityOnPolicyRunner` and `LatentDecoderVecEnvWrapper`.

- Create `scripts/train_latent_velocity_rl.sh`
  - Launches the new task with direct latent PPO and `latent_action_clip=6.0`.

- Add tests:
  - `tests/test_latent_velocity_task.py`
  - `tests/test_latent_velocity_wrapper.py`
  - `tests/test_train_latent_velocity_rl_script.py`

---

### Task 1: Register A Latent Velocity Env Variant

**Files:**
- Create: `src/mjlab/tasks/latentvelocity/__init__.py`
- Create: `src/mjlab/tasks/latentvelocity/config/__init__.py`
- Create: `src/mjlab/tasks/latentvelocity/config/g1/env_cfgs.py`
- Test: `tests/test_latent_velocity_task.py`

- [ ] **Step 1: Write the failing registration/env test**

Create `tests/test_latent_velocity_task.py`:

```python
"""Tests for latent velocity RL task configuration."""

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.latentvelocity.rl import LatentVelocityOnPolicyRunner


def test_latent_velocity_task_is_registered() -> None:
  task_id = "Mjlab-LatentRL-Flat-Unitree-G1"

  assert task_id in list_tasks()
  assert load_runner_cls(task_id) is LatentVelocityOnPolicyRunner
  assert load_rl_cfg(task_id).latent_dim == 64
  assert load_rl_cfg(task_id).latent_action_clip == 6.0


def test_latent_velocity_env_keeps_velocity_rewards_unchanged() -> None:
  baseline = unitree_g1_flat_env_cfg()
  cfg = load_env_cfg("Mjlab-LatentRL-Flat-Unitree-G1")

  assert set(cfg.rewards.keys()) == set(baseline.rewards.keys())
  for name, reward in baseline.rewards.items():
    latent_reward = cfg.rewards[name]
    assert latent_reward.func is reward.func
    assert latent_reward.weight == reward.weight
    assert latent_reward.params == reward.params

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_latent_velocity_task.py -q
```

Expected: import or assertion failure because the new task and runner do not exist yet.

- [ ] **Step 3: Add package markers and the env factory**

Create `src/mjlab/tasks/latentvelocity/__init__.py`:

```python
"""Latent velocity RL tasks."""
```

Create `src/mjlab/tasks/latentvelocity/config/__init__.py`:

```python
"""Latent velocity task configurations."""
```

Create `src/mjlab/tasks/latentvelocity/config/g1/env_cfgs.py`:

```python
"""Unitree G1 latent velocity RL environment configurations."""

from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.distillation.mdp.observations import build_proprio_actor_terms
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.envs import ManagerBasedRlEnvCfg

def unitree_g1_flat_latent_rl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat velocity config for latent-space RL."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.observations["proprio_actor"] = ObservationGroupCfg(
    terms=build_proprio_actor_terms(history_steps=0),
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg
```

- [ ] **Step 4: Run the task test again**

Run:

```bash
uv run pytest tests/test_latent_velocity_task.py -q
```

Expected: still fails because RL cfg and task registration are not added yet.

---

### Task 2: Add Latent RL Config And Task Registration

**Files:**
- Create: `src/mjlab/tasks/latentvelocity/config/g1/rl_cfg.py`
- Create: `src/mjlab/tasks/latentvelocity/config/g1/__init__.py`
- Create: `src/mjlab/tasks/latentvelocity/rl/__init__.py`
- Create: `src/mjlab/tasks/latentvelocity/rl/runner.py`
- Test: `tests/test_latent_velocity_task.py`

- [ ] **Step 1: Add latent runner config fields**

Create `src/mjlab/tasks/latentvelocity/config/g1/rl_cfg.py`:

```python
"""RL configuration for Unitree G1 latent velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class LatentVelocityPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  latent_decoder_checkpoint_path: str = ""
  latent_dim: int = 64
  latent_action_clip: float = 6.0
  proprio_obs_group: str = "proprio_actor"


def unitree_g1_latent_velocity_ppo_runner_cfg() -> LatentVelocityPpoRunnerCfg:
  """Create PPO runner configuration for Unitree G1 latent velocity RL."""
  cfg = LatentVelocityPpoRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_latent_velocity",
    run_name="latent_rl_flat_g1",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
    clip_actions=None,
  )
  return cfg
```

`clip_actions=None` is intentional. Latent clipping belongs in the latent decoder wrapper, not in the base joint-action wrapper.

- [ ] **Step 2: Add a temporary runner stub**

Create `src/mjlab/tasks/latentvelocity/rl/runner.py`:

```python
"""Runner for latent velocity RL tasks."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class LatentVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner for direct latent velocity RL."""
```

Create `src/mjlab/tasks/latentvelocity/rl/__init__.py`:

```python
from mjlab.tasks.latentvelocity.rl.runner import (
  LatentVelocityOnPolicyRunner as LatentVelocityOnPolicyRunner,
)
```

- [ ] **Step 3: Register the task**

Create `src/mjlab/tasks/latentvelocity/config/g1/__init__.py`:

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.latentvelocity.rl import LatentVelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_latent_rl_env_cfg,
)
from .rl_cfg import (
  unitree_g1_latent_velocity_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="Mjlab-LatentRL-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_latent_rl_env_cfg(),
  play_env_cfg=unitree_g1_flat_latent_rl_env_cfg(play=True),
  rl_cfg=unitree_g1_latent_velocity_ppo_runner_cfg(),
  runner_cls=LatentVelocityOnPolicyRunner,
)
```

- [ ] **Step 4: Verify registration tests pass**

Run:

```bash
uv run pytest tests/test_latent_velocity_task.py -q
```

Expected: PASS.

---

### Task 3: Implement Frozen Decoder VecEnv Wrapper

**Files:**
- Create: `src/mjlab/tasks/latentvelocity/rl/latent_decoder_wrapper.py`
- Modify: `src/mjlab/tasks/latentvelocity/rl/__init__.py`
- Test: `tests/test_latent_velocity_wrapper.py`

- [ ] **Step 1: Write failing wrapper tests**

Create `tests/test_latent_velocity_wrapper.py`:

```python
"""Tests for latent decoder VecEnv wrapper."""

import torch
from tensordict import TensorDict

from mjlab.tasks.latentvelocity.rl.latent_decoder_wrapper import LatentDecoderVecEnvWrapper


class _DummyBaseEnv:
  def __init__(self):
    self.num_envs = 2
    self.num_actions = 29
    self.device = torch.device("cpu")
    self.max_episode_length = 100
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self.cfg = object()
    self.last_actions = None

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    return TensorDict(
      {
        "actor": torch.zeros(self.num_envs, 4),
        "critic": torch.zeros(self.num_envs, 5),
        "proprio_actor": torch.ones(self.num_envs, 3),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions):
    self.last_actions = actions.detach().clone()
    return (
      self.get_observations(),
      torch.ones(self.num_envs),
      torch.zeros(self.num_envs, dtype=torch.long),
      {},
    )

  def reset(self):
    return self.get_observations(), {}

  def close(self):
    pass


class _DummyDecoder(torch.nn.Module):
  decoder_obs_group = "proprio_actor"
  latent_dim = 2

  def decode(self, obs, z):
    proprio = obs["proprio_actor"]
    return torch.cat([proprio, z], dim=-1)


def test_latent_wrapper_exposes_latent_action_dim_and_steps_decoded_actions() -> None:
  base_env = _DummyBaseEnv()
  wrapper = LatentDecoderVecEnvWrapper(
    base_env,
    decoder=_DummyDecoder(),
    latent_dim=2,
    proprio_obs_group="proprio_actor",
    latent_action_clip=6.0,
  )

  obs = wrapper.get_observations()
  actions = torch.tensor([[10.0, -10.0], [0.5, -0.5]])
  next_obs, rewards, dones, extras = wrapper.step(actions)

  assert wrapper.num_actions == 2
  assert obs.batch_size == torch.Size([2])
  assert next_obs.batch_size == torch.Size([2])
  assert rewards.tolist() == [1.0, 1.0]
  assert dones.tolist() == [0, 0]
  torch.testing.assert_close(
    base_env.last_actions,
    torch.tensor([[1.0, 1.0, 1.0, 6.0, -6.0], [1.0, 1.0, 1.0, 0.5, -0.5]]),
  )
  assert extras["log"]["latent/norm_mean"] > 0.0
  assert extras["log"]["latent/abs_max"] == 6.0


def test_latent_wrapper_freezes_decoder_parameters() -> None:
  decoder = torch.nn.Linear(3, 2)
  base_env = _DummyBaseEnv()

  wrapper = LatentDecoderVecEnvWrapper(
    base_env,
    decoder=decoder,
    latent_dim=2,
    proprio_obs_group="proprio_actor",
    latent_action_clip=6.0,
  )

  assert not wrapper.decoder.training
  assert all(not p.requires_grad for p in wrapper.decoder.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_latent_velocity_wrapper.py -q
```

Expected: import failure because `latent_decoder_wrapper.py` does not exist.

- [ ] **Step 3: Implement wrapper**

Create `src/mjlab/tasks/latentvelocity/rl/latent_decoder_wrapper.py`:

```python
"""Latent-action VecEnv wrapper for frozen decoder policies."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class LatentDecoderVecEnvWrapper(VecEnv):
  """Expose latent actions to RSL-RL and decode them to joint actions."""

  def __init__(
    self,
    env: VecEnv,
    decoder: torch.nn.Module,
    latent_dim: int,
    proprio_obs_group: str,
    latent_action_clip: float = 6.0,
  ) -> None:
    self.env = env
    self.decoder = decoder
    self.latent_dim = int(latent_dim)
    self.proprio_obs_group = proprio_obs_group
    self.latent_action_clip = float(latent_action_clip)
    self.decoder.eval()
    for param in self.decoder.parameters():
      param.requires_grad_(False)

    self.num_envs = env.num_envs
    self.num_actions = self.latent_dim
    self.device = env.device
    self.max_episode_length = env.max_episode_length
    self._last_obs = env.get_observations()

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.env.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.env.seed(seed)

  def get_observations(self) -> TensorDict:
    self._last_obs = self.env.get_observations()
    return self._last_obs

  def reset(self):
    obs, extras = self.env.reset()
    self._last_obs = obs
    return obs, extras

  @torch.no_grad()
  def step(self, latent_actions: torch.Tensor):
    latent_actions = torch.clamp(
      latent_actions,
      min=-self.latent_action_clip,
      max=self.latent_action_clip,
    )
    decoder_obs = TensorDict(
      {self.proprio_obs_group: self._last_obs[self.proprio_obs_group]},
      batch_size=list(self._last_obs.batch_size),
      device=self._last_obs.device,
    )
    joint_actions = self.decoder.decode(decoder_obs, latent_actions)
    obs, rewards, dones, extras = self.env.step(joint_actions.to(self.env.device))
    self._last_obs = obs
    extras = dict(extras)
    log_extras = dict(extras.get("log", {}))
    log_extras["latent/norm_mean"] = latent_actions.norm(dim=-1).mean()
    log_extras["latent/abs_max"] = latent_actions.abs().max()
    log_extras["latent/decoded_action_norm_mean"] = joint_actions.norm(dim=-1).mean()
    extras["log"] = log_extras
    return obs, rewards, dones, extras

  def close(self) -> None:
    self.env.close()
```

- [ ] **Step 4: Export wrapper**

In `src/mjlab/tasks/latentvelocity/rl/__init__.py`, add:

```python
from mjlab.tasks.latentvelocity.rl.latent_decoder_wrapper import (
  LatentDecoderVecEnvWrapper as LatentDecoderVecEnvWrapper,
)
```

- [ ] **Step 5: Run wrapper tests**

Run:

```bash
uv run pytest tests/test_latent_velocity_wrapper.py -q
```

Expected: PASS.

---

### Task 4: Load Frozen Decoder Checkpoint In Runner

**Files:**
- Modify: `src/mjlab/tasks/latentvelocity/rl/runner.py`
- Test: `tests/test_latent_velocity_wrapper.py`

- [ ] **Step 1: Add failing checkpoint validation tests**

Append to `tests/test_latent_velocity_wrapper.py`:

```python
import pytest

from mjlab.tasks.latentvelocity.rl.runner import LatentVelocityOnPolicyRunner


def test_latent_runner_requires_decoder_checkpoint() -> None:
  base_env = _DummyBaseEnv()
  cfg = {
    "actor": {
      "hidden_dims": [16],
      "activation": "elu",
      "obs_normalization": False,
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    },
    "critic": {"hidden_dims": [16], "activation": "elu", "obs_normalization": False},
    "algorithm": {
      "class_name": "PPO",
      "num_learning_epochs": 1,
      "num_mini_batches": 1,
      "learning_rate": 1.0e-3,
      "schedule": "fixed",
      "gamma": 0.99,
      "lam": 0.95,
      "entropy_coef": 0.0,
      "desired_kl": 0.01,
      "max_grad_norm": 1.0,
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": True,
      "clip_param": 0.2,
    },
    "obs_groups": {"actor": ("actor",), "critic": ("critic",)},
    "num_steps_per_env": 2,
    "max_iterations": 1,
    "save_interval": 50,
    "experiment_name": "test",
    "logger": "tensorboard",
    "upload_model": False,
    "clip_actions": None,
    "latent_decoder_checkpoint_path": "",
    "latent_dim": 2,
    "latent_action_clip": 6.0,
    "proprio_obs_group": "proprio_actor",
  }

  with pytest.raises(ValueError, match="latent_decoder_checkpoint_path"):
    LatentVelocityOnPolicyRunner(base_env, cfg, log_dir=None, device="cpu")
```

- [ ] **Step 2: Run test to verify it fails for the current stub**

Run:

```bash
uv run pytest tests/test_latent_velocity_wrapper.py::test_latent_runner_requires_decoder_checkpoint -q
```

Expected: FAIL because the stub does not validate the checkpoint path.

- [ ] **Step 3: Implement runner checkpoint loading**

In `src/mjlab/tasks/latentvelocity/rl/runner.py`, add imports:

```python
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.tasks.distillation.rl.models import build_latent_student_model
from mjlab.tasks.latentvelocity.rl.latent_decoder_wrapper import LatentDecoderVecEnvWrapper
```

Replace the stub with:

```python
class LatentVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner that trains latent actions through a frozen decoder."""

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    decoder = self._load_decoder(env, train_cfg, device)
    latent_env = LatentDecoderVecEnvWrapper(
      env,
      decoder=decoder,
      latent_dim=int(train_cfg["latent_dim"]),
      proprio_obs_group=train_cfg.get("proprio_obs_group", "proprio_actor"),
      latent_action_clip=float(train_cfg["latent_action_clip"]),
    )
    super().__init__(latent_env, train_cfg, log_dir, device)

  @staticmethod
  def _load_decoder(env, train_cfg: dict, device: str) -> torch.nn.Module:
    checkpoint_path = train_cfg.get("latent_decoder_checkpoint_path", "")
    if not checkpoint_path:
      raise ValueError("latent_decoder_checkpoint_path must be provided for latent velocity RL.")
    if not Path(checkpoint_path).exists():
      raise FileNotFoundError(f"latent decoder checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "latent":
      raise ValueError("latent decoder checkpoint must be a latent distillation checkpoint.")
    if "policy_state_dict" not in checkpoint:
      raise ValueError("latent decoder checkpoint is missing policy_state_dict.")

    obs = env.get_observations().to(device)
    proprio_obs_group = train_cfg.get("proprio_obs_group", "proprio_actor")
    model_obs = TensorDict(
      {
        "teacher_actor": torch.zeros(
          obs.batch_size[0],
          checkpoint["policy_state_dict"]["encoder.obs_normalizer._mean"].shape[-1],
          device=device,
        ),
        proprio_obs_group: obs[proprio_obs_group],
      },
      batch_size=list(obs.batch_size),
      device=device,
    )
    model = build_latent_student_model(
      obs=model_obs,
      encoder_obs_group="teacher_actor",
      decoder_obs_group=proprio_obs_group,
      action_dim=env.num_actions,
      latent_dim=int(train_cfg["latent_dim"]),
      encoder_hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      decoder_hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
    model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model
```

- [ ] **Step 4: Run checkpoint validation test**

Run:

```bash
uv run pytest tests/test_latent_velocity_wrapper.py::test_latent_runner_requires_decoder_checkpoint -q
```

Expected: PASS.

- [ ] **Step 5: Run wrapper test file**

Run:

```bash
uv run pytest tests/test_latent_velocity_wrapper.py -q
```

Expected: PASS.

---

### Task 5: Add Training Script

**Files:**
- Create: `scripts/train_latent_velocity_rl.sh`
- Test: `tests/test_train_latent_velocity_rl_script.py`

- [ ] **Step 1: Write failing script test**

Create `tests/test_train_latent_velocity_rl_script.py`:

```python
"""Tests for latent velocity RL training script."""

import os
import subprocess
from pathlib import Path


def test_train_latent_velocity_rl_script_defaults() -> None:
  script = Path("scripts/train_latent_velocity_rl.sh")
  assert script.exists()

  env = os.environ.copy()
  env["LATENT_DECODER_CKPT"] = "/tmp/latent_decoder.pt"
  env["NUM_ENVS"] = "16"
  env["MAX_ITERATIONS"] = "3"
  result = subprocess.run(
    ["bash", "-n", str(script)],
    env=env,
    check=True,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0
  text = script.read_text()
  assert "Mjlab-LatentRL-Flat-Unitree-G1" in text
  assert "--agent.latent_decoder_checkpoint_path" in text
  assert "--agent.latent_action_clip 6.0" in text
  assert "--agent.clip_actions None" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_train_latent_velocity_rl_script.py -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Create script**

Create `scripts/train_latent_velocity_rl.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Mjlab-LatentRL-Flat-Unitree-G1}"
LATENT_DECODER_CKPT="${LATENT_DECODER_CKPT:-}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUN_NAME="${RUN_NAME:-latent_rl_flat_g1}"

if [[ -z "$LATENT_DECODER_CKPT" ]]; then
  echo "LATENT_DECODER_CKPT must point to a latent distillation checkpoint." >&2
  exit 1
fi

uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.latent_decoder_checkpoint_path "$LATENT_DECODER_CKPT" \
  --agent.latent_dim 64 \
  --agent.latent_action_clip 6.0 \
  --agent.experiment_name g1_latent_velocity \
  --agent.run_name "$RUN_NAME" \
  --agent.max_iterations "$MAX_ITERATIONS" \
  --agent.num_steps_per_env 24 \
  --agent.upload-model False \
  --debug False
```

- [ ] **Step 4: Run script tests and shell syntax check**

Run:

```bash
uv run pytest tests/test_train_latent_velocity_rl_script.py -q
bash -n scripts/train_latent_velocity_rl.sh
```

Expected: PASS and exit 0.

---

### Task 6: Add Smoke Tests For CLI And Registry

**Files:**
- Modify: `tests/test_latent_velocity_task.py`

- [ ] **Step 1: Add CLI help smoke test**

Append to `tests/test_latent_velocity_task.py`:

```python
import subprocess


def test_latent_velocity_train_help_exposes_decoder_flags() -> None:
  result = subprocess.run(
    ["uv", "run", "train", "Mjlab-LatentRL-Flat-Unitree-G1", "--help"],
    check=True,
    capture_output=True,
    text=True,
  )

  assert "--agent.latent-decoder-checkpoint-path" in result.stdout
  assert "--agent.latent-action-clip" in result.stdout
```

- [ ] **Step 2: Run the smoke test**

Run:

```bash
uv run pytest tests/test_latent_velocity_task.py::test_latent_velocity_train_help_exposes_decoder_flags -q
```

Expected: PASS.

- [ ] **Step 3: Run all related tests**

Run:

```bash
uv run pytest tests/test_latent_velocity_task.py tests/test_latent_velocity_wrapper.py tests/test_train_latent_velocity_rl_script.py tests/test_velocity_task.py -q
```

Expected: PASS.

---

### Task 7: Manual Training Dry Run

**Files:**
- No code changes.

- [ ] **Step 1: Run help command**

Run:

```bash
uv run train Mjlab-LatentRL-Flat-Unitree-G1 --help
```

Expected: help includes:

```text
--agent.latent-decoder-checkpoint-path
--agent.latent-action-clip
--agent.latent-dim
```

- [ ] **Step 2: Run a short debug train**

Use a real latent distillation checkpoint:

```bash
LATENT_DECODER_CKPT=/absolute/path/to/model_XXXX.pt \
NUM_ENVS=64 \
MAX_ITERATIONS=2 \
bash scripts/train_latent_velocity_rl.sh
```

Expected:
- Training starts without checkpoint shape mismatch.
- RSL-RL action std is reported for 64D latent policy.
- Logs contain latent diagnostics from `extras`:
  - `latent/norm_mean`
  - `latent/abs_max`
  - `latent/decoded_action_norm_mean`
- No decoder parameters are optimized.

- [ ] **Step 3: Compare against scratch baseline**

Run the scratch baseline with matching env count, steps, iterations, and seeds:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 64 \
  --agent.max_iterations 2 \
  --agent.num_steps_per_env 24 \
  --agent.experiment_name g1_velocity_baseline_debug \
  --agent.run_name scratch_debug \
  --agent.upload-model False \
  --debug True
```

Expected: Both tasks train in the same velocity env family and report comparable reward terms.

---

## Evaluation Checklist

Track these metrics for the real comparison:

- `Episode/reward` or equivalent total reward.
- Per-reward components from velocity task:
  - `track_linear_velocity`
  - `track_angular_velocity`
  - `upright`
  - `pose`
  - `foot_slip`
  - `foot_clearance`
- Termination rates:
  - `fell_over`
  - `time_out`
- Latent diagnostics:
  - `latent/norm_mean`
  - `latent/abs_max`
  - `latent/decoded_action_norm_mean`
- Sample efficiency:
  - environment steps to reach the same reward threshold.
- Wall-clock efficiency:
  - minutes to reach the same reward threshold.

The primary comparison is:

```text
Mjlab-Velocity-Flat-Unitree-G1
vs.
Mjlab-LatentRL-Flat-Unitree-G1
```

Both use the same walking reward. The only intended difference is the policy action interface: direct joint action versus direct latent action through a frozen decoder.

## Self-Review

- No tracking task or tracking reward is part of this plan.
- The second-stage policy does not use the first-stage teacher encoder.
- The decoder is loaded from latent distillation and frozen.
- Latent action clipping is `[-6, 6]` through `latent_action_clip`, not base `clip_actions`.
- Velocity reward, command, termination, randomization, and curriculum are preserved from the flat G1 velocity task.
