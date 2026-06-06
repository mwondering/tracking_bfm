"""Tests for tracking runner adaptive sampling iteration hooks."""

from types import SimpleNamespace

import torch

from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class _FakeMotionCommand:
  def __init__(self) -> None:
    self.iterations: list[int] = []

  def begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    self.iterations.append(iteration)


class _FakeCommandManager:
  def __init__(self, motion: _FakeMotionCommand) -> None:
    self.motion = motion

  def get_term(self, name: str) -> _FakeMotionCommand:
    assert name == "motion"
    return self.motion


class _FakeEnv:
  def __init__(self, motion: _FakeMotionCommand) -> None:
    self.device = "cpu"
    self.max_episode_length = 100
    self.episode_length_buf = torch.zeros(2, dtype=torch.long)
    self.unwrapped = SimpleNamespace(command_manager=_FakeCommandManager(motion))

  def get_observations(self) -> torch.Tensor:
    return torch.zeros(2, 3)

  def step(self, actions: torch.Tensor):
    return (
      torch.zeros(2, 3),
      torch.zeros(2),
      torch.zeros(2, dtype=torch.bool),
      {},
    )


class _FakePolicy:
  output_std = torch.ones(1)


class _FakeAlg:
  learning_rate = 1.0e-3
  intrinsic_rewards = None
  rnd = SimpleNamespace(weight=None)

  def train_mode(self) -> None:
    pass

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(obs.shape[0], 1)

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    pass

  def compute_returns(self, obs: torch.Tensor) -> None:
    pass

  def update(self) -> dict[str, float]:
    return {}

  def get_policy(self) -> _FakePolicy:
    return _FakePolicy()


class _FakeLogger:
  writer = None
  log_dir = ""

  def init_logging_writer(self) -> None:
    pass

  def process_env_step(self, rewards, dones, extras, intrinsic_rewards) -> None:
    pass

  def log(self, **kwargs) -> None:
    pass


def test_tracking_runner_advances_adaptive_window_once_per_learning_iteration() -> None:
  motion = _FakeMotionCommand()
  runner = object.__new__(MotionTrackingOnPolicyRunner)
  runner.env = _FakeEnv(motion)
  runner.device = "cpu"
  runner.alg = _FakeAlg()
  runner.logger = _FakeLogger()
  runner.is_distributed = False
  runner.current_learning_iteration = 0
  runner.cfg = {
    "num_steps_per_env": 1,
    "check_for_nan": False,
    "algorithm": {"rnd_cfg": None},
    "save_interval": 100,
    "upload_model": False,
  }

  runner.learn(num_learning_iterations=3)

  assert motion.iterations == [0, 1, 2]


def test_tracking_runner_save_skips_onnx_export_when_upload_disabled(
  tmp_path, monkeypatch
) -> None:
  runner = object.__new__(MotionTrackingOnPolicyRunner)
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
  runner.alg = SimpleNamespace(
    save=lambda: {
      "actor_state_dict": {},
      "critic_state_dict": {},
      "optimizer_state_dict": {},
    }
  )
  runner.current_learning_iteration = 0
  runner.cfg = {"upload_model": False}
  runner.logger = SimpleNamespace(save_model=lambda *args, **kwargs: None)
  runner.registry_name = None

  export_calls = []

  def record_export(*args, **kwargs):
    export_calls.append((args, kwargs))

  monkeypatch.setattr(runner, "export_policy_to_onnx", record_export)

  checkpoint_path = tmp_path / "model_0.pt"
  runner.save(str(checkpoint_path))

  assert checkpoint_path.exists()
  assert export_calls == []
  assert not list(tmp_path.glob("*.onnx"))
