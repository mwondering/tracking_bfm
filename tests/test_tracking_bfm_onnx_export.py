"""Tests for tracking_bfm deployment ONNX export helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.deployment.tracking_bfm_onnx_export import (
  _apply_distillation_student_obs_overrides,
  _apply_motion_source,
  detect_checkpoint_family,
  export_actor_model_to_onnx,
  resolve_deploy_onnx_path,
)


def _make_actor(
  obs_dim: int = 8, output_dim: int = 4, obs_normalization: bool = True
) -> MLPModel:
  obs = TensorDict({"actor": torch.zeros(1, obs_dim)})
  return MLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=output_dim,
    hidden_dims=[32, 32],
    activation="elu",
    obs_normalization=obs_normalization,
  )


def _train_normalizer(actor: MLPModel, n_batches: int = 50, batch_size: int = 64) -> None:
  actor.train()
  for _ in range(n_batches):
    obs = TensorDict({"actor": torch.randn(batch_size, actor.obs_dim) * 5 + 3})
    actor.update_normalization(obs)
  actor.eval()


def _model_output(actor: MLPModel, x_flat: torch.Tensor) -> torch.Tensor:
  obs = TensorDict({"actor": x_flat})
  with torch.no_grad():
    return actor(obs)


def test_detect_checkpoint_family_tracking() -> None:
  family = detect_checkpoint_family({"actor_state_dict": {"weight": torch.tensor([1.0])}})
  assert family == "tracking"


def test_detect_checkpoint_family_distillation() -> None:
  family = detect_checkpoint_family(
    {"policy_state_dict": {"weight": torch.tensor([1.0])}}
  )
  assert family == "distillation"


def test_detect_checkpoint_family_rejects_unknown() -> None:
  with pytest.raises(ValueError, match="Unsupported checkpoint format"):
    detect_checkpoint_family({"optimizer_state_dict": {}})


def test_resolve_deploy_onnx_path_default_name() -> None:
  path = resolve_deploy_onnx_path("/tmp/exp/model_5000.pt")
  assert path == Path("/tmp/exp/deploy_model_5000.onnx")


def test_resolve_deploy_onnx_path_custom_name() -> None:
  path = resolve_deploy_onnx_path("/tmp/exp/model_5000.pt", output_name="custom_name")
  assert path == Path("/tmp/exp/custom_name.onnx")


def test_apply_motion_source_sets_multi_motion_path() -> None:
  class MotionCfg:
    motion_path = ""
    motion_file = ""

  env_cfg = type("EnvCfg", (), {"commands": {"motion": MotionCfg()}})()

  _apply_motion_source(env_cfg, motion_path="/tmp/motions", motion_file=None)

  assert env_cfg.commands["motion"].motion_path == "/tmp/motions"
  assert env_cfg.commands["motion"].motion_file == ""


def test_apply_motion_source_rejects_both_motion_sources() -> None:
  env_cfg = type("EnvCfg", (), {"commands": {"motion": object()}})()

  with pytest.raises(ValueError, match="Provide only one of"):
    _apply_motion_source(env_cfg, motion_path="/tmp/motions", motion_file="/tmp/a.npz")


def test_apply_distillation_student_obs_overrides_sets_command_and_robot_history() -> None:
  class Term:
    def __init__(self) -> None:
      self.params = {}
      self.history_length = 0

  class Motion:
    history_steps = 0
    future_steps = 1

  terms = {
    "ee_pose": Term(),
    "base_lin_vel_b": Term(),
    "base_ang_vel_b": Term(),
    "anchor_height_w": Term(),
    "projected_gravity": Term(),
    "base_ang_vel": Term(),
    "joint_pos": Term(),
    "joint_vel": Term(),
    "actions": Term(),
  }
  env_cfg = type(
    "EnvCfg",
    (),
    {
      "commands": {"motion": Motion()},
      "observations": {
        "student_actor": type("Group", (), {"terms": terms})(),
      },
    },
  )()

  _apply_distillation_student_obs_overrides(
    env_cfg,
    student_history_steps=3,
    student_future_steps=4,
    student_robot_history_steps=20,
  )

  assert env_cfg.commands["motion"].history_steps == 3
  assert env_cfg.commands["motion"].future_steps == 4
  for name in ("ee_pose", "base_lin_vel_b", "base_ang_vel_b", "anchor_height_w"):
    assert terms[name].params["history_steps"] == 3
    assert terms[name].params["future_steps"] == 4
  for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
    assert terms[name].history_length == 20


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_actor_model_to_onnx_roundtrip_and_minimal_metadata() -> None:
  ort = pytest.importorskip("onnxruntime")
  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)
  x = torch.randn(1, actor.obs_dim)
  expected = _model_output(actor, x)

  with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "model_1234.pt"
    checkpoint_path.write_text("dummy")

    onnx_path = export_actor_model_to_onnx(
      actor=actor,
      checkpoint_path=checkpoint_path,
      task_id="Mjlab-Trackingbfm-Flat-Unitree-G1",
      checkpoint_family="tracking",
      obs_group="actor",
      robot_name="g1",
    )

    assert onnx_path == checkpoint_path.parent / "deploy_model_1234.onnx"
    assert onnx_path.exists()
    onnx.checker.check_model(str(onnx_path))

    sess = ort.InferenceSession(str(onnx_path))
    [actual] = sess.run(None, {"obs": x.numpy()})
    torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=0)

    metadata_props = {
      prop.key: prop.value for prop in onnx.load(str(onnx_path)).metadata_props
    }
    assert metadata_props == {
      "task_id": "Mjlab-Trackingbfm-Flat-Unitree-G1",
      "obs_group": "actor",
      "checkpoint_family": "tracking",
      "robot_name": "g1",
    }

    assert sess.get_inputs()[0].shape == [1, actor.obs_dim]
    assert sess.get_outputs()[0].shape == [1, expected.shape[1]]
