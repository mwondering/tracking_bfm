"""Tests for latent tracking deployment ONNX export helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
from torch import nn

from mjlab.deployment.latent_tracking_bfm_onnx_export import (
  LatentTrackingDeploymentOnnxModel,
  export_actor_decoder_model_to_onnx,
  resolve_latent_deploy_onnx_path,
)


class _ConstantActor(nn.Module):
  def __init__(self, obs_dim: int, latent: torch.Tensor) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.latent_value = latent

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _ConstantActorOnnx(self.obs_dim, self.latent_value)


class _ConstantActorOnnx(nn.Module):
  def __init__(self, obs_dim: int, latent: torch.Tensor) -> None:
    super().__init__()
    self.input_size = obs_dim
    self.latent_value = latent

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs[:, : self.latent_value.shape[0]] + self.latent_value.unsqueeze(0)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["latent_actions"]


class _DecoderContainer(nn.Module):
  def __init__(self, proprio_dim: int, latent_dim: int) -> None:
    super().__init__()
    self.latent_dim = latent_dim
    self.decoder = _LatentEchoDecoder(proprio_dim, latent_dim)


class _LatentEchoDecoder(nn.Module):
  def __init__(self, proprio_dim: int, latent_dim: int) -> None:
    super().__init__()
    self.proprio_dim = proprio_dim
    self.latent_dim = latent_dim

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _LatentEchoDecoderOnnx(self.proprio_dim, self.latent_dim)


class _LatentEchoDecoderOnnx(nn.Module):
  def __init__(self, proprio_dim: int, latent_dim: int) -> None:
    super().__init__()
    self.input_size = proprio_dim + latent_dim
    self.latent_dim = latent_dim

  def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
    return decoder_input[:, -self.latent_dim :]

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["decoder_input"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


def test_resolve_latent_deploy_onnx_path_default_name() -> None:
  path = resolve_latent_deploy_onnx_path("/tmp/exp/model_7000.pt")

  assert path == Path("/tmp/exp/deploy_model_7000.onnx")


def test_combined_model_clamps_latent_before_decoding() -> None:
  actor = _ConstantActor(obs_dim=5, latent=torch.tensor([2.0, -3.0, 0.25]))
  decoder = _DecoderContainer(proprio_dim=4, latent_dim=3)
  model = LatentTrackingDeploymentOnnxModel(
    actor=actor,
    decoder=decoder,
    latent_action_clip=0.5,
  )

  actions = model(torch.zeros(2, 5), torch.randn(2, 4))

  expected = torch.tensor([[0.5, -0.5, 0.25], [0.5, -0.5, 0.25]])
  torch.testing.assert_close(actions, expected)
  assert model.input_names == ["obs", "proprio"]
  assert model.output_names == ["actions"]
  assert model.get_dummy_inputs()[0].shape == (1, 5)
  assert model.get_dummy_inputs()[1].shape == (1, 4)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_actor_decoder_model_to_onnx_roundtrip_and_metadata() -> None:
  ort = pytest.importorskip("onnxruntime")
  actor = _ConstantActor(obs_dim=5, latent=torch.tensor([2.0, -3.0, 0.25]))
  decoder = _DecoderContainer(proprio_dim=4, latent_dim=3)

  with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "model_7000.pt"
    checkpoint_path.write_text("dummy")
    decoder_checkpoint_path = Path(tmpdir) / "decoder_100.pt"
    decoder_checkpoint_path.write_text("dummy")

    onnx_path = export_actor_decoder_model_to_onnx(
      actor=actor,
      decoder=decoder,
      checkpoint_path=checkpoint_path,
      decoder_checkpoint_path=decoder_checkpoint_path,
      task_id="Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage",
      obs_group="actor",
      proprio_obs_group="proprio_actor",
      latent_action_clip=0.5,
      robot_name="g1",
    )

    assert onnx_path == checkpoint_path.parent / "deploy_model_7000.onnx"
    assert onnx_path.exists()
    onnx.checker.check_model(str(onnx_path))

    obs = torch.zeros(1, 5)
    proprio = torch.randn(1, 4)
    expected = torch.tensor([[0.5, -0.5, 0.25]])
    sess = ort.InferenceSession(str(onnx_path))
    [actual] = sess.run(None, {"obs": obs.numpy(), "proprio": proprio.numpy()})
    torch.testing.assert_close(torch.from_numpy(actual), expected)

    metadata_props = {
      prop.key: prop.value for prop in onnx.load(str(onnx_path)).metadata_props
    }
    assert metadata_props == {
      "task_id": "Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage",
      "checkpoint_family": "latent_tracking",
      "decoder_checkpoint": str(decoder_checkpoint_path),
      "obs_group": "actor",
      "proprio_obs_group": "proprio_actor",
      "robot_name": "g1",
    }
    assert sess.get_inputs()[0].name == "obs"
    assert sess.get_inputs()[0].shape == [1, 5]
    assert sess.get_inputs()[1].name == "proprio"
    assert sess.get_inputs()[1].shape == [1, 4]
    assert sess.get_outputs()[0].shape == [1, 3]
