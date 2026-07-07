from __future__ import annotations

from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
  import tomli as tomllib


def test_onnx_export_console_scripts_are_registered() -> None:
  pyproject = tomllib.loads(Path("pyproject.toml").read_text())
  scripts = pyproject["project"]["scripts"]

  assert (
    scripts["export-tracking-bfm-onnx"]
    == "mjlab.scripts.export_tracking_bfm_onnx:main"
  )
  assert (
    scripts["export-latent-tracking-bfm-onnx"]
    == "mjlab.scripts.export_latent_tracking_bfm_onnx:main"
  )


def test_latent_analysis_console_script_is_registered() -> None:
  pyproject = tomllib.loads(Path("pyproject.toml").read_text())
  scripts = pyproject["project"]["scripts"]

  assert scripts["analyze-latent-space"] == "mjlab.scripts.analyze_latent_space:main"


def test_generate_motion_manifest_console_script_is_registered() -> None:
  pyproject = tomllib.loads(Path("pyproject.toml").read_text())
  scripts = pyproject["project"]["scripts"]

  assert (
    scripts["generate-motion-manifest"]
    == "mjlab.scripts.generate_motion_manifest:main"
  )


def test_adaptive_bin_viewer_console_script_is_registered() -> None:
  pyproject = tomllib.loads(Path("pyproject.toml").read_text())
  scripts = pyproject["project"]["scripts"]

  assert scripts["adaptive-bin-viewer"] == "mjlab.tasks.tracking.viewer.server:main"
