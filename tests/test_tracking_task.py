"""Tests specific to motion tracking tasks."""

import math

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a single- or multi-motion command config."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg)), (
      f"Task {task_id} motion command is not a supported motion command cfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_g1_tracking_penalizes_waist_action_rate(
  g1_tracking_task_ids: list[str],
) -> None:
  """G1 tracking tasks should include a waist-only action-rate penalty."""
  for task_id in g1_tracking_task_ids:
    if "LatentTracking" in task_id:
      continue
    cfg = load_env_cfg(task_id)

    assert "waist_action_rate_l2" in cfg.rewards
    reward = cfg.rewards["waist_action_rate_l2"]
    assert reward.func is mdp.joint_action_rate_l2
    assert reward.params["action_name"] == "joint_pos"
    assert reward.params["asset_cfg"].joint_names == (
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
    )


def test_tracking_bfm_defaults_to_inertia_randomization() -> None:
  """BFM tracking should randomize torso and non-torso inertial parameters."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1")

  assert "base_inertia" in cfg.events
  assert "body_inertia" in cfg.events
  assert "base_com" not in cfg.events

  base_event = cfg.events["base_inertia"]
  body_event = cfg.events["body_inertia"]

  assert base_event.mode == "startup"
  assert base_event.func is dr.pseudo_inertia
  assert base_event.params["asset_cfg"].body_names == ("torso_link",)
  assert base_event.params["alpha_range"] == (
    0.5 * math.log(0.92),
    0.5 * math.log(1.08),
  )
  assert base_event.params["t_range"] == (-0.01, 0.01)

  assert body_event.mode == "startup"
  assert body_event.func is dr.pseudo_inertia
  assert body_event.params["asset_cfg"].body_names == r"^(?!torso_link$).+"
  assert body_event.params["alpha_range"] == (
    0.5 * math.log(0.95),
    0.5 * math.log(1.05),
  )
  assert body_event.params["t_range"] == (-0.005, 0.005)


def test_tracking_bfm_play_keeps_inertia_randomization() -> None:
  """BFM play mode should keep startup inertial DR like other startup DR."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1", play=True)

  assert "push_robot" not in cfg.events
  assert "base_inertia" in cfg.events
  assert "body_inertia" in cfg.events


def test_tracking_1stage_task_uses_sparse_actor_obs() -> None:
  """The 1-stage tracking task should expose sparse actor observations only."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage")

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  assert set(actor_terms.keys()) == {
    "ee_pose",
    "base_lin_vel_b",
    "base_ang_vel_b",
    "anchor_height_w",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  }
  assert actor_terms["ee_pose"].params["ee_body_names"] == (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  assert actor_terms["ee_pose"].params["anchor_body_name"] == "pelvis"
  assert actor_terms["base_lin_vel_b"].params["anchor_body_name"] == "pelvis"
  assert actor_terms["base_ang_vel_b"].params["anchor_body_name"] == "pelvis"
  assert actor_terms["anchor_height_w"].params["anchor_body_name"] == "pelvis"

  assert "body_pos" in critic_terms
  assert "body_ori" in critic_terms
  assert cfg.commands["motion"].history_steps == 0
  assert cfg.commands["motion"].future_steps == 1


def test_tracking_bfm_action_trunk_task_config() -> None:
  """The action-trunk tracking task should expose a 4-slice policy action."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk")

  assert cfg.action_trunk_len == 4
  assert cfg.decimation == 4
  assert "joint_pos" in cfg.actions
  assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
  assert cfg.actions["joint_pos"].scale == G1_ACTION_SCALE
