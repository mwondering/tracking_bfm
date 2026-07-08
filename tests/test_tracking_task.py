"""Tests specific to motion tracking tasks."""

from typing import cast

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.tasks.tracking.rl.attention_models import TERM_DIMS


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
    if task_id.endswith("-NoRegNoDR"):
      continue
    cfg = load_env_cfg(task_id)

    assert "waist_action_rate_l2" in cfg.rewards
    reward = cfg.rewards["waist_action_rate_l2"]
    assert reward.func is mdp.joint_action_rate_l2
    assert reward.weight == -5.0e-2
    assert reward.params["action_name"] == "joint_pos"
    assert reward.params["asset_cfg"].joint_names == (
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
    )


def test_g1_tracking_global_root_position_weight_is_one(
  g1_tracking_task_ids: list[str],
) -> None:
  """G1 tracking tasks should emphasize global root position tracking."""
  for task_id in g1_tracking_task_ids:
    if "LatentTracking" in task_id:
      continue
    cfg = load_env_cfg(task_id)

    assert cfg.rewards["motion_global_root_pos"].weight == 1.0


def test_g1_tracking_foot_friction_uses_sonic_plus_range(
  g1_tracking_task_ids: list[str],
) -> None:
  """G1 tracking foot friction should cover the SONIC range with a higher cap."""
  for task_id in g1_tracking_task_ids:
    if task_id.endswith("-NoRegNoDR"):
      continue
    cfg = load_env_cfg(task_id)

    assert cfg.events["foot_friction"].params["ranges"] == (0.3, 2.0)


def test_tracking_bfm_defaults_to_torso_mass_and_com_randomization() -> None:
  """BFM tracking should randomize torso mass/COM and leave inertia disabled."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1")

  assert "base_com" in cfg.events
  assert "base_mass" in cfg.events
  assert "base_inertia" not in cfg.events
  assert "body_inertia" not in cfg.events
  base_event = cfg.events["base_com"]
  mass_event = cfg.events["base_mass"]

  assert base_event.mode == "startup"
  assert base_event.func is dr.body_com_offset
  assert base_event.params["asset_cfg"].body_names == ("torso_link",)
  assert base_event.params["ranges"] == {
    0: (-0.075, 0.075),
    1: (-0.075, 0.075),
    2: (-0.075, 0.075),
  }
  assert mass_event.mode == "startup"
  assert mass_event.func is dr.body_mass
  assert mass_event.params["asset_cfg"].body_names == ("torso_link",)
  assert mass_event.params["operation"] == "add"
  assert mass_event.params["ranges"] == (-1.0, 1.0)


def test_tracking_bfm_play_keeps_inertia_randomization() -> None:
  """BFM play mode should keep startup torso COM DR like other startup DR."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1", play=True)

  assert "push_robot" not in cfg.events
  assert "base_com" in cfg.events
  assert "base_mass" in cfg.events
  assert "base_inertia" not in cfg.events
  assert "body_inertia" not in cfg.events


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
  motion_cmd = cast(MultiMotionCommandCfg, cfg.commands["motion"])
  assert motion_cmd.history_steps == 0
  assert motion_cmd.future_steps == 1


def test_tracking_bfm_action_trunk_task_config() -> None:
  """The action-trunk tracking task should expose a 4-slice policy action."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk")

  assert cfg.action_trunk_len == 4
  assert cfg.decimation == 4
  assert "joint_pos" in cfg.actions
  assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
  assert cfg.actions["joint_pos"].scale == G1_ACTION_SCALE


def test_tracking_bfm_test_optimal_uses_full_critic_actor_obs() -> None:
  """The optimality probe should give the policy the full critic observation."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal")

  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]

  assert set(actor.terms.keys()) == set(critic.terms.keys())
  assert actor.enable_corruption is False
  assert critic.enable_corruption is False
  assert all(term.noise is None for term in actor.terms.values())


def test_tracking_bfm_test_optimal_uses_global_body_pose_rewards() -> None:
  """The optimality probe should track body poses in the world frame."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal")

  assert (
    cfg.rewards["motion_body_pos"].func is mdp.motion_global_body_position_error_exp
  )
  assert (
    cfg.rewards["motion_body_ori"].func is mdp.motion_global_body_orientation_error_exp
  )
  assert cfg.rewards["motion_body_lin_vel"].func is (
    mdp.motion_global_body_linear_velocity_error_exp
  )
  assert cfg.rewards["motion_body_ang_vel"].func is (
    mdp.motion_global_body_angular_velocity_error_exp
  )


def test_tracking_bfm_test_optimal_no_reg_no_dr_removes_interference() -> None:
  """The pure optimality probe should remove DR and regularization rewards."""
  cfg = load_env_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR")

  assert cfg.events == {}
  assert "action_rate_l2" not in cfg.rewards
  assert "waist_action_rate_l2" not in cfg.rewards
  assert "joint_limit" not in cfg.rewards
  assert "self_collisions" not in cfg.rewards

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  assert motion_cmd.pose_range == {}
  assert motion_cmd.velocity_range == {}
  assert motion_cmd.joint_position_range == (0.0, 0.0)


ATTENTION_TEST_OPTIMAL_TASKS = {
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-FullObsCausalAttn-NoRegNoDR": (
    "mjlab.tasks.tracking.rl.attention_models:FullObsCausalAttentionActor"
  ),
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-ProprioRefCrossAttn-NoRegNoDR": (
    "mjlab.tasks.tracking.rl.attention_models:ProprioRefCrossAttentionActor"
  ),
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttn-NoRegNoDR": (
    "mjlab.tasks.tracking.rl.attention_models:HistProprioCrossAttentionActor"
  ),
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttnActorCritic-NoRegNoDR": (
    "mjlab.tasks.tracking.rl.attention_models:HistProprioCrossAttentionActor"
  ),
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR": (
    "mjlab.tasks.tracking.rl.attention_models:SparseTrackFullRefAttentionActor"
  ),
}


def test_tracking_attention_test_optimal_tasks_are_registered() -> None:
  registered = set(list_tasks())

  for task_id in ATTENTION_TEST_OPTIMAL_TASKS:
    assert task_id in registered


def test_tracking_attention_test_optimal_uses_no_future_ref_and_actor_history() -> None:
  for task_id in ATTENTION_TEST_OPTIMAL_TASKS:
    cfg = load_env_cfg(task_id)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionCommandCfg)
    assert motion_cmd.history_steps == 0
    assert motion_cmd.future_steps == 1

    actor_terms = cfg.observations["actor"].terms
    assert actor_terms
    assert tuple(actor_terms) == tuple(TERM_DIMS)
    for term in actor_terms.values():
      assert term.history_length == 11
      assert term.flatten_history_dim is True


def test_tracking_attention_test_optimal_keeps_baseline_mlp_critic() -> None:
  baseline = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR"),
  )

  for task_id in (
    task_id
    for task_id in ATTENTION_TEST_OPTIMAL_TASKS
    if "SparseTrackFullRefAttn" not in task_id
    and "HistProprioCrossAttnActorCritic" not in task_id
  ):
    rl_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(task_id))

    assert rl_cfg.critic == baseline.critic
    assert rl_cfg.actor.class_name == ATTENTION_TEST_OPTIMAL_TASKS[task_id]


def test_hist_proprio_cross_attention_baseline_keeps_mlp_critic() -> None:
  baseline = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR"),
  )
  task_id = (
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-HistProprioCrossAttn-NoRegNoDR"
  )

  cfg = load_env_cfg(task_id)
  rl_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(task_id))

  assert rl_cfg.critic == baseline.critic
  for term in cfg.observations["critic"].terms.values():
    assert term.history_length == 0


def test_hist_proprio_cross_actor_critic_variant_uses_transformer_critic() -> None:
  task_id = (
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-"
    "HistProprioCrossAttnActorCritic-NoRegNoDR"
  )

  cfg = load_env_cfg(task_id)
  rl_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(task_id))

  assert (
    rl_cfg.critic.class_name
    == "mjlab.tasks.tracking.rl.attention_models:HistProprioCrossAttentionCritic"
  )
  assert rl_cfg.critic.distribution_cfg is None
  for term in cfg.observations["critic"].terms.values():
    assert term.history_length == 11
    assert term.flatten_history_dim is True


def test_sparsetrack_attention_test_optimal_uses_conservative_ppo_settings() -> None:
  rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg(
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR"
    ),
  )

  assert rl_cfg.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 0.8,
    "std_type": "scalar",
    "std_range": (0.001, 1.0),
  }
  assert rl_cfg.num_steps_per_env == 32
  assert rl_cfg.algorithm.learning_rate == 2.0e-5
  assert rl_cfg.algorithm.num_learning_epochs == 2
  assert rl_cfg.algorithm.num_mini_batches == 16
  assert rl_cfg.algorithm.entropy_coef == 0.005


def test_sparsetrack_attention_test_optimal_uses_transformer_critic() -> None:
  rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg(
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR"
    ),
  )

  assert (
    rl_cfg.critic.class_name
    == "mjlab.tasks.tracking.rl.attention_models:SparseTrackFullRefAttentionCritic"
  )


def test_sparsetrack_attention_test_optimal_uses_actor_critic_learning_rates() -> None:
  rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg(
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-SparseTrackFullRefAttn-NoRegNoDR"
    ),
  )

  assert rl_cfg.algorithm.actor_learning_rate == 2.0e-5
  assert rl_cfg.algorithm.critic_learning_rate == 1.0e-3
