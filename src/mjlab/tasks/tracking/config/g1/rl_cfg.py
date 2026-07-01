"""RL configuration for Unitree G1 tracking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from .attention_cfg import (
  AttentionVariant,
  TrackingAttentionModelCfg,
  tracking_attention_actor_cfg,
)

SPARSETRACK_FULL_REF_CRITIC_CLASS = (
  "mjlab.tasks.tracking.rl.attention_models:SparseTrackFullRefAttentionCritic"
)
SPARSETRACK_SPLIT_LR_PPO_CLASS = "mjlab.tasks.tracking.rl.ppo:SparseTrackSplitLrPPO"


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
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
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=300_000,
  )


def unitree_g1_trackingbfm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=300_000,
  )


def unitree_g1_trackingbfm_attention_ppo_runner_cfg(
  variant: AttentionVariant,
) -> RslRlOnPolicyRunnerCfg:
  """Create BFM PPO cfg with an attention actor and the baseline MLP critic."""
  cfg = unitree_g1_trackingbfm_ppo_runner_cfg()
  cfg.actor = tracking_attention_actor_cfg(variant)
  cfg.experiment_name = "test_optimal_tracking_bfm_attention"
  if variant == "sparsetrack_full_ref":
    actor_cfg = cfg.actor
    assert isinstance(actor_cfg, TrackingAttentionModelCfg)
    cfg.critic = TrackingAttentionModelCfg(
      class_name=SPARSETRACK_FULL_REF_CRITIC_CLASS,
      hidden_dims=(512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=None,
      history_length=actor_cfg.history_length,
      frame_dim=actor_cfg.frame_dim,
      command_dim=actor_cfg.command_dim,
      num_dofs=actor_cfg.num_dofs,
      d_model=actor_cfg.d_model,
      num_heads=actor_cfg.num_heads,
      ffn_dim=actor_cfg.ffn_dim,
      history_layers=actor_cfg.history_layers,
      cross_layers=actor_cfg.cross_layers,
      dropout=actor_cfg.dropout,
      attention_activation=actor_cfg.attention_activation,
      head_hidden_dims=actor_cfg.head_hidden_dims,
      task_embedder_hidden_dims=actor_cfg.task_embedder_hidden_dims,
      reduced_task_dim=None,
    )
    cfg.algorithm.class_name = SPARSETRACK_SPLIT_LR_PPO_CLASS
    cfg.algorithm.learning_rate = 2.0e-5
    cfg.algorithm.actor_learning_rate = 2.0e-5
    cfg.algorithm.critic_learning_rate = 1.0e-3
    cfg.algorithm.num_learning_epochs = 2
    cfg.algorithm.num_mini_batches = 16
    cfg.algorithm.entropy_coef = 0.005
    cfg.num_steps_per_env = 32
  return cfg


def unitree_g1_trackingbfm_action_trunk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the G1 BFM action-trunk task."""
  cfg = unitree_g1_trackingbfm_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_action_trunk"
  return cfg
