MOTION_PATH=/home/lenovo/DATASETS/Data10k
TEACHER_CKPT=/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0522_tracking_teacher/model_66000.pt

uv run train Mjlab-LatentDistillation-Flat-Unitree-G1 \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs 2048 \
    --env.commands.motion.sampling-mode uniform \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --agent.teacher_checkpoint_path "$TEACHER_CKPT" \
    --agent.latent_dim 64 \
    --agent.kl_weight 1e-4 \
    --agent.kl_warmup_iterations 2000 \
    --agent.free_nats_per_dim 0.02 \
    --agent.latent_smooth_weight 5e-4 \
    --agent.mmd_max_samples 1024 \
    --agent.latent_smooth_max_pairs 2048 \
    --agent.max_iterations 30000 \
    --agent.num_steps_per_env 24 \
    --agent.save_interval 500 \
    --agent.beta_decay_steps 1 \
    --agent.experiment_name g1_latent_distillation \
    --agent.run_name latent_distill_g1 \
    --agent.wandb_project tracking_bfm_distillation \
    --agent.upload-model False \
    --debug False \
    --agent.latent_regularization wae_mmd \
    --agent.mmd_weight 3e-2 \
    --agent.mmd_kernel_scales "(0.25,0.5,1.0,2.0,4.0,8.0)" \
    --agent.learning_rate 3e-4 \
    # --gpu_ids "[4,5]"
