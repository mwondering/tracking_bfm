MOTION_PATH=/home/lenovo/DATASETS/Data10k
TEACHER_CKPT=/home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0522_tracking_teacher/model_66000.pt

uv run train Mjlab-LatentDistillation-Flat-Unitree-G1 \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs 512 \
    --env.commands.motion.sampling-mode uniform \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --agent.teacher_checkpoint_path "$TEACHER_CKPT" \
    --agent.latent_dim 64 \
    --agent.kl_weight 1e-4 \
    --agent.kl_warmup_iterations 2000 \
    --agent.free_nats_per_dim 0.02 \
    --agent.latent_smooth_weight 1e-3 \
    --agent.max_iterations 30000 \
    --agent.num_steps_per_env 24 \
    --agent.save_interval 2000 \
    --agent.beta_decay_steps 1 \
    --agent.experiment_name g1_latent_distillation \
    --agent.run_name latent_distill_g1 \
    --agent.wandb_project tracking_bfm_distillation \
    --agent.upload-model False \
    --debug False \
    # --gpu_ids "[4,5]"