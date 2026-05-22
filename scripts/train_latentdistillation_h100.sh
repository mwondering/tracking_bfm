TEACHER_CKPT=/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic_prior/2026-05-20_13-53-11_teacher_v2_decimation4_4gpu_16384_resume2/model_70000.pt
MOTION_PATH=/data/zcy/motion_data

uv run train Mjlab-LatentDistillation-Flat-Unitree-G1 \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs 8192\
    --env.commands.motion.sampling-mode uniform \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --agent.teacher_checkpoint_path "$TEACHER_CKPT" \
    --agent.latent_dim 64 \
    --agent.kl_weight 1e-4 \
    --agent.kl_warmup_iterations 500 \
    --agent.free_nats_per_dim 0.02 \
    --agent.latent_smooth_weight 0 \
    --agent.max_iterations 30000 \
    --agent.num_steps_per_env 24 \
    --agent.save_interval 2000 \
    --agent.beta_decay_steps 1 \
    --agent.experiment_name g1_latent_distillation \
    --agent.run_name latent_distill_g1 \
    --agent.wandb_project tracking_bfm_distillation \
    --agent.upload-model False \
    --debug False \
    --gpu_ids "[4,5,6,7]" \
    --agent.latent_regularization wae_mmd \
    --agent.mmd_weight 1e-2 \
    --agent.mmd_kernel_scales "(0.5,1.0,2.0,4.0)" \
    --agent.latent_smooth_weight 0.0
