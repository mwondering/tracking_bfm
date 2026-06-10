TEACHER_CKPT=/data/wxy/tracking_bfm/logs/rsl_rl/teacher_amass_lafan_noiton_sonic_prior/2026-05-20_13-53-11_teacher_v2_decimation4_4gpu_16384_resume2/model_70000.pt
MOTION_PATH=/data/zcy/motion_data/
LATENT_REGULARIZATION="${LATENT_REGULARIZATION:-bfmzero_sphere}"
SPHERE_ORTHONORMAL_WEIGHT="${SPHERE_ORTHONORMAL_WEIGHT:-1e-3}"
SPHERE_KNN_SMOOTH_WEIGHT="${SPHERE_KNN_SMOOTH_WEIGHT:-1e-3}"
SPHERE_KNN_K="${SPHERE_KNN_K:-4}"
SPHERE_KNN_MAX_SAMPLES="${SPHERE_KNN_MAX_SAMPLES:-2048}"

uv run train Mjlab-LatentDistillation-Flat-Unitree-G1 \
    --env.commands.motion.motion-path "$MOTION_PATH" \
    --env.scene.num-envs 16384 \
    --env.commands.motion.sampling-mode adaptive \
    --env.commands.motion.history_steps 0 \
    --env.commands.motion.future_steps 1 \
    --agent.teacher_checkpoint_path "$TEACHER_CKPT" \
    --agent.latent_dim 64 \
    --agent.kl_weight 1e-4 \
    --agent.kl_warmup_iterations 500 \
    --agent.free_nats_per_dim 0.02 \
    --agent.max_iterations 30000 \
    --agent.num_steps_per_env 24 \
    --agent.save_interval 500 \
    --agent.beta_decay_steps 1 \
    --agent.experiment_name g1_latent_distillation \
    --agent.run_name latent_distill_g1_full_2gpu_16384 \
    --agent.wandb_project tracking_bfm_distillation \
    --agent.upload-model False \
    --debug False \
    --gpu_ids "[1,2,3]" \
    --agent.latent_regularization "$LATENT_REGULARIZATION" \
    --agent.mmd_weight 3e-2 \
    --agent.mmd_kernel_scales "(0.25,0.5,1.0,2.0,4.0,8.0)" \
    --agent.learning_rate 2e-4 \
    --agent.latent_smooth_weight 10e-4 \
    --agent.sphere_orthonormal_weight "$SPHERE_ORTHONORMAL_WEIGHT" \
    --agent.sphere_knn_smooth_weight "$SPHERE_KNN_SMOOTH_WEIGHT" \
    --agent.sphere_knn_k "$SPHERE_KNN_K" \
    --agent.sphere_knn_max_samples "$SPHERE_KNN_MAX_SAMPLES"
