uv run analyze-latent-space Mjlab-LatentDistillation-Flat-Unitree-G1 \
    --checkpoint-file /home/lenovo/workspace/UNICTL/tracking_bfm/logs/rsl_rl/0529_ckpt/latent_distillation_full/model_5000.pt \
    --motion-path /home/lenovo/DATASETS/test_motion/dance1_subject2_0_3945/ \
    --output-dir logs/latent_analysis/dance_full \
    --num-envs 2048 \
    --num-points 500000 \
    --device cuda:0 \
    --motion-history-steps 0 \
    --motion-future-steps 1 \
    --no-deterministic
