uv run analyze-latent-space Mjlab-LatentDistillation-Flat-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_latent_distillation/2026-05-22_22-38-30_latent_distill_g1/model_4500.pt \
    --motion-path /home/lenovo/DATASETS/Data10k/homejrhanprojectsPBHC-InternalPBHC-Motiong1robotlafanwalk1_subject1_0_7840_cont_mask_inter05_S00-30_ \
    --output-dir logs/latent_analysis/walk \
    --num-envs 2048 \
    --num-points 500000 \
    --device cuda:0 \
    --motion-history-steps 0 \
    --motion-future-steps 1 \
    --no-deterministic
