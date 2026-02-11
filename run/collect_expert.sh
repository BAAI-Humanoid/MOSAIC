# Expert trajectory collection
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/collect_expert_trajectories.py \
    --task=Expert-General-Tracking-Flat-G1-MOSAIC-v0 \
    --checkpoint_path=/path/to/checkpoint \
    --motion /path/to/motion \
    --output_path /path/to/output \
    --num_envs=800 \
    --steps_per_env=500 \
    --student_task=MOSAIC-Distill-General-Tracking-Flat-G1-v0 \
    --disable_student_noise
