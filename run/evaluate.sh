# Evaluation
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
    --num_envs=1 \
    --task=MOSAIC-MultiTeacher-Residual-Tracking-Flat-G1-v0 \
    --motion /path/to/motion \
    --load_run=/path/to/run \
    --checkpoint=/path/to/checkpoint \
    --disable_motion_group_sampling
