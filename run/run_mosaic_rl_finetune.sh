# 2nd stage
# MOSAIC rl finetune with teleop motions
HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=6 scripts/rsl_rl/train.py \
    --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
    --distributed \
    --num_envs=24000 \
    --motion /path/to/teleop_motions \
    --headless \
    --logger wandb \
    --log_project_name GMT_MOSAIC_RL \
    --run_name GMT_MOSAIC_RL_FINETUNE \
    --load_run=/path/to/gmt_run \
    --checkpoint=/path/to/checkpoint \
    --resume

# HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py \
#     --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
#     --num_envs=12000 \
#     --motion /path/to/teleop_motions \
#     --headless \
#     --logger wandb \
#     --log_project_name GMT_MOSAIC_RL \
#     --run_name GMT_MOSAIC_RL_FINETUNE \
#     --load_run=/path/to/gmt_run \
#     --checkpoint=/path/to/checkpoint \
#     --resume
