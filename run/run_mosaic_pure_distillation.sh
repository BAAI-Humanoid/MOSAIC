# distillation stage (optional)
# MOSAIC pure distillation
HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=6 scripts/rsl_rl/train.py \
    --task=MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs=24000 \
    --motion /path/to/motion \
    --headless \
    --logger wandb \
    --log_project_name GMT_MOSAIC_Distill \
    --run_name GMT_MOSAIC_DISTILLATION

# HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py \
#     --task=MOSAIC-Pure-Distill-General-Tracking-Flat-G1-v0 \
#     --num_envs=12000 \
#     --motion /path/to/motion \
#     --headless \
#     --logger wandb \
#     --log_project_name GMT_MOSAIC_Distill \
#     --run_name GMT_MOSAIC_DISTILLATION
