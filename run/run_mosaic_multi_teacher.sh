# 3rd stage
# MOSAIC multi-teacher residual
HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=6 scripts/rsl_rl/train.py \
    --task=MOSAIC-MultiTeacher-Residual-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs=24000 \
    --motion /path/to/motion \
    --headless \
    --logger wandb \
    --log_project_name GMT_MOSAIC_MultiTeacher_Residual \
    --run_name GMT_MOSAIC_MULTITEACHER_RESIDUAL \
    --max_iterations 2001

# HYDRA_FULL_ERROR=1 python scripts/rsl_rl/train.py \
#     --task=MOSAIC-MultiTeacher-Residual-Tracking-Flat-G1-v0 \
#     --num_envs=12000 \
#     --motion /path/to/motion \
#     --headless \
#     --logger wandb \
#     --log_project_name GMT_MOSAIC_MultiTeacher_Residual \
#     --run_name GMT_MOSAIC_MULTITEACHER_RESIDUAL \
#     --max_iterations 2001
