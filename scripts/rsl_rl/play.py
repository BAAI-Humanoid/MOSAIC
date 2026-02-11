"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion", type=str, default=None, help="Path to the motion file or motion directory.")
parser.add_argument("--skip_critic", action="store_true", default=False, help="Only load actor weights.")
parser.add_argument("--disable_motion_group_sampling", action="store_true", default=False, help="Disable motion group sampling ratios (use uniform sampling).")
parser.add_argument(
    "--start_frame",
    type=int,
    default=10,
    help="Start frame index (0-based) for motion playback.",
)
parser.add_argument(
    "--enable_motion_randomization",
    action="store_true",
    default=False,
    help="Keep motion randomization ranges (pose/velocity/joint) instead of zeroing them.",
)
parser.add_argument(
    "--disable_obs_noise",
    action="store_true",
    default=True,
    help="Disable observation corruption/noise during playback.",
)
parser.add_argument(
    "--disable_events",
    action="store_true",
    default=True,
    help="Disable event manager randomizations during playback.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Load policy configuration from checkpoint's params/agent.yaml to ensure compatibility
    # This overrides the default configuration with the checkpoint's actual configuration
    checkpoint_dir = os.path.dirname(resume_path)
    params_yaml_path = os.path.join(checkpoint_dir, "params", "agent.yaml")
    if os.path.exists(params_yaml_path):
        import yaml
        with open(params_yaml_path, 'r') as f:
            checkpoint_cfg = yaml.safe_load(f)

        # Override policy configuration from checkpoint
        if 'policy' in checkpoint_cfg:
            policy_cfg = checkpoint_cfg['policy']
            if 'ref_vel_skip_first_layer' in policy_cfg:
                agent_cfg.policy.ref_vel_skip_first_layer = policy_cfg['ref_vel_skip_first_layer']
                print(f"[Play] Loaded ref_vel_skip_first_layer={policy_cfg['ref_vel_skip_first_layer']} from checkpoint")
            if 'ref_vel_dim' in policy_cfg:
                agent_cfg.policy.ref_vel_dim = policy_cfg['ref_vel_dim']

    if args_cli.motion is not None:
        print(f"[INFO]: Using motion directory or file from CLI: {args_cli.motion}")
        env_cfg.commands.motion.motion = args_cli.motion
        # Optionally disable motion group sampling ratios for evaluation
        if args_cli.disable_motion_group_sampling and hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
            env_cfg.commands.motion.motion_group_sampling_ratios = None
            print("[INFO]: Disabled motion group sampling ratios for evaluation (uniform sampling).")
        if args_cli.start_frame is not None and hasattr(env_cfg.commands.motion, "start_frame"):
            env_cfg.commands.motion.start_from_beginning = True
            env_cfg.commands.motion.start_frame = args_cli.start_frame
            print(f"[INFO]: Forcing motion playback to start from frame {args_cli.start_frame}.")
    else:
        raise ValueError("Motion file or motion directory is required for evaluation.")

    if not args_cli.enable_motion_randomization and hasattr(env_cfg, "commands"):
        motion_cfg = getattr(env_cfg.commands, "motion", None)
        if motion_cfg is not None:
            zero_ranges = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            if hasattr(motion_cfg, "pose_range"):
                motion_cfg.pose_range = dict(zero_ranges)
            if hasattr(motion_cfg, "velocity_range"):
                motion_cfg.velocity_range = dict(zero_ranges)
            if hasattr(motion_cfg, "joint_position_range"):
                motion_cfg.joint_position_range = (0.0, 0.0)
            print("[INFO]: Zeroed motion randomization ranges for evaluation.")

    if args_cli.disable_obs_noise and hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic", "ref_vel_estimator"):
            if hasattr(env_cfg.observations, group_name):
                group_cfg = getattr(env_cfg.observations, group_name)
                if hasattr(group_cfg, "enable_corruption"):
                    group_cfg.enable_corruption = False
        print("[INFO]: Disabled observation corruption/noise for evaluation.")

    if args_cli.disable_events and hasattr(env_cfg, "events"):
        env_cfg.events = None
        print("[INFO]: Disabled event manager for evaluation.")

    # Disable time-out termination during play to allow continuous replay
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        print("[INFO]: Disabling timeout termination for playback run.")
        env_cfg.terminations.time_out = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False, load_critic=not args_cli.skip_critic)

    # obtain the trained policy for inference
    # Check if velocity estimator is enabled
    use_velocity_estimator = (hasattr(ppo_runner.alg, 'ref_vel_estimator') and
                              ppo_runner.alg.ref_vel_estimator is not None and
                              hasattr(ppo_runner.alg, 'use_estimate_ref_vel') and
                              ppo_runner.alg.use_estimate_ref_vel)

    if use_velocity_estimator:
        print("[Play] Using velocity estimator for inference")
        # For velocity estimator, we need to handle observation processing manually
        # Cannot use get_inference_policy because it doesn't handle velocity augmentation
        policy = None  # Will process observations manually in the loop
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # Get velocity estimator info if available
    ref_vel_estimator = None
    ref_vel_estimator_obs_dim = None
    if hasattr(ppo_runner.alg, 'ref_vel_estimator') and ppo_runner.alg.ref_vel_estimator is not None:
        ref_vel_estimator = ppo_runner.alg.ref_vel_estimator
        if hasattr(ppo_runner.alg, 'ref_vel_estimator_obs_shape') and ppo_runner.alg.ref_vel_estimator_obs_shape is not None:
            ref_vel_estimator_obs_dim = ppo_runner.alg.ref_vel_estimator_obs_shape[0]

    export_motion_policy_as_onnx(
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
        ref_vel_estimator=ref_vel_estimator,
        ref_vel_estimator_obs_dim=ref_vel_estimator_obs_dim,
    )

    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # reset environment
    env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if use_velocity_estimator:
                # Access observation manager directly to get all observation groups
                obs_manager = env.unwrapped.observation_manager

                # Compute observations for all groups - returns dict[group_name, tensor]
                obs_dict = obs_manager.compute()

                # Extract policy and ref_vel_estimator observations
                policy_obs = obs_dict["policy"].to(ppo_runner.device)
                ref_vel_estimator_obs = obs_dict["ref_vel_estimator"].to(ppo_runner.device)

                # Normalize policy obs
                policy_obs_normalized = ppo_runner.obs_normalizer(policy_obs)

                # Estimate velocity
                estimated_ref_vel = ppo_runner.alg.ref_vel_estimator(ref_vel_estimator_obs) * 1.0
                print(f"[Play] Estimated ref vel: {estimated_ref_vel.cpu().numpy()}")

                # Augment observations
                obs_augmented = torch.cat([policy_obs_normalized, estimated_ref_vel], dim=-1)

                # Get actions
                actions = ppo_runner.alg.policy.act_inference(obs_augmented)
            else:
                # Standard inference without velocity estimator
                obs, _ = env.get_observations()
                actions = policy(obs)

            # env stepping
            _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
