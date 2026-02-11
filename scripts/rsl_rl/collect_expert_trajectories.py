"""
Collect Expert Trajectories for MOSAIC Off-Policy Learning
"""

import argparse
import os
import pathlib
import sys

import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect expert trajectories from a trained policy.")
parser.add_argument("--motion", type=str, required=True, help="Path to the motion directory.")
parser.add_argument("--output_path", type=str, required=True, help="Path to save expert trajectories (.npy).")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to policy checkpoint (.pt file).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--steps_per_env", type=int, default=100, help="Number of steps to collect per environment (trajectory length).")
parser.add_argument("--max_trajectories", type=int, default=None, help="Maximum trajectories to collect (default: all motions).")
parser.add_argument("--start_index", type=int, default=0, help="Start index in the motion list.")
parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive) in the motion list.")
parser.add_argument("--enable_terminations", action="store_true", default=True, help="Keep termination terms enabled.")
parser.add_argument("--student_obs_history", type=int, default=0, help="History length for student observations (0=no history). Overrides student_task if both provided.")
parser.add_argument("--student_task", type=str, default=None, help="Student task name to extract observation config (e.g., MOSAIC-Distill-General-Tracking-Flat-G1-v0).")
parser.add_argument("--disable_student_noise", action="store_true", default=False, help="Disable noise/corruption for student observations (for testing history consistency).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.collection import ExpertTrajectoryCollector


def _resolve_checkpoint(agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    """Resolve checkpoint path from args, wandb, or local logs."""
    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)
        resume_path = os.path.abspath(f"./logs/rsl_rl/temp/{file}")
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        return resume_path

    if args_cli.checkpoint_path:
        return os.path.abspath(args_cli.checkpoint_path)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    return resume_path


def _adjust_num_envs_for_motions(num_envs: int, num_motions: int) -> int:
    """Cap num_envs to available motions to avoid idle environments."""
    if num_motions <= 0:
        raise ValueError("No motions found for collection.")
    return min(num_envs, num_motions)


def _disable_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Disable startup/interval randomization for deterministic collection."""
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        disabled_events = []
        for name in vars(env_cfg.events).keys():
            if not name.startswith("_"):
                disabled_events.append(name)
                setattr(env_cfg.events, name, None)
        print(f"[INFO] Disabled {len(disabled_events)} event randomization terms for collection:")
        for event_name in disabled_events:
            print(f"  - {event_name}")



def _policy_step(
    runner: OnPolicyRunner, obs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run policy inference and return actions plus action distribution stats."""
    obs_in = obs
    if runner.cfg.get("empirical_normalization", False):
        obs_in = runner.obs_normalizer(obs_in)
    runner.alg.policy.update_distribution(obs_in)
    actions = runner.alg.policy.act_inference(obs_in)
    action_mean = runner.alg.policy.action_mean.detach()
    action_sigma = runner.alg.policy.action_std.detach()
    return actions, action_mean, action_sigma


def _set_robot_state(command_term, robot, env_ids: torch.Tensor):
    """Set robot state from reference motion."""
    root_pos = command_term.body_pos_w[:, 0].clone()
    root_ori = command_term.body_quat_w[:, 0].clone()
    root_lin_vel = command_term.body_lin_vel_w[:, 0].clone()
    root_ang_vel = command_term.body_ang_vel_w[:, 0].clone()
    root_states = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    robot.write_root_state_to_sim(root_states[env_ids], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        command_term.joint_pos[env_ids], command_term.joint_vel[env_ids], env_ids=env_ids
    )


def _compute_obs_dict(base_env, group_name: str = "policy") -> dict[str, torch.Tensor]:
    """
    Compute observation dictionary for a given group.
    """
    obs_dict = {}
    obs_manager = base_env.observation_manager
    group_term_names = obs_manager._group_obs_term_names[group_name]
    group_term_cfgs = obs_manager._group_obs_term_cfgs[group_name]

    for term_name, term_cfg in zip(group_term_names, group_term_cfgs):
        # Compute term's value
        obs_value = term_cfg.func(base_env, **term_cfg.params).clone()

        # Apply post-processing (modifiers, noise, clip, scale)
        if term_cfg.modifiers is not None:
            for modifier in term_cfg.modifiers:
                obs_value = modifier.func(obs_value, **modifier.params)
        if term_cfg.noise:
            obs_value = term_cfg.noise.func(obs_value, term_cfg.noise)
        if term_cfg.clip:
            obs_value = obs_value.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
        if term_cfg.scale is not None:
            obs_value = obs_value.mul_(term_cfg.scale)

        # Handle observation manager's own history buffer (if enabled)
        if term_cfg.history_length > 0:
            obs_manager._group_obs_term_history_buffer[group_name][term_name].append(obs_value)
            if term_cfg.flatten_history_dim:
                obs_dict[term_name] = obs_manager._group_obs_term_history_buffer[group_name][term_name].buffer.reshape(
                    base_env.num_envs, -1
                )
            else:
                obs_dict[term_name] = obs_manager._group_obs_term_history_buffer[group_name][term_name].buffer
        else:
            obs_dict[term_name] = obs_value

    return obs_dict


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect expert trajectories from trained policy."""

    print("=" * 80)
    print("Expert Trajectory Collection for MOSAIC")
    print("=" * 80)

    # Parse config
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if env_cfg.scene.num_envs < 1:
        raise ValueError("num_envs must be >= 1.")

    if args_cli.motion is None:
        raise ValueError("Motion file or motion directory is required for collection.")

    motion_path = pathlib.Path(args_cli.motion)
    if not motion_path.exists():
        raise ValueError(f"Motion path not found: {motion_path}")

    # Setup motion configuration
    env_cfg.commands.motion.motion = args_cli.motion
    env_cfg.commands.motion.debug_vis = False
    if hasattr(env_cfg.commands.motion, "resample_motions_every_s"):
        env_cfg.commands.motion.resample_motions_every_s = 1.0e9  # Don't resample during collection
    _disable_randomization(env_cfg)

    # Cap num_envs to avoid idle environments when motions < envs
    if motion_path.is_dir():
        file_glob = getattr(env_cfg.commands.motion, "file_glob", "*.npz")
        motion_paths = list(motion_path.rglob(file_glob))
        num_motions_hint = len(motion_paths)
    else:
        num_motions_hint = 1
    if num_motions_hint == 0:
        raise ValueError(f"No motion files found in: {motion_path}")
    adjusted_envs = _adjust_num_envs_for_motions(env_cfg.scene.num_envs, num_motions_hint)
    if adjusted_envs != env_cfg.scene.num_envs:
        print(f"[WARN] Capping num_envs from {env_cfg.scene.num_envs} to {adjusted_envs} "
              f"to match {num_motions_hint} motions.")
        env_cfg.scene.num_envs = adjusted_envs

    # Disable terminations if requested
    if not args_cli.enable_terminations and hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None
        print("[INFO] Disabled termination terms for collection.")

    # Resolve checkpoint path
    resume_path = _resolve_checkpoint(agent_cfg)
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Motion directory: {args_cli.motion}")
    print(f"[INFO] Output path: {args_cli.output_path}")
    print(f"[INFO] Number of envs: {env_cfg.scene.num_envs}")
    print(f"[INFO] Steps per env (trajectory length): {args_cli.steps_per_env}")
    print(f"[INFO] Student obs history: {args_cli.student_obs_history}")

    # Add policy_student observation group from STUDENT_TASK (if specified)
    # This must be done BEFORE environment creation
    # Even if TEACHER_TASK == STUDENT_TASK, we add policy_student to test if observations match
    if args_cli.student_task is not None:
        print(f"\n[INFO] Adding 'policy_student' observation group from task: {args_cli.student_task}")
        try:
            # Get student task spec and extract PolicyCfg
            student_task_spec = gym.spec(args_cli.student_task)
            student_env_cfg_entry_point = student_task_spec.kwargs.get("env_cfg_entry_point")
            if student_env_cfg_entry_point is not None:
                student_env_cfg = student_env_cfg_entry_point()
                if hasattr(student_env_cfg, 'observations') and hasattr(student_env_cfg.observations, 'policy'):
                    student_policy_cfg = student_env_cfg.observations.policy

                    # Only disable student noise if flag is set
                    # Teacher keeps its configured noise settings from the environment
                    if args_cli.disable_student_noise:
                        print(f"[INFO] --disable_student_noise flag set, disabling student noise")
                        if hasattr(student_policy_cfg, 'enable_corruption'):
                            student_policy_cfg.enable_corruption = False
                            print(f"[INFO] Disabled corruption/noise for policy_student group")
                    else:
                        print(f"[INFO] Student noise enabled (matches training distribution)")

                    # Add 'policy_student' to env_cfg.observations
                    if hasattr(env_cfg, 'observations'):
                        env_cfg.observations.policy_student = student_policy_cfg
                        print(f"[INFO] Successfully added 'policy_student' observation group to env_cfg")
                        print(f"[INFO] Student policy config: history_length={getattr(student_policy_cfg, 'history_length', 0)}")
        except Exception as e:
            print(f"[WARNING] Failed to add policy_student group: {e}")
            import traceback
            traceback.print_exc()

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env: ManagerBasedRLEnv = env.unwrapped
    command_term = base_env.command_manager.get_term("motion")
    if hasattr(command_term, "_resample_motions_every_steps"):
        command_term._resample_motions_every_steps = int(1e12)  # Prevent remapping

    # Clone metrics to avoid gradient issues
    with torch.inference_mode(False):
        for key, value in command_term.metrics.items():
            command_term.metrics[key] = value.clone()

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Load policy
    print("[INFO] Loading policy...")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Try to load checkpoint, if dimension mismatch occurs, load only actor weights
    try:
        ppo_runner.load(resume_path)
        print("[INFO] Policy loaded successfully")
    except RuntimeError as e:
        if "size mismatch" in str(e) and "critic" in str(e):
            print(f"[WARNING] Critic dimension mismatch: {e}")
            print("[INFO] Loading only actor weights (critic not needed for collection)")

            # Load checkpoint manually
            loaded_dict = torch.load(resume_path, weights_only=False)

            # Filter out critic weights
            actor_state_dict = {k: v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("actor")}

            # Load only actor weights (strict=False to ignore missing critic weights)
            ppo_runner.alg.policy.load_state_dict(actor_state_dict, strict=False)
            print("[INFO] Actor weights loaded successfully (critic skipped)")

            # CRITICAL: Load observation normalizer if available
            if ppo_runner.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
                ppo_runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                print("[INFO] Observation normalizer loaded successfully")
            else:
                print("[WARNING] No observation normalizer found in checkpoint, observations may be incorrect!")
        else:
            raise

    ppo_runner.eval_mode()

    # Get observation and action dimensions
    obs, extras = env.reset()
    obs_dim = obs.shape[-1]
    action_dim = env.num_actions

    print(f"[INFO] Teacher observation dimension: {obs_dim}")
    print(f"[INFO] Action dimension: {action_dim}")

    # Check if we have policy_student observation group
    has_policy_student = "policy_student" in base_env.observation_manager._group_obs_term_names

    if has_policy_student:
        print(f"\n[INFO] Environment has 'policy_student' observation group")
        # Get student obs dimension from environment
        obs_temp, extras_temp = env.reset()
        if "policy_student" in extras_temp.get("observations", {}):
            student_obs_dim = extras_temp["observations"]["policy_student"].shape[1]
            print(f"[INFO] Student observation dimension: {student_obs_dim}")
        else:
            student_obs_dim = None
    else:
        print(f"\n[INFO] No 'policy_student' group found, will use 'policy' for both teacher and student")
        student_obs_dim = None

    # Get motion information
    base_env = env.unwrapped
    command_term = base_env.command_manager.get_term("motion")
    motion_paths = command_term.motion_dir_loader.motion_paths
    start_index = max(args_cli.start_index, 0)
    end_index = args_cli.end_index if args_cli.end_index is not None else len(motion_paths)
    end_index = min(end_index, len(motion_paths))

    if start_index >= end_index:
        raise ValueError(f"Invalid motion index range: [{start_index}, {end_index})")

    selected_motion_count = end_index - start_index
    batch_size = env_cfg.scene.num_envs

    print(f"[INFO] Total motions available: {len(motion_paths)}")
    print(f"[INFO] Processing motions: {start_index} to {end_index} ({selected_motion_count} motions)")

    # Initialize trajectory collector
    max_trajectories = args_cli.max_trajectories if args_cli.max_trajectories else selected_motion_count
    collector = ExpertTrajectoryCollector(
        obs_dim=obs_dim,
        action_dim=action_dim,
        student_obs_dim=student_obs_dim,
        max_trajectories=max_trajectories,
        device=agent_cfg.device
    )

    print(f"[INFO] Trajectory collector initialized")
    print(f"  Max trajectories: {max_trajectories}")
    print(f"  Steps per trajectory: {args_cli.steps_per_env}")
    print(f"  Total steps to collect: {max_trajectories * args_cli.steps_per_env}")

    # Prepare motion queue
    motion_queue = list(range(start_index, end_index))
    robot = base_env.scene["robot"]

    def _assign_motions_to_envs(env_ids: list[int], motion_ids: list[int]):
        """Assign motions to environments (will be applied on next reset)."""
        if not env_ids:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=command_term.device, dtype=torch.long)
        motion_tensor = torch.as_tensor(motion_ids, device=command_term.device, dtype=torch.long)

        # Set motion indices - these will be used when environment resets
        command_term.env_motion_indices[env_ids_tensor] = motion_tensor
        command_term.time_steps[env_ids_tensor] = 0
        if hasattr(command_term, "motion_end_buf"):
            command_term.motion_end_buf[env_ids_tensor] = False

        # Note: Environment manages history reset automatically for all observation groups

    # Initial assignment
    # Assign motions to all environments
    initial_count = min(batch_size, len(motion_queue))
    if initial_count > 0:
        initial_envs = list(range(initial_count))
        initial_motions = [motion_queue.pop(0) for _ in range(initial_count)]
        _assign_motions_to_envs(initial_envs, initial_motions)

    # Get observations (like play.py - no reset!)
    # The environment is already initialized by gym.make()
    obs, extras = env.get_observations()

    # Track steps per environment
    env_step_counts = torch.zeros(batch_size, dtype=torch.long, device=command_term.device)
    trajectories_collected = 0

    print("\n" + "=" * 80)
    print("Starting trajectory collection...")
    print("=" * 80 + "\n")

    # Main collection loop
    with torch.inference_mode():
        # Continue until we collect enough trajectories
        # Even when motion_queue is empty, finish current trajectories
        while trajectories_collected < max_trajectories:
            # Compute current observations from teacher's policy group
            obs_dict_teacher = _compute_obs_dict(base_env, "policy")
            policy_term_names = base_env.observation_manager._group_obs_term_names["policy"]

            # Concatenate teacher observations using policy group order
            obs_components = [obs_dict_teacher[term_name] for term_name in policy_term_names if term_name in obs_dict_teacher]
            teacher_obs = torch.cat(obs_components, dim=-1)

            # Get student observations
            student_obs = None
            if has_policy_student:
                # Use environment's policy_student group (environment manages history)
                obs_dict_student = _compute_obs_dict(base_env, "policy_student")
                policy_student_term_names = base_env.observation_manager._group_obs_term_names["policy_student"]
                student_obs_components = [obs_dict_student[term_name] for term_name in policy_student_term_names if term_name in obs_dict_student]
                if student_obs_components:
                    student_obs = torch.cat(student_obs_components, dim=-1)
            else:
                # No policy_student group, use teacher obs as student obs
                student_obs = teacher_obs

            # Get action from policy
            actions, action_mean, action_sigma = _policy_step(ppo_runner, obs)

            # Collect state-action pair
            collector.add_step(
                observations=teacher_obs,
                actions=actions,
                action_mean=action_mean,
                action_sigma=action_sigma,
                student_observations=student_obs
            )

            # Environment step
            obs, _, dones, extras = env.step(actions)

            # Increment step counts for all environments
            env_step_counts += 1

            # Check which environments have reached steps_per_env
            completed_mask = env_step_counts >= args_cli.steps_per_env
            if completed_mask.any():
                completed_indices = completed_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                num_completed = len(completed_indices)

                # Mark trajectory ends
                collector.mark_trajectory_end(count=num_completed)
                trajectories_collected += num_completed

                # Assign new motions to completed environments
                new_envs = []
                new_motions = []
                for env_id in completed_indices:
                    env_step_counts[env_id] = 0  # Reset step count
                    if motion_queue and trajectories_collected < max_trajectories:
                        new_envs.append(env_id)
                        new_motions.append(motion_queue.pop(0))

                # Assign new motions (will be applied when env auto-resets)
                if new_envs:
                    _assign_motions_to_envs(new_envs, new_motions)

                # Manually reset these environments to apply new motions immediately
                # This ensures the robot starts from the new motion's initial pose
                if new_envs:
                    env_ids_to_reset = torch.tensor(new_envs, device=base_env.device, dtype=torch.long)
                    base_env.reset(env_ids=env_ids_to_reset.cpu().tolist())

                    # OPTIONAL: Skip a few frames after reset to let physics stabilize
                    # Uncomment if you see instability after reset
                    # for _ in range(3):
                    #     dummy_actions, _, _ = _policy_step(ppo_runner, obs)
                    #     obs, _, _, _ = env.step(dummy_actions)

            # Progress logging
            if trajectories_collected % 100 == 0 and trajectories_collected > 0:
                total_steps = trajectories_collected * args_cli.steps_per_env
                print(f"[PROGRESS] Trajectories: {trajectories_collected}/{max_trajectories}, "
                      f"Steps: {total_steps}, "
                      f"Queue: {len(motion_queue)}")

    print("\n" + "=" * 80)
    print("Collection complete!")
    print("=" * 80)
    print(f"  Total trajectories collected: {trajectories_collected}")
    print(f"  Total steps collected: {trajectories_collected * args_cli.steps_per_env}")
    print(f"  Steps per trajectory: {args_cli.steps_per_env}")

    # Get and display statistics
    stats = collector.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save trajectories
    print(f"\nSaving trajectories to: {args_cli.output_path}")
    output_path = pathlib.Path(args_cli.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collector.save(args_cli.output_path)

    print("\n" + "=" * 80)
    print("Expert trajectory collection completed successfully!")
    print(f"Data saved to: {args_cli.output_path}")
    print("=" * 80)

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
