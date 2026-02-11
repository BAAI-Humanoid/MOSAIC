"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage - Single motion file
    python replay_npz.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz

    # Usage - Motion folder
    python replay_npz.py --motion_folder source/whole_body_tracking/whole_body_tracking/assets/g1/motions/

    # Usage - Motion folder with custom glob pattern
    python replay_npz.py --motion_folder /path/to/motions --file_glob "walk_*.npz"
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Avoid binary incompatibility by preventing mixed user-site packages.

    See scripts/batch_csv_to_npz.py for the detailed rationale.
    """

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site  # noqa: WPS433 (runtime import by design)

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(p: str) -> bool:
        if not p:
            return False
        if isinstance(user_site, str) and p == user_site:
            return True
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site_path(p)]

    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, default=None, help="the path to the motion file.")
parser.add_argument("--motion_folder", type=str, default=None, help="the path to the motion folder containing multiple .npz files.")
parser.add_argument("--file_glob", type=str, default="*.npz", help="glob pattern for motion files when using --motion_folder.")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot platform name used to replay this motion (e.g. g1, h1_2).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# validate arguments
if args_cli.motion_file is None and args_cli.motion_folder is None:
    parser.error("Either --motion_file or --motion_folder must be specified.")
if args_cli.motion_file is not None and args_cli.motion_folder is not None:
    parser.error("Cannot specify both --motion_file and --motion_folder at the same time.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Robot platform registry
##
from whole_body_tracking.robots.robot_registry import available_robot_names, get_robot_platform
from whole_body_tracking.tasks.tracking.mdp import MotionLoader, MultiMotionLoader


def create_replay_scene_cfg(robot_cfg: ArticulationCfg):
    """Create a scene configuration for replaying motions with the specified robot."""

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Setup keyboard interface
    appwindow = carb.get_framework().acquire_interface(carb.input.IInput)
    keyboard = appwindow.get_keyboard()

    # Load motion(s) based on command line arguments
    if args_cli.motion_file is not None:
        # Single motion file mode
        motion_file = args_cli.motion_file
        motion = MotionLoader(
            motion_file,
            torch.tensor([0], dtype=torch.long, device=sim.device),
            sim.device,
        )
        time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

        print("\n" + "="*60)
        print("Single Motion Mode - No keyboard controls available")
        print("="*60 + "\n")

        # Simulation loop for single motion
        while simulation_app.is_running():
            time_steps += 1
            reset_ids = time_steps >= motion.time_step_total
            time_steps[reset_ids] = 0

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
            root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_dt)

            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

    else:
        # Multi-motion folder mode
        motion_folder = args_cli.motion_folder

        # Load all motions from the folder
        multi_motion = MultiMotionLoader(
            motion_folder,
            body_indexes=[0],  # Only need root body for replay
            device=sim.device,
            file_glob=args_cli.file_glob,
        )

        num_motions = len(multi_motion)
        print(f"[replay_npz] Loaded {num_motions} motion files from {motion_folder}")
        for i, path in enumerate(multi_motion.motion_paths):
            print(f"  [{i}] {path}")

        # Track current motion index and time step for each environment
        env_motion_indices = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
        time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

        # State for keyboard controls
        current_motion_idx = 0  # Global current motion for manual switching
        auto_cycle = True  # Auto cycle through motions when they finish
        paused = False  # Pause playback

        # Assign initial motions to environments
        for env_id in range(scene.num_envs):
            env_motion_indices[env_id] = current_motion_idx

        print(f"\n[replay_npz] Replaying motions in {scene.num_envs} environment(s)")
        print("\n" + "="*60)
        print("Keyboard Controls:")
        print("  N / RIGHT ARROW  : Next motion")
        print("  P / LEFT ARROW   : Previous motion")
        print("  R                : Restart current motion")
        print("  A                : Toggle auto-cycle mode")
        print("  SPACE            : Pause/Resume playback")
        print("  0-9              : Jump to motion index (0-9)")
        print("="*60)
        print(f"\nCurrently playing: Motion {current_motion_idx}/{num_motions-1}")
        print(f"Auto-cycle: {'ON' if auto_cycle else 'OFF'}")
        print(f"Status: {'Playing' if not paused else 'PAUSED'}\n")

        # Track key states to prevent rapid toggling
        key_states = {
            carb.input.KeyboardInput.N: False,
            carb.input.KeyboardInput.P: False,
            carb.input.KeyboardInput.R: False,
            carb.input.KeyboardInput.A: False,
            carb.input.KeyboardInput.SPACE: False,
            carb.input.KeyboardInput.RIGHT: False,
            carb.input.KeyboardInput.LEFT: False,
        }

        # Simulation loop for multiple motions
        while simulation_app.is_running():
            # Handle keyboard input
            prev_motion_idx = current_motion_idx
            prev_auto_cycle = auto_cycle
            prev_paused = paused

            # Check for next motion (N or RIGHT)
            if keyboard.get_key_state(carb.input.KeyboardInput.N) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.N]:
                key_states[carb.input.KeyboardInput.N] = True
                current_motion_idx = (current_motion_idx + 1) % num_motions
                time_steps[:] = 0
            elif keyboard.get_key_state(carb.input.KeyboardInput.N) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.N] = False

            if keyboard.get_key_state(carb.input.KeyboardInput.RIGHT) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.RIGHT]:
                key_states[carb.input.KeyboardInput.RIGHT] = True
                current_motion_idx = (current_motion_idx + 1) % num_motions
                time_steps[:] = 0
            elif keyboard.get_key_state(carb.input.KeyboardInput.RIGHT) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.RIGHT] = False

            # Check for previous motion (P or LEFT)
            if keyboard.get_key_state(carb.input.KeyboardInput.P) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.P]:
                key_states[carb.input.KeyboardInput.P] = True
                current_motion_idx = (current_motion_idx - 1) % num_motions
                time_steps[:] = 0
            elif keyboard.get_key_state(carb.input.KeyboardInput.P) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.P] = False

            if keyboard.get_key_state(carb.input.KeyboardInput.LEFT) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.LEFT]:
                key_states[carb.input.KeyboardInput.LEFT] = True
                current_motion_idx = (current_motion_idx - 1) % num_motions
                time_steps[:] = 0
            elif keyboard.get_key_state(carb.input.KeyboardInput.LEFT) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.LEFT] = False

            # Check for restart (R)
            if keyboard.get_key_state(carb.input.KeyboardInput.R) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.R]:
                key_states[carb.input.KeyboardInput.R] = True
                time_steps[:] = 0
            elif keyboard.get_key_state(carb.input.KeyboardInput.R) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.R] = False

            # Check for auto-cycle toggle (A)
            if keyboard.get_key_state(carb.input.KeyboardInput.A) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.A]:
                key_states[carb.input.KeyboardInput.A] = True
                auto_cycle = not auto_cycle
            elif keyboard.get_key_state(carb.input.KeyboardInput.A) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.A] = False

            # Check for pause toggle (SPACE)
            if keyboard.get_key_state(carb.input.KeyboardInput.SPACE) == carb.input.KeyboardEventType.KEY_PRESS and not key_states[carb.input.KeyboardInput.SPACE]:
                key_states[carb.input.KeyboardInput.SPACE] = True
                paused = not paused
            elif keyboard.get_key_state(carb.input.KeyboardInput.SPACE) == carb.input.KeyboardEventType.KEY_RELEASE:
                key_states[carb.input.KeyboardInput.SPACE] = False

            # Check for number keys (0-9)
            for num in range(min(10, num_motions)):
                key = getattr(carb.input.KeyboardInput, f"KEY_{num}")
                if keyboard.get_key_state(key) == carb.input.KeyboardEventType.KEY_PRESS:
                    if num < num_motions:
                        current_motion_idx = num
                        time_steps[:] = 0

            # Print status if changed
            if current_motion_idx != prev_motion_idx or auto_cycle != prev_auto_cycle or paused != prev_paused:
                print(f"\rMotion: {current_motion_idx}/{num_motions-1} | Auto-cycle: {'ON' if auto_cycle else 'OFF'} | {'Playing' if not paused else 'PAUSED'}  ", end="")

            # Update motion assignment
            env_motion_indices[:] = current_motion_idx

            # Update time steps only if not paused
            if not paused:
                time_steps += 1

                # Check for motion completion
                motion_length = multi_motion.motion_length(current_motion_idx)
                if time_steps[0] >= motion_length:
                    if auto_cycle:
                        # Auto cycle to next motion
                        current_motion_idx = (current_motion_idx + 1) % num_motions
                        time_steps[:] = 0
                    else:
                        # Loop current motion
                        time_steps[:] = 0

            # Gather motion data for current timesteps
            joint_pos = multi_motion.gather("joint_pos", env_motion_indices, time_steps, sim.device)
            joint_vel = multi_motion.gather("joint_vel", env_motion_indices, time_steps, sim.device)
            body_pos_w = multi_motion.gather("body_pos_w", env_motion_indices, time_steps, sim.device)
            body_quat_w = multi_motion.gather("body_quat_w", env_motion_indices, time_steps, sim.device)
            body_lin_vel_w = multi_motion.gather("body_lin_vel_w", env_motion_indices, time_steps, sim.device)
            body_ang_vel_w = multi_motion.gather("body_ang_vel_w", env_motion_indices, time_steps, sim.device)

            # Update robot states
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = body_pos_w[:, 0] + scene.env_origins
            root_states[:, 3:7] = body_quat_w[:, 0]
            root_states[:, 7:10] = body_lin_vel_w[:, 0]
            root_states[:, 10:] = body_ang_vel_w[:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_dt)

            # Follow first environment
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    # Resolve robot platform from registry
    try:
        robot_platform = get_robot_platform(args_cli.robot)
    except KeyError as e:
        raise SystemExit(f"{e}\nAvailable robots: {available_robot_names()}") from None

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg_class = create_replay_scene_cfg(robot_platform.cfg)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
