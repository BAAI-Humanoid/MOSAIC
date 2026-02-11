"""Batch convert CSV motions in a folder to NPZ using the same pipeline as csv_to_npz.py.

This script launches one Omniverse/Isaac session, sets up the same robot/scene,
then iterates all matching CSV files in the input folder and writes NPZ files
with a user-specified prefix added to the base filename.
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Avoid binary incompatibility by preventing mixed user-site packages.

    Typical failure mode:
    - numpy is imported from ~/.local/... (user site)
    - then IsaacSim/Kit loads another bundled numpy component (pip_prebundle)
    - results in: ValueError: numpy.dtype size changed (binary incompatibility)

    We aggressively remove user-site paths from sys.path before importing isaaclab/isaacsim.
    """

    # Best-effort: also propagate to any subprocesses (doesn't affect current interpreter startup).
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site  # noqa: WPS433 (runtime import by design)

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(p: str) -> bool:
        if not p:
            return False
        # canonical user site
        if isinstance(user_site, str) and p == user_site:
            return True
        # common user-site patterns
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site_path(p)]

    # If numpy got imported earlier for any reason, force re-import after sanitization.
    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

import argparse
import os
from pathlib import Path
import numpy as np

from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Batch convert CSV motions in a folder to NPZ format.")
parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input CSV motion files.")
parser.add_argument("--file_pattern", type=str, default="*.csv", help="Glob pattern to match input files.")
parser.add_argument("--output_prefix", type=str, required=True, help='Prefix to add to output names, e.g. "g1_lafan".')
parser.add_argument("--output_dir", type=str, default=None, help="Output folder (default: same as input_dir).")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motions.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motions.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional global frame range applied to every file. Index starts from 1. If omitted, loads all frames.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot platform name (e.g. g1, h1_2). See whole_body_tracking.robots.robot_registry.ROBOT_PLATFORMS.",
)
parser.add_argument("--log_wandb", action="store_true", help="If set, log each NPZ to wandb as an artifact.")
parser.add_argument(
    "--min_duration",
    type=float,
    default=3.0,
    help="Minimum duration (seconds) for motions to be converted. Motions shorter than this will be skipped. Default: 3.0",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.robot_registry import available_robot_names, get_robot_platform


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


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.
        Args:
            rotations: shape (B, 4). dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def iter_states(self):
        """Yields states for the full motion length."""
        for _ in range(self.output_frames):
            state = (
                self.motion_base_poss[self.current_idx : self.current_idx + 1],
                self.motion_base_rots[self.current_idx : self.current_idx + 1],
                self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
                self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
                self.motion_dof_poss[self.current_idx : self.current_idx + 1],
                self.motion_dof_vels[self.current_idx : self.current_idx + 1],
            )
            self.current_idx += 1
            yield state


def setup_sim_and_scene(robot_cfg: ArticulationCfg):
    """Setup simulation and scene with one robot."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg_class = create_replay_scene_cfg(robot_cfg)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    return sim, scene


def process_one_file(
    sim: SimulationContext,
    scene: InteractiveScene,
    csv_path: str,
    output_npz_path: str,
    output_name: str,
    joint_names: list[str],
) -> float | None:
    """Replay one CSV and save NPZ (and optionally log to wandb).
    
    Returns:
        Motion duration in seconds if successful, None if skipped (too short).
    """
    # robot and joint indices
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # motion loader
    motion = MotionLoader(
        motion_file=csv_path,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
    )
    
    # Check duration threshold
    if motion.duration < args_cli.min_duration:
        print(f"[SKIP]: Motion duration ({motion.duration:.2f}s) < min_duration ({args_cli.min_duration:.2f}s), skipping: {csv_path}")
        return None

    # logger buffers
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # iterate the whole motion once
    for (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
    ) in motion.iter_states():
        # root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # no physics stepping; only render and update scene
        sim.render()
        scene.update(sim.get_physics_dt())

        # record one frame
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

    # stack and save
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
    np.savez(output_npz_path, **log)
    print(f"[INFO]: Saved NPZ: {output_npz_path}")

    if args_cli.log_wandb:
        import wandb

        COLLECTION = output_name
        run = wandb.init(project="csv_to_npz", name=COLLECTION, reinit=True)
        print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
        REGISTRY = "motions"
        logged_artifact = run.log_artifact(artifact_or_path=output_npz_path, name=COLLECTION, type=REGISTRY)
        run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
        print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
        run.finish()
    
    return motion.duration


def main():
    # Resolve robot platform from registry
    try:
        robot_platform = get_robot_platform(args_cli.robot)
    except KeyError as e:
        raise SystemExit(f"{e}\nAvailable robots: {available_robot_names()}") from None

    input_dir = Path(args_cli.input_dir).expanduser().resolve()
    pattern = args_cli.file_pattern
    output_dir = Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.rglob(pattern))
    if len(csv_files) == 0:
        print(f"[WARN]: No files matched under {input_dir} with pattern '{pattern}'")
        return

    sim, scene = setup_sim_and_scene(robot_platform.cfg)
    durations: list[float] = []
    skipped_count = 0
    try:
        all_files_length = len(csv_files)
        for i, csv_path in enumerate(tqdm(csv_files, desc="Converting motions", unit="file")):
            csv_path = Path(csv_path)
            relative_csv = csv_path.relative_to(input_dir)
            output_name = f"{args_cli.output_prefix}_{relative_csv.stem}"
            output_npz_path = (output_dir / relative_csv).with_name(f"{output_name}.npz")
            output_npz_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO]: Converting {i+1}/{all_files_length}: {csv_path} -> {output_npz_path}")
            duration = process_one_file(sim, scene, str(csv_path), str(output_npz_path), output_name, robot_platform.joint_names)
            if duration is not None:
                durations.append(duration)
            else:
                skipped_count += 1
        print(f"[INFO]: Skipped {skipped_count} motions (duration < {args_cli.min_duration}s)")
    finally:
        # Print statistics before closing simulator (simulation_app.close() may exit the program)
        print("\n" + "=" * 80)
        print("CONVERSION STATISTICS")
        print("=" * 80)
        print(f"Total files processed: {all_files_length}")
        print(f"Successfully converted: {len(durations)}")
        print(f"Skipped (duration < {args_cli.min_duration}s): {skipped_count}")
        
        if len(durations) > 0:
            durations_array = np.array(durations)
            mean_duration = float(np.mean(durations_array))
            std_duration = float(np.std(durations_array))
            median_duration = float(np.median(durations_array))
            min_duration = float(np.min(durations_array))
            max_duration = float(np.max(durations_array))
            
            print(f"\nMotion Duration Statistics (seconds):")
            print(f"  Mean:   {mean_duration:.3f}")
            print(f"  Std:    {std_duration:.3f}")
            print(f"  Median: {median_duration:.3f}")
            print(f"  Min:    {min_duration:.3f}")
            print(f"  Max:    {max_duration:.3f}")
        else:
            print("\nNo motions were successfully converted.")
        print("=" * 80)
        
        # close sim app (this may exit the program, so do it last)
        simulation_app.close()


if __name__ == "__main__":
    main()


