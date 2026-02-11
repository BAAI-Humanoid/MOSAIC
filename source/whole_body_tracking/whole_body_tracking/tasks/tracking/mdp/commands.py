from __future__ import annotations

import math
import numpy as np
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_rank_world_size(shard_by: str = "global") -> tuple[int, int]:
    """Best-effort (rank, world_size) for multi-GPU dataset sharding.

    Priority:
    1) torch.distributed (if initialized)
    2) environment variables (torchrun/accelerate/slurm compatible)
    """

    shard_by = str(shard_by).lower()
    if shard_by not in {"global", "local"}:
        raise ValueError(f"Invalid shard_by={shard_by!r}. Expected 'global' or 'local'.")

    # torch.distributed path
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
            return rank, world_size
    except Exception:
        pass

    # env var path
    def _get_int_env(name: str, default: int) -> int:
        val = os.getenv(name, "")
        if val == "":
            return default
        try:
            return int(val)
        except Exception:
            return default

    if shard_by == "local":
        rank = _get_int_env("LOCAL_RANK", 0)
        world_size = _get_int_env("LOCAL_WORLD_SIZE", _get_int_env("WORLD_SIZE", 1))
    else:
        rank = _get_int_env("RANK", _get_int_env("LOCAL_RANK", 0))
        world_size = _get_int_env("WORLD_SIZE", _get_int_env("LOCAL_WORLD_SIZE", 1))

    world_size = max(int(world_size), 1)
    rank = int(rank) % world_size
    return rank, world_size


def _select_motion_paths_for_rank(
    motion_paths: list[str],
    *,
    max_motions: int | None,
    shard_across_gpus: bool,
    shard_by: str,
    shard_seed: int,
    shard_strategy: str = "chunk",
) -> tuple[list[str], dict[str, int | str | bool]]:
    """Select a (possibly sharded) subset of motion_paths for the current process.

    Design goals:
    - When motions are plentiful and max_motions is small (e.g. <= num_envs), enable
      deterministic *disjoint* subsets across GPUs (when possible), to reduce duplicates.
    - Keep default behavior unchanged unless shard_across_gpus or max_motions is provided.
    """

    total = len(motion_paths)
    info: dict[str, int | str | bool] = {
        "total_motions": total,
        "selected_motions": total,
        "shard_across_gpus": bool(shard_across_gpus),
        "shard_by": str(shard_by),
        "shard_seed": int(shard_seed),
        "shard_strategy": str(shard_strategy),
        "rank": 0,
        "world_size": 1,
        "max_motions": int(max_motions) if max_motions is not None else -1,
    }

    if total == 0:
        return motion_paths, info

    if max_motions is None:
        # No selection requested.
        return motion_paths, info

    max_motions = int(max_motions)
    if max_motions <= 0:
        raise ValueError("max_motions must be a positive integer when provided.")

    rank, world_size = _get_rank_world_size(shard_by=shard_by)
    info["rank"] = rank
    info["world_size"] = world_size

    # Deterministic shuffle of paths to avoid correlated filesystem ordering.
    indices = list(range(total))
    rng = random.Random(int(shard_seed))
    rng.shuffle(indices)

    # If not sharding, just take the first max_motions after shuffle.
    if (not shard_across_gpus) or (world_size <= 1):
        selected_idx = indices[: min(max_motions, total)]
        selected = [motion_paths[i] for i in selected_idx]
        info["selected_motions"] = len(selected)
        return selected, info

    # Sharded selection:
    # If dataset is large enough, create disjoint fixed-size shards (best case).
    if total >= world_size * max_motions:
        start = rank * max_motions
        selected_idx = indices[start : start + max_motions]
        selected = [motion_paths[i] for i in selected_idx]
        info["selected_motions"] = len(selected)
        return selected, info

    # Otherwise, fall back to disjoint partitioning then (optionally) cap.
    shard_strategy = str(shard_strategy).lower()
    if shard_strategy not in {"chunk", "stride"}:
        raise ValueError(f"Invalid shard_strategy={shard_strategy!r}. Expected 'chunk' or 'stride'.")

    if shard_strategy == "stride":
        shard_idx = indices[rank::world_size]
    else:
        # chunk: contiguous chunks after shuffle
        chunk_size = int(math.ceil(total / float(world_size)))
        start = rank * chunk_size
        shard_idx = indices[start : start + chunk_size]

    if len(shard_idx) > max_motions:
        shard_idx = shard_idx[:max_motions]

    selected = [motion_paths[i] for i in shard_idx]
    info["selected_motions"] = len(selected)
    return selected, info


def _maybe_log_motion_shard_to_wandb_summary(
    shard_info: dict[str, int | str | bool], cfg: "MultiMotionCommandCfg"
) -> None:
    """Best-effort: log one-time shard info to Weights&Biases summary.

    Intended behavior:
    - If torch.distributed is initialized, gather per-rank loaded counts and write summary from rank0 only.
    - If wandb is not available or not initialized, do nothing.
    """

    if not getattr(cfg, "motion_dataset_log_wandb_summary", True):
        return

    # Import wandb lazily and safely.
    try:
        import wandb  # type: ignore

        run = getattr(wandb, "run", None)
        if run is None:
            return
    except Exception:
        return

    rank = int(shard_info.get("rank", 0)) if isinstance(shard_info.get("rank", 0), (int, float)) else 0
    world_size = (
        int(shard_info.get("world_size", 1)) if isinstance(shard_info.get("world_size", 1), (int, float)) else 1
    )
    total = (
        int(shard_info.get("total_motions", 0)) if isinstance(shard_info.get("total_motions", 0), (int, float)) else 0
    )
    loaded = (
        int(shard_info.get("selected_motions", 0))
        if isinstance(shard_info.get("selected_motions", 0), (int, float))
        else 0
    )

    # Try distributed gather for per-rank reporting.
    loaded_list: list[int] | None = None
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.tensor([loaded], dtype=torch.long, device=device)
            out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(out, t)
            loaded_list = [int(x.item()) for x in out]
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:
        loaded_list = None

    # Only rank0 writes summary to avoid collisions.
    if rank != 0:
        return

    # Stable keys for easy browsing.
    run.summary["motion_dataset/world_size"] = int(world_size)
    run.summary["motion_dataset/total_motions_seen_by_rank0"] = int(total)
    run.summary["motion_dataset/shard_enabled"] = bool(getattr(cfg, "motion_dataset_shard_across_gpus", False))
    run.summary["motion_dataset/shard_by"] = str(getattr(cfg, "motion_dataset_shard_by", "global"))
    run.summary["motion_dataset/shard_strategy"] = str(getattr(cfg, "motion_dataset_shard_strategy", "chunk"))
    run.summary["motion_dataset/shard_seed"] = int(getattr(cfg, "motion_dataset_shard_seed", 0))
    run.summary["motion_dataset/load_cap"] = (
        int(getattr(cfg, "motion_dataset_load_cap", -1))
        if getattr(cfg, "motion_dataset_load_cap", None) is not None
        else None
    )

    if loaded_list is not None:
        # Store as a compact string to avoid schema issues.
        run.summary["motion_dataset/loaded_motions_per_rank"] = str(loaded_list)
        run.summary["motion_dataset/loaded_motions_sum"] = int(sum(loaded_list))
        run.summary["motion_dataset/loaded_motions_min"] = int(min(loaded_list)) if len(loaded_list) > 0 else 0
        run.summary["motion_dataset/loaded_motions_max"] = int(max(loaded_list)) if len(loaded_list) > 0 else 0
    else:
        run.summary["motion_dataset/loaded_motions_rank0"] = int(loaded)


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.start_from_beginning:
            start_frame = max(int(self.cfg.start_frame), 0)
            start_frame = min(start_frame, max(self.motion.time_step_total - 1, 0))
            self.time_steps[env_ids] = start_frame
        else:
            self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    start_from_beginning: bool = False
    start_frame: int = 0

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class MultiMotionLoader:
    """Preload motion files into contiguous tensors for fast batched sampling."""

    def __init__(
        self,
        motion_dir: str,
        body_indexes: Sequence[int],
        device: str | torch.device = "cpu",
        file_glob: str = "*.npz",
        storage_device: str | torch.device | None = None,
        *,
        max_motions: int | None = None,
        shard_across_gpus: bool = False,
        shard_by: str = "global",
        shard_seed: int = 0,
        shard_strategy: str = "chunk",
        motion_groups: dict[str, list[str]] | None = None,
    ):
        motion_dir_path = Path(motion_dir).expanduser().resolve()
        assert motion_dir_path.is_dir(), f"Invalid directory path: {motion_dir}"
        all_motion_paths = sorted(str(path) for path in motion_dir_path.rglob(file_glob) if path.is_file())
        assert len(all_motion_paths) > 0, f"No motion files matched in: {motion_dir} with pattern: {file_glob}"
        # Group-aware sharding: keep every GPU seeing every group when possible.
        rank, world_size = _get_rank_world_size(shard_by=shard_by)

        def _assign_group_name(motion_path: str) -> str:
            if motion_groups is None:
                return "default"
            for group_name, folder_patterns in motion_groups.items():
                for pattern in folder_patterns:
                    if pattern in motion_path:
                        return group_name
            return "default"

        use_group_sharding = motion_groups is not None and (shard_across_gpus or max_motions is not None)
        if use_group_sharding:
            group_to_paths: dict[str, list[str]] = {}
            for motion_path in all_motion_paths:
                group_name = _assign_group_name(motion_path)
                group_to_paths.setdefault(group_name, []).append(motion_path)

            # Validate configured groups exist in the dataset.
            missing_groups = []
            for group_name in motion_groups.keys():
                if len(group_to_paths.get(group_name, [])) == 0:
                    missing_groups.append(group_name)
            if missing_groups:
                raise ValueError(
                    "No motions matched for motion_groups: "
                    f"{missing_groups}. Check motion_groups patterns or dataset layout."
                )

            nonempty_groups = [g for g, paths in group_to_paths.items() if len(paths) > 0]
            num_groups = len(nonempty_groups)

            if max_motions is not None:
                max_motions = int(max_motions)
                if max_motions < num_groups:
                    raise ValueError(
                        f"motion_dataset_load_cap={max_motions} is smaller than the number of "
                        f"non-empty groups ({num_groups}). Increase the cap to ensure every group is loaded."
                    )

                total_paths = sum(len(group_to_paths[g]) for g in nonempty_groups)
                # Start with 1 per group to guarantee coverage.
                group_caps = {g: 1 for g in nonempty_groups}
                remaining = max_motions - num_groups
                if remaining > 0 and total_paths > 0:
                    # Distribute remaining capacity proportionally by group size.
                    extras = {}
                    for g in nonempty_groups:
                        extras[g] = int(math.floor(remaining * len(group_to_paths[g]) / total_paths))
                    used = sum(extras.values())
                    leftover = remaining - used
                    # Assign leftover one by one to groups with available capacity.
                    for g in nonempty_groups:
                        if leftover <= 0:
                            break
                        extras[g] += 1
                        leftover -= 1
                    for g in nonempty_groups:
                        group_caps[g] = min(len(group_to_paths[g]), group_caps[g] + extras[g])
            else:
                group_caps = {g: None for g in nonempty_groups}

            selected_paths = []
            shard_info = {
                "total_motions": len(all_motion_paths),
                "selected_motions": 0,
                "shard_across_gpus": bool(shard_across_gpus),
                "shard_by": str(shard_by),
                "shard_seed": int(shard_seed),
                "shard_strategy": str(shard_strategy),
                "rank": int(rank),
                "world_size": int(world_size),
                "max_motions": int(max_motions) if max_motions is not None else -1,
                "group_mode": "per_group",
                "group_shards": {},
            }

            for idx, group_name in enumerate(nonempty_groups):
                group_paths = group_to_paths[group_name]
                group_cap = group_caps[group_name]
                # If group is smaller than world size, avoid empty ranks by disabling sharding for this group.
                group_shard_across = shard_across_gpus
                if group_shard_across and world_size > 1 and len(group_paths) < world_size:
                    group_shard_across = False

                group_selected, group_info = _select_motion_paths_for_rank(
                    group_paths,
                    max_motions=group_cap,
                    shard_across_gpus=group_shard_across,
                    shard_by=shard_by,
                    shard_seed=int(shard_seed) + (idx + 1) * 10007,
                    shard_strategy=shard_strategy,
                )
                shard_info["group_shards"][group_name] = group_info
                selected_paths.extend(group_selected)

            shard_info["selected_motions"] = len(selected_paths)
        else:
            selected_paths, shard_info = _select_motion_paths_for_rank(
                all_motion_paths,
                max_motions=max_motions,
                shard_across_gpus=shard_across_gpus,
                shard_by=shard_by,
                shard_seed=shard_seed,
                shard_strategy=shard_strategy,
            )
        # Expose for debugging/analysis
        self.motion_paths_all = all_motion_paths
        self.motion_paths = selected_paths
        self.shard_info = shard_info

        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device

        body_idx_tensor = torch.as_tensor(body_indexes, dtype=torch.long, device="cpu")
        if body_idx_tensor.ndim != 1:
            raise ValueError("body_indexes must be a 1D sequence of indices.")
        body_idx_np = body_idx_tensor.cpu().numpy()

        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_list: list[torch.Tensor] = []
        body_quat_list: list[torch.Tensor] = []
        body_lin_vel_list: list[torch.Tensor] = []
        body_ang_vel_list: list[torch.Tensor] = []
        lengths: list[int] = []
        fps_list: list[float] = []

        for motion_path in self.motion_paths:
            with np.load(motion_path) as data:
                fps_value = float(np.asarray(data["fps"]).reshape(-1)[0])
                fps_list.append(fps_value)

                joint_pos_tensor = torch.from_numpy(np.asarray(data["joint_pos"], dtype=np.float32)).to(
                    self.storage_device
                )
                joint_vel_tensor = torch.from_numpy(np.asarray(data["joint_vel"], dtype=np.float32)).to(
                    self.storage_device
                )

                body_pos_tensor = torch.from_numpy(
                    np.asarray(data["body_pos_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_quat_tensor = torch.from_numpy(
                    np.asarray(data["body_quat_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_lin_vel_tensor = torch.from_numpy(
                    np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_ang_vel_tensor = torch.from_numpy(
                    np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)

                joint_pos_list.append(joint_pos_tensor)
                joint_vel_list.append(joint_vel_tensor)
                body_pos_list.append(body_pos_tensor)
                body_quat_list.append(body_quat_tensor)
                body_lin_vel_list.append(body_lin_vel_tensor)
                body_ang_vel_list.append(body_ang_vel_tensor)
                lengths.append(joint_pos_tensor.shape[0])

        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self.body_pos_w = torch.cat(body_pos_list, dim=0)
        self.body_quat_w = torch.cat(body_quat_list, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_list, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_list, dim=0)

        # Pin memory only when tensors are on CPU (GPU tensors cannot be pinned).
        if self.storage_device.type == "cpu":
            self.joint_pos = self.joint_pos.pin_memory()
            self.joint_vel = self.joint_vel.pin_memory()
            self.body_pos_w = self.body_pos_w.pin_memory()
            self.body_quat_w = self.body_quat_w.pin_memory()
            self.body_lin_vel_w = self.body_lin_vel_w.pin_memory()
            self.body_ang_vel_w = self.body_ang_vel_w.pin_memory()

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_lengths = lengths_tensor
        self.motion_offsets = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.total_frames = int(lengths_tensor.sum().item())

        # Build motion-to-group mapping for multi-teacher support
        self.motion_to_group: dict[int, str] = {}

        if motion_groups is not None:
            # Map each motion to its group based on path patterns
            for motion_idx, motion_path in enumerate(self.motion_paths):
                group_assigned = False
                # Check each group's folder patterns
                for group_name, folder_patterns in motion_groups.items():
                    for pattern in folder_patterns:
                        if pattern in motion_path:
                            self.motion_to_group[motion_idx] = group_name
                            group_assigned = True
                            break
                    if group_assigned:
                        break

                # If no match, assign to default group
                if not group_assigned:
                    self.motion_to_group[motion_idx] = "default"
        else:
            # If no groups specified, all motions belong to default group
            for motion_idx in range(len(self.motion_paths)):
                self.motion_to_group[motion_idx] = "default"

        # Print motion group distribution
        from collections import Counter
        group_counts = Counter(self.motion_to_group.values())
        print(f"[MultiMotionLoader] Motion group distribution:")
        for group_name in sorted(group_counts.keys()):
            count = group_counts[group_name]
            print(f"  - {group_name}: {count} motions")

    def __len__(self) -> int:
        return len(self.motion_paths)

    def motion_length(self, motion_index: int) -> int:
        return int(self.motion_lengths[motion_index].item())

    def compute_global_indices(self, motion_indices: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        lengths = self.motion_lengths[motion_indices]
        max_valid = torch.clamp(lengths - 1, min=0)
        clamped = torch.minimum(frame_indices, max_valid)
        offsets = self.motion_offsets[motion_indices]
        return offsets + clamped

    def gather_from_global(
        self, attr: str, global_indices: torch.Tensor, out_device: torch.device | str
    ) -> torch.Tensor:
        source_tensor = getattr(self, attr)
        if not isinstance(out_device, torch.device):
            out_device = torch.device(out_device)
        if global_indices.device != source_tensor.device:
            local_indices = global_indices.to(source_tensor.device)
        else:
            local_indices = global_indices
        gathered = source_tensor.index_select(0, local_indices)
        if gathered.device != out_device:
            gathered = gathered.to(out_device, non_blocking=True)
        return gathered

    def gather(
        self,
        attr: str,
        motion_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        out_device: torch.device | str,
    ) -> torch.Tensor:
        global_indices = self.compute_global_indices(motion_indices, frame_indices)
        return self.gather_from_global(attr, global_indices, out_device)


class MultiMotionCommand(CommandTerm):
    """Command term that supports loading and training with multiple motions from a folder.
    
    - Samples motions across files according to difficulty-based and novelty-based sampling.
    - Within each motion, time-step sampling is adaptive based on failure counts (same as single-motion logic).
    - Periodically remaps environment ids to a fresh set of motions so all motions get chances to be sampled.
    """

    cfg: "MultiMotionCommandCfg"

    def __init__(self, cfg: "MultiMotionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        body_index_array = self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0]
        body_index_tensor = torch.as_tensor(body_index_array, dtype=torch.long)
        self.body_indexes = body_index_tensor.to(self.device)
        body_index_list = body_index_tensor.cpu().tolist()

        preload_device = (
            torch.device(self.cfg.motion_preload_device)
            if self.cfg.motion_preload_device is not None
            else torch.device(self.device)
        )

        keys = ["x","y","z","roll","pitch","yaw"]
        self._pose_ranges = torch.tensor([self.cfg.pose_range.get(k,(0.,0.)) for k in keys],
                                        device=self.device, dtype=torch.float32)
        self._vel_ranges  = torch.tensor([self.cfg.velocity_range.get(k,(0.,0.)) for k in keys],
                                        device=self.device, dtype=torch.float32)

        # Optional: cap number of motions loaded per process (useful with large datasets).
        # Auto cap is only enabled when sharding is enabled and user doesn't specify a cap.
        load_cap = self.cfg.motion_dataset_load_cap
        if load_cap is None and self.cfg.motion_dataset_shard_across_gpus:
            candidates = [int(self.num_envs)]
            k_cfg = getattr(self.cfg, "max_active_motions", None)
            if k_cfg is not None:
                candidates.append(int(k_cfg))
            load_cap = int(min(candidates)) if len(candidates) > 0 else None

        self.motion_dir_loader = MultiMotionLoader(
            self.cfg.motion,
            body_index_list,
            device=self.device,
            file_glob=self.cfg.file_glob,
            storage_device=preload_device,
            max_motions=load_cap,
            shard_across_gpus=self.cfg.motion_dataset_shard_across_gpus,
            shard_by=self.cfg.motion_dataset_shard_by,
            shard_seed=self.cfg.motion_dataset_shard_seed,
            shard_strategy=self.cfg.motion_dataset_shard_strategy,
            motion_groups=self.cfg.motion_groups,
        )

        self.num_motions_total = len(self.motion_dir_loader)

        # ---- dataset shard one-time logging ----
        self._motion_dataset_shard_info = getattr(self.motion_dir_loader, "shard_info", {}) or {}
        if getattr(self.cfg, "motion_dataset_log_shard_info", False):
            print(f"[MultiMotionLoader] shard_info={self._motion_dataset_shard_info}")
        _maybe_log_motion_shard_to_wandb_summary(self._motion_dataset_shard_info, self.cfg)

        self.sim_dt = env.cfg.decimation * env.cfg.sim.dt
        self.frames_per_bin = max(1, int(round(1.0 / self.sim_dt)))

        self.motion_lengths = self.motion_dir_loader.motion_lengths.to(self.device)
        self.motion_lengths_minus_one = (self.motion_lengths - 1).clamp(min=0)
        self.motion_length_denominator = self.motion_lengths.clamp(min=1)

        self.motion_bin_counts = (self.motion_lengths // self.frames_per_bin) + 1
        self.motion_bin_counts_float = self.motion_bin_counts.to(torch.float32)
        self.max_bin_count = int(self.motion_bin_counts.max().item())
        self.bin_index_range = torch.arange(self.max_bin_count, device=self.device)
        self.motion_bin_mask = self.bin_index_range.unsqueeze(0) < self.motion_bin_counts.unsqueeze(1)
        self.motion_end_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.bin_failed_count = torch.zeros(
            self.num_motions_total, self.max_bin_count, dtype=torch.float32, device=self.device
        )
        self.current_bin_failed = torch.zeros_like(self.bin_failed_count)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Track motion groups for multi-teacher support
        self.env_motion_groups = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Create group name to index mappings
        unique_groups = set(self.motion_dir_loader.motion_to_group.values())
        self.group_name_to_idx: dict[str, int] = {}
        self.idx_to_group_name: dict[int, str] = {}
        for idx, group_name in enumerate(sorted(unique_groups)):
            self.group_name_to_idx[group_name] = idx
            self.idx_to_group_name[idx] = group_name

        print(f"[MultiMotionCommand] Registered motion groups: {list(self.group_name_to_idx.keys())}")

        # Build reverse mapping: group_name -> list of motion indices
        self.group_to_motions: dict[str, list[int]] = {group_name: [] for group_name in self.group_name_to_idx.keys()}
        for motion_idx, group_name in self.motion_dir_loader.motion_to_group.items():
            self.group_to_motions[group_name].append(motion_idx)

        # Convert to tensors for efficient sampling
        self.group_to_motions_tensor: dict[str, torch.Tensor] = {}
        for group_name, motion_list in self.group_to_motions.items():
            self.group_to_motions_tensor[group_name] = torch.tensor(motion_list, dtype=torch.long, device=self.device)
            print(f"[MultiMotionCommand]   - Group '{group_name}': {len(motion_list)} motions")

        # Pre-allocate relative pose buffers
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Shared kernel for adaptive sampling
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # Sampling cadence for motion-to-env remap
        if self.cfg.resample_motions_every_s <= 0:
            self._resample_motions_every_steps = 0
        else:
            steps = max(1, int(round(self.cfg.resample_motions_every_s / self.sim_dt)))
            self._resample_motions_every_steps = steps
        self._global_sim_step = 0

        self._remap_version = 0
        self._env_remap_version = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_std"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_min"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_sampling_prob_entropy"] = torch.zeros(self.num_envs, device=self.device)

        prob_init = 1.0 / max(self.num_motions_total, 1)
        self.motion_sampling_probs = torch.full(
            (self.num_motions_total,), prob_init, dtype=torch.float32, device=self.device
        )
        self.motion_sample_counts = torch.zeros(
            self.num_motions_total, dtype=torch.float32, device=self.device
        )
        self.motion_assigned_counts = torch.zeros(
            self.num_motions_total, dtype=torch.float32, device=self.device
        )
        self.motion_fail_counts = torch.zeros(
            self.num_motions_total, dtype=torch.float32, device=self.device
        )

        # Initial assignment of motions to envs and initial resample of time
        all_envs = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._assign_motions(all_envs)
        # Do not resample here: termination manager may not be ready during managers' construction.
        # Time steps start at zero; sampling and writes happen in _update_command().

    # ------------- properties (gathered across envs/motions) -------------
    def _gather_future_by_motion(self, getter: str, horizon: int) -> torch.Tensor:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        motion_indices = self.env_motion_indices
        base_indices = self.time_steps.unsqueeze(1)
        offsets = torch.arange(horizon, device=self.device, dtype=torch.long).view(1, -1)
        frame_indices = base_indices + offsets
        max_valid = self.motion_lengths_minus_one[motion_indices].unsqueeze(1)
        frame_indices = torch.minimum(frame_indices, max_valid)

        flat_motion = motion_indices.unsqueeze(1).expand_as(frame_indices).reshape(-1)
        flat_frames = frame_indices.reshape(-1)
        gathered = self.motion_dir_loader.gather(getter, flat_motion, flat_frames, out_device=self.device)
        new_shape = (self.num_envs, horizon) + gathered.shape[1:]
        return gathered.view(new_shape)

    @property
    def command(self) -> torch.Tensor:
        horizon = self.cfg.motion_horizon
        joint_pos_seq = self._gather_future_by_motion("joint_pos", horizon)
        if self.cfg.command_velocity:
            joint_vel_seq = self._gather_future_by_motion("joint_vel", horizon)
            command_seq = torch.cat([joint_pos_seq, joint_vel_seq], dim=-1)
        else:
            command_seq = joint_pos_seq
        return command_seq.reshape(self.num_envs, -1)

    def _gather_by_motion(self, getter: str) -> torch.Tensor:
        return self.motion_dir_loader.gather(
            getter, self.env_motion_indices, self.time_steps, out_device=self.device
        )
    
    def _gather_by_motion_for_envs(self, getter: str, env_ids: torch.Tensor) -> torch.Tensor:
        motion_idx = self.env_motion_indices[env_ids]
        frame_idx = self.time_steps[env_ids]
        return self.motion_dir_loader.gather(getter, motion_idx, frame_idx, out_device=self.device)

    def _update_sampling_prob_metrics(self):
        probs = self._compute_motion_sampling_probs()
        if probs.numel() == 0:
            zero = 0.0
            self.metrics["motion_sampling_prob_mean"].fill_(zero)
            self.metrics["motion_sampling_prob_std"].fill_(zero)
            self.metrics["motion_sampling_prob_min"].fill_(zero)
            self.metrics["motion_sampling_prob_max"].fill_(zero)
            self.metrics["motion_sampling_prob_entropy"].fill_(zero)
            return

        mean_val = probs.mean().item()
        std_val = probs.std(unbiased=False).item()
        min_val = probs.min().item()
        max_val = probs.max().item()
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        norm_entropy = entropy / max(math.log(max(probs.numel(), 1)), 1e-12)

        self.metrics["motion_sampling_prob_mean"].fill_(mean_val)
        self.metrics["motion_sampling_prob_std"].fill_(std_val)
        self.metrics["motion_sampling_prob_min"].fill_(min_val)
        self.metrics["motion_sampling_prob_max"].fill_(max_val)
        self.metrics["motion_sampling_prob_entropy"].fill_(norm_entropy)

    def _motion_sampling_progress(self) -> float:
        """Return ramp progress in [0, 1] for motion-level sampling weights.

        Motivation: early in training, fail statistics are noisy (often everything fails), so we
        start from uniform sampling and gradually increase the configured weights.
        """

        warmup_s = float(getattr(self.cfg, "motion_sampling_warmup_s", 0.0))
        ramp_s = float(getattr(self.cfg, "motion_sampling_ramp_s", 0.0))
        warmup_steps = max(0, int(round(warmup_s / max(self.sim_dt, 1e-12))))
        ramp_steps = max(0, int(round(ramp_s / max(self.sim_dt, 1e-12))))

        if self._global_sim_step <= warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0

        x = (float(self._global_sim_step - warmup_steps)) / float(ramp_steps)
        x = max(0.0, min(1.0, x))

        schedule = str(getattr(self.cfg, "motion_sampling_schedule", "linear")).lower()
        if schedule == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * x)
        # default: linear
        return x

    def _in_motion_sampling_warmup(self) -> bool:
        """True if we are still in motion-level sampling warmup window."""
        warmup_s = float(getattr(self.cfg, "motion_sampling_warmup_s", 0.0))
        warmup_steps = max(0, int(round(warmup_s / max(self.sim_dt, 1e-12))))
        return self._global_sim_step <= warmup_steps

    def _compute_motion_sampling_probs(self) -> torch.Tensor:
        num_bins = self.num_motions_total
        if num_bins == 0:
            self.motion_sampling_probs = torch.empty(0, device=self.device)
            return self.motion_sampling_probs

        sample_counts = self.motion_sample_counts
        assigned_counts = self.motion_assigned_counts
        fail_counts = self.motion_fail_counts
        
        # fail-based difficulty
        fail_rates = torch.zeros_like(sample_counts)
        valid_mask = sample_counts > 0
        if valid_mask.any():
            fail_rates[valid_mask] = fail_counts[valid_mask] / sample_counts[valid_mask].clamp(min=1e-6)

        mean_fail = fail_rates.mean()
        beta_cap = self.cfg.cap_beta * mean_fail
        if beta_cap > 0:
            capped_rates = torch.minimum(fail_rates, beta_cap)
        else:
            capped_rates = torch.zeros_like(fail_rates)

        capped_sum = capped_rates.sum()
        if capped_sum > 0:
            prob_fail = capped_rates / capped_sum
        else:
            prob_fail = torch.zeros_like(capped_rates)

        # novelty term: prefer less-sampled motions
        novelty = 1.0 / torch.sqrt(assigned_counts + 1.0)
        if novelty.sum() > 0:
            prob_novel = novelty / novelty.sum()
        else:
            prob_novel = torch.zeros_like(novelty)
        
        # uniform term: prefer more uniform sampling
        prob_uniform = torch.full_like(prob_fail, 1.0 / max(num_bins, 1))

        # mix the terms
        progress = self._motion_sampling_progress()
        w_fail_target = float(self.cfg.weight_fail)
        w_novel_target = float(self.cfg.weight_novel)
        w_fail = progress * w_fail_target
        w_novel = progress * w_novel_target

        # keep weights well-formed
        w_sum = w_fail + w_novel
        if w_sum > 1.0:
            w_fail = w_fail / w_sum
            w_novel = w_novel / w_sum
            w_uniform = 0.0
        else:
            w_uniform = max(0.0, 1.0 - w_fail - w_novel)

        probs = w_fail * prob_fail + w_novel * prob_novel + w_uniform * prob_uniform

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = prob_uniform
        else:
            probs = probs / probs_sum

        self.motion_sampling_probs = probs
        return probs

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather_by_motion("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather_by_motion("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_ang_vel_w")

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        # Use body_pos_w then index anchor across bodies
        pos = self._gather_by_motion("body_pos_w")
        return pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        quat = self._gather_by_motion("body_quat_w")
        return quat[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        vel = self._gather_by_motion("body_lin_vel_w")
        return vel[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        vel = self._gather_by_motion("body_ang_vel_w")
        return vel[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _assign_motions(self, env_ids: torch.Tensor):
        """Assign motions to given env ids using difficulty-based and novelty-based sampling."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Check if motion group sampling ratios are configured
        if self.cfg.motion_group_sampling_ratios is not None:
            # Use ratio-based sampling: split environments by group and sample from each group's pool
            self._assign_motions_with_ratios(env_ids)
            return

        probs_motion = self._compute_motion_sampling_probs()

        k_cfg = getattr(self.cfg, "max_active_motions", None)
        use_active_pool = (k_cfg is not None) and (self.num_motions_total > k_cfg)

        if use_active_pool:
            # sample a active set from the motion database (no replacement)
            K = min(k_cfg, self.num_motions_total, n_envs)
            active_motions = torch.multinomial(probs_motion, K, replacement=False)

            # using these motions to fill the envs
            # make every motion get about n_envs / K envs
            reps = n_envs // K
            rem = n_envs % K

            base = active_motions.repeat_interleave(reps)
            if rem > 0:
                extra = active_motions[torch.randperm(K, device=self.device)[:rem]]
                sampled = torch.cat([base, extra], dim=0)
            else:
                sampled = base

            # randomize the order of the envs
            perm = torch.randperm(n_envs, device=self.device)
            self.env_motion_indices[env_ids] = sampled[perm]

            # Update motion groups for multi-teacher support
            for i, env_id in enumerate(env_ids):
                motion_idx = self.env_motion_indices[env_id].item()
                group_name = self.motion_dir_loader.motion_to_group.get(motion_idx, "default")
                group_idx = self.group_name_to_idx[group_name]
                self.env_motion_groups[env_id] = group_idx

            # update the motion sample counts
            unique_motions, counts = torch.unique(sampled, return_counts=True)
            self.motion_assigned_counts.index_add_(
                0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
            )

        else:
            if self.cfg.unique_per_batch and self.num_motions_total >= n_envs:
                sampled = torch.multinomial(probs_motion, n_envs, replacement=False)
            else:
                sampled = torch.multinomial(probs_motion, n_envs, replacement=True)

            self.env_motion_indices[env_ids] = sampled

            # Update motion groups for multi-teacher support
            for i, env_id in enumerate(env_ids):
                motion_idx = self.env_motion_indices[env_id].item()
                group_name = self.motion_dir_loader.motion_to_group.get(motion_idx, "default")
                group_idx = self.group_name_to_idx[group_name]
                self.env_motion_groups[env_id] = group_idx

            unique_motions, counts = torch.unique(sampled, return_counts=True)
            self.motion_assigned_counts.index_add_(
                0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
            )

        self._update_sampling_prob_metrics()

    def _assign_motions_with_ratios(self, env_ids: torch.Tensor):
        """Assign motions to environments using motion group sampling ratios.

        This method splits environments according to configured sampling ratios,
        then samples motions from each group's pool separately while maintaining
        difficulty-based and novelty-based sampling within each group.

        Args:
            env_ids: Tensor of environment IDs to assign motions to
        """
        n_envs = len(env_ids)
        ratios = self.cfg.motion_group_sampling_ratios

        # Validate ratios
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) > 1e-5:
            print(f"[MultiMotionCommand] Warning: motion_group_sampling_ratios sum to {ratio_sum:.4f}, not 1.0. Normalizing...")
            ratios = {k: v / ratio_sum for k, v in ratios.items()}

        # Check that all ratio groups exist in motion groups
        for group_name in ratios.keys():
            if group_name not in self.group_name_to_idx:
                raise ValueError(f"Sampling ratio specified for group '{group_name}' but this group doesn't exist. "
                               f"Available groups: {list(self.group_name_to_idx.keys())}")

        # Compute global motion sampling probabilities (for difficulty/novelty weighting)
        probs_motion_global = self._compute_motion_sampling_probs()

        # Split environments by group according to ratios
        group_env_splits = {}
        start_idx = 0
        for group_name, ratio in sorted(ratios.items()):
            n_envs_group = int(round(n_envs * ratio))
            # Ensure we don't exceed total environments
            if group_name == list(sorted(ratios.keys()))[-1]:  # Last group gets remainder
                n_envs_group = n_envs - start_idx

            if n_envs_group > 0:
                group_env_splits[group_name] = env_ids[start_idx:start_idx + n_envs_group]
                start_idx += n_envs_group

        print(f"[MultiMotionCommand] Assigning {n_envs} environments with ratios: "
              f"{', '.join([f'{k}={len(v)}/{n_envs}' for k, v in group_env_splits.items()])}")

        # Sample motions for each group
        for group_name, group_env_ids in group_env_splits.items():
            self._assign_motions_for_group(group_name, group_env_ids, probs_motion_global)

        self._update_sampling_prob_metrics()

    def _assign_motions_for_group(self, group_name: str, env_ids: torch.Tensor, probs_motion_global: torch.Tensor):
        """Assign motions from a specific group to given environments.

        Args:
            group_name: Name of the motion group to sample from
            env_ids: Tensor of environment IDs to assign motions to
            probs_motion_global: Global motion sampling probabilities (for all motions)
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Get motion indices for this group
        group_motion_indices = self.group_to_motions_tensor[group_name]
        n_motions_in_group = len(group_motion_indices)

        if n_motions_in_group == 0:
            raise ValueError(f"Group '{group_name}' has no motions!")

        # Extract and renormalize probabilities for this group's motions
        probs_group = probs_motion_global[group_motion_indices]
        probs_group = probs_group / probs_group.sum()

        # Sample motions from this group
        k_cfg = getattr(self.cfg, "max_active_motions", None)
        use_active_pool = (k_cfg is not None) and (self.num_motions_total > k_cfg)

        if use_active_pool:
            # Sample active motions from this group's pool
            K = min(k_cfg, n_motions_in_group, n_envs)
            active_motion_indices_in_group = torch.multinomial(probs_group, K, replacement=False)
            active_motions = group_motion_indices[active_motion_indices_in_group]

            # Distribute these motions across environments
            reps = n_envs // K
            rem = n_envs % K

            base = active_motions.repeat_interleave(reps)
            if rem > 0:
                extra = active_motions[torch.randperm(K, device=self.device)[:rem]]
                sampled = torch.cat([base, extra], dim=0)
            else:
                sampled = base

            # Randomize order
            perm = torch.randperm(n_envs, device=self.device)
            self.env_motion_indices[env_ids] = sampled[perm]

        else:
            # Direct sampling from group
            if self.cfg.unique_per_batch and n_motions_in_group >= n_envs:
                sampled_indices_in_group = torch.multinomial(probs_group, n_envs, replacement=False)
            else:
                sampled_indices_in_group = torch.multinomial(probs_group, n_envs, replacement=True)

            sampled = group_motion_indices[sampled_indices_in_group]
            self.env_motion_indices[env_ids] = sampled

        # Update motion groups (all environments in this batch belong to the same group)
        group_idx = self.group_name_to_idx[group_name]
        self.env_motion_groups[env_ids] = group_idx

        # Update motion sample counts
        unique_motions, counts = torch.unique(sampled, return_counts=True)
        self.motion_assigned_counts.index_add_(
            0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Within-motion sampling for the provided environment indices."""
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        motion_indices = self.env_motion_indices[env_ids]

        lengths = self.motion_lengths[motion_indices]
        lengths_minus_one = self.motion_lengths_minus_one[motion_indices]
        denominators = self.motion_length_denominator[motion_indices]
        bin_counts = self.motion_bin_counts[motion_indices]
        bin_counts_float = self.motion_bin_counts_float[motion_indices]
        bin_mask = self.motion_bin_mask[motion_indices]

        # During warmup, we intentionally do NOT update failure statistics to avoid
        # cold-start bias (everything fails early, which makes difficulty estimates noisy).
        if not self._in_motion_sampling_warmup():
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                fail_envs = env_ids[episode_failed]
                fail_motion_idx = motion_indices[episode_failed]
                fail_bin_counts = bin_counts[episode_failed]
                fail_denominators = denominators[episode_failed]
                fail_bins = torch.clamp(
                    (self.time_steps[fail_envs] * fail_bin_counts) // fail_denominators,
                    max=fail_bin_counts - 1,
                )
                linear_indices = fail_motion_idx * self.max_bin_count + fail_bins
                self.current_bin_failed.view(-1).index_add_(
                    0,
                    linear_indices,
                    torch.ones_like(fail_bins, dtype=self.current_bin_failed.dtype),
                )
                self.motion_fail_counts.index_add_(
                    0,
                    fail_motion_idx,
                    torch.ones_like(fail_motion_idx, dtype=self.motion_fail_counts.dtype),
                )

        prob = self.bin_failed_count[motion_indices]
        uniform_term = (self.cfg.adaptive_uniform_ratio / bin_counts_float).unsqueeze(1)
        prob = (prob + uniform_term) * bin_mask

        kernel_tail = self.kernel.numel() - 1
        if kernel_tail > 0:
            prob = F.conv1d(
                F.pad(prob.unsqueeze(1), (0, kernel_tail), mode="replicate"),
                self.kernel.view(1, 1, -1),
            ).squeeze(1)
        else:
            prob = prob.clone()

        prob = prob * bin_mask
        prob_sum = prob.sum(dim=1, keepdim=True)
        zero_rows = prob_sum <= 0
        if torch.any(zero_rows):
            prob[zero_rows] = bin_mask[zero_rows].float()
            prob_sum = prob.sum(dim=1, keepdim=True)
        prob = prob / prob_sum

        sampled_bins = torch.multinomial(prob, 1).squeeze(1)
        rand_offset = torch.rand(len(env_ids), device=self.device)

        lengths_minus_one_float = lengths_minus_one.to(torch.float32)
        time_steps = torch.where(
            lengths_minus_one == 0,
            torch.zeros_like(lengths_minus_one),
            (
                (sampled_bins.to(torch.float32) + rand_offset)
                / torch.clamp(bin_counts_float, min=1.0)
                * lengths_minus_one_float
            ).long(),
        )
        time_steps = torch.clamp(time_steps, max=lengths_minus_one)
        self.time_steps[env_ids] = time_steps

        entropy = -(prob * (prob + 1e-12).log()).sum(dim=1)
        log_bins = torch.log(torch.clamp(bin_counts_float, min=1.0))
        entropy_norm = torch.where(log_bins > 0, entropy / log_bins, torch.zeros_like(entropy))
        self.metrics["sampling_entropy"][env_ids] = entropy_norm

        pmax, imax = prob.max(dim=1)
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.to(torch.float32) / torch.clamp(bin_counts_float, min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.cfg.start_from_beginning:
            start_frame = max(int(self.cfg.start_frame), 0)
            lengths_minus_one = self.motion_lengths[self.env_motion_indices[env_ids]] - 1
            lengths_minus_one = torch.clamp(lengths_minus_one, min=0)
            start_frame_tensor = torch.full_like(lengths_minus_one, start_frame)
            self.time_steps[env_ids] = torch.minimum(start_frame_tensor, lengths_minus_one)
        else:
            self._adaptive_sampling(env_ids)

        motion_indices = self.env_motion_indices[env_ids]
        self.motion_sample_counts.index_add_(
            0,
            motion_indices,
            torch.ones_like(motion_indices, dtype=self.motion_sample_counts.dtype),
        )

        # Gather current sampled states
        body_pos = self._gather_by_motion_for_envs("body_pos_w", env_ids)
        body_quat = self._gather_by_motion_for_envs("body_quat_w", env_ids)
        body_lin = self._gather_by_motion_for_envs("body_lin_vel_w", env_ids)
        body_ang = self._gather_by_motion_for_envs("body_ang_vel_w", env_ids)
        jpos = self._gather_by_motion_for_envs("joint_pos", env_ids)
        jvel = self._gather_by_motion_for_envs("joint_vel", env_ids)

        root_pos = (body_pos[:, 0] + self._env.scene.env_origins[env_ids])
        root_ori = body_quat[:, 0]
        root_lin_vel = body_lin[:, 0].clone()
        root_ang_vel = body_ang[:, 0].clone()

        # Random pose/velocity deltas around sampled states
        rand_samples = sample_uniform(self._pose_ranges[:, 0], self._pose_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        rand_samples = sample_uniform(self._vel_ranges[:, 0], self._vel_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = jpos.clone()
        joint_vel = jvel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_metrics(self):
        self.metrics["error_anchor_pos"].copy_(torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1))
        self.metrics["error_anchor_rot"].copy_(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        self.metrics["error_anchor_lin_vel"].copy_(torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1))
        self.metrics["error_anchor_ang_vel"].copy_(torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1))

        self.metrics["error_body_pos"].copy_(torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1))
        self.metrics["error_body_rot"].copy_(quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1))

        self.metrics["error_body_lin_vel"].copy_(torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1))
        self.metrics["error_body_ang_vel"].copy_(torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1))

        self.metrics["error_joint_pos"].copy_(torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1))
        self.metrics["error_joint_vel"].copy_(torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1))
        self._update_sampling_prob_metrics()

    def _update_command(self):
        self._global_sim_step += 1
        self.time_steps += 1

        # Per-motion episode end detection and resampling
        motion_lengths = self.motion_lengths[self.env_motion_indices]
        ended = self.time_steps >= motion_lengths                                  # (num_envs,)
        self.motion_end_buf[:] = ended
        # envs_to_resample = torch.where(self.time_steps >= motion_lengths)[0]
        # if envs_to_resample.numel() > 0:
        #     self._resample_command(envs_to_resample)

        # Compute relative body poses vs anchor
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update per-motion failure statistics using EMA (skip during warmup)
        if not self._in_motion_sampling_warmup():
            mask = self.motion_bin_mask.float()
            if self.current_bin_failed.any():
                self.bin_failed_count = (
                    self.cfg.adaptive_alpha * self.current_bin_failed
                    + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
                ) * mask
        self.current_bin_failed.zero_()

        # Periodic motion remap
        if self._resample_motions_every_steps > 0 and \
            (self._global_sim_step % self._resample_motions_every_steps) == 0:
            self._remap_version += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        need = self._env_remap_version[env_ids] < self._remap_version
        envs_need_remap = env_ids[need]
        if envs_need_remap.numel() > 0:
            self._assign_motions(envs_need_remap)
            self._env_remap_version[envs_need_remap] = self._remap_version

        self.motion_end_buf[env_ids] = False
        
        return super().reset(env_ids=env_ids)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name))
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the multi-motion command."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    motion: str = MISSING
    file_glob: str = "*.npz"
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    motion_preload_device: str | None = 'cuda'

    motion_horizon: int = 1

    # ---- multi-teacher support: motion groups ----
    # Motion groups configuration for multi-teacher support.
    # Maps group names to folder patterns for categorizing motions.
    # Example: {"lafan": ["lafan_npz_10s_without_fall_and_getup"], "fld": ["motions_fld_test"]}
    # If None, all motions belong to a single "default" group.
    motion_groups: dict[str, list[str]] | None = None

    # Motion group sampling ratios - controls proportion of environments assigned to each group.
    # Maps group names to sampling ratios (should sum to 1.0).
    # Example: {"lafan": 0.7, "fld": 0.3} means 70% of envs use lafan motions, 30% use fld motions.
    # If None, environments are assigned uniformly across all available motions (ignoring groups).
    motion_group_sampling_ratios: dict[str, float] | None = None

    # ---- dataset slicing / sharding across GPUs ----
    # If True, each GPU process loads a disjoint subset of motions (when possible),
    # reducing duplicates during multi-GPU training. This is especially useful when
    # total motions >> (num_envs or max_active_motions).
    motion_dataset_shard_across_gpus: bool = True
    # Use 'global' rank/world_size (RANK/WORLD_SIZE) or 'local' (LOCAL_RANK/LOCAL_WORLD_SIZE).
    motion_dataset_shard_by: str = "global"
    # Deterministic shuffle seed before slicing.
    motion_dataset_shard_seed: int = 0
    # Sharding strategy when total < world_size * load_cap: 'chunk' (contiguous) or 'stride' (round-robin).
    motion_dataset_shard_strategy: str = "chunk"
    # Cap number of motions loaded per process. If None and sharding enabled, auto-caps to min(num_envs, max_active_motions).
    motion_dataset_load_cap: int | None = None
    # If True, print shard info once at startup (useful for sanity checks).
    motion_dataset_log_shard_info: bool = False
    # If True, write shard info once into wandb run.summary (rank0 only). No-op if wandb isn't used.
    motion_dataset_log_wandb_summary: bool = True

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    start_from_beginning: bool = False
    start_frame: int = 0

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    weight_fail: float = 0.5
    weight_novel: float = 0.3
    cap_beta: float = 2.0

    command_velocity: bool = True

    # ---- motion-level sampling weight schedule (uniform -> ramp to weights above) ----
    # insight for the values:
    # - warmup_s: make sure every motion is sampled at least once. (10-20 times of resample_motions_every_s)
    # - ramp_s: make sure the weights are not too small. (20-50 times of resample_motions_every_s)
    # - schedule: "linear" or "cosine" is the schedule type for the ramp. (cosine is better)
    # Warmup: keep motion sampling uniform for this duration (seconds).
    motion_sampling_warmup_s: float = 1000000000.0
    # Ramp: linearly/cosine ramp fail/novel weights from 0 -> target over this duration (seconds).
    motion_sampling_ramp_s: float = 1000000000.0
    # Schedule type for ramp: "linear" or "cosine".
    motion_sampling_schedule: str = "linear"

    # Resampling cadence (seconds) for motion-to-env reassignment (set to 0 or 1e9 to disable)
    resample_motions_every_s: float = 1000000000.0
    # Whether to sample motions without replacement per remap batch when possible
    unique_per_batch: bool = True

    max_active_motions: int | None = 10000

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
