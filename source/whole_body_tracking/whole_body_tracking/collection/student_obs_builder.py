"""
Student Observation Builder for Expert Trajectory Collection

This module provides isolated observation building for collecting expert trajectories
with different student observation configurations than the teacher policy.

Key Features:
- Isolated history buffers (doesn't modify environment)
- Flexible observation composition
- Support for history-aware observations
- Compatible with MOSAIC's expert BC

Usage:
    # Create builder with student obs config
    builder = StudentObsBuilder(
        env=env,
        student_obs_config=student_obs_cfg,
        device='cuda:0'
    )

    # During collection
    student_obs = builder.get_observation(obs_dict)
    builder.update_history(obs_dict)

    # On episode reset
    builder.reset_history(done_env_ids)
"""

import torch
from typing import Dict, List, Optional
from isaaclab.envs import ManagerBasedRLEnv


class StudentObsBuilder:
    """
    Isolated builder for computing student observations without modifying env.
    Manages its own history buffers and observation composition.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        obs_keys: List[str],
        history_length: int = 0,
        device: str = 'cuda:0'
    ):
        """
        Args:
            env: The environment (used to get observations)
            obs_keys: List of observation keys to include (e.g., ['command', 'joint_pos', ...])
            history_length: Number of history frames (0 = no history)
            device: Device to store buffers
        """
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.obs_keys = obs_keys
        self.history_length = history_length

        # Get observation dimensions from environment
        self.obs_dims = {}

        # Parse observation dimensions from policy observations
        if not hasattr(env, "observation_manager"):
            raise ValueError("Environment must expose an observation_manager.")

        # Get observation term names and configs for policy group
        if "policy" not in env.observation_manager._group_obs_term_names:
            raise ValueError("Environment must have 'policy' observation group")

        policy_term_names = env.observation_manager._group_obs_term_names["policy"]
        policy_term_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]

        # Get dimensions for each observation term
        for term_name, term_cfg in zip(policy_term_names, policy_term_cfgs):
            if term_name in obs_keys:
                # Call the observation function to get shape
                obs_value = term_cfg.func(env, **term_cfg.params)
                self.obs_dims[term_name] = obs_value.shape[-1]
                print(f"[StudentObsBuilder] {term_name}: dim={obs_value.shape[-1]}")

        # Initialize history buffers if needed
        self.history_buffers = {}
        if history_length > 0:
            for key in obs_keys:
                if key in self.obs_dims:
                    obs_dim = self.obs_dims[key]
                    self.history_buffers[key] = torch.zeros(
                        self.num_envs, history_length, obs_dim,
                        device=device
                    )
                    print(f"[StudentObsBuilder] Created history buffer for {key}: "
                          f"shape=({self.num_envs}, {history_length}, {obs_dim})")

        # Calculate total dimension
        self.total_dim = self._calculate_total_dim()

        print(f"[StudentObsBuilder] Initialized:")
        print(f"  Obs keys: {self.obs_keys}")
        print(f"  Obs dims: {self.obs_dims}")
        print(f"  History length: {self.history_length}")
        print(f"  Total dimension: {self.total_dim}")

    def _calculate_total_dim(self) -> int:
        """Calculate total observation dimension including history"""
        total = 0
        for key in self.obs_keys:
            if key in self.obs_dims:
                obs_dim = self.obs_dims[key]
                if self.history_length > 0:
                    # Include history: current + (history_length - 1) past frames
                    total += obs_dim * self.history_length
                else:
                    # No history: just current frame
                    total += obs_dim
        return total

    def get_observation(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compose student observation from components.

        Args:
            obs_dict: Dictionary of observations from environment

        Returns:
            Student observation tensor [num_envs, total_dim]
        """
        components = []

        for key in self.obs_keys:
            if key not in obs_dict:
                print(f"[StudentObsBuilder] Warning: {key} not in obs_dict, skipping")
                continue

            obs_value = obs_dict[key]

            if self.history_length > 0 and key in self.history_buffers:
                # Include history: [current, past_1, past_2, ...]
                buffer = self.history_buffers[key]
                # Flatten: [N, history_length, obs_dim] -> [N, history_length * obs_dim]
                history_data = buffer.reshape(buffer.shape[0], -1)
                components.append(history_data)
            else:
                # No history: just current observation
                components.append(obs_value)

        if components:
            return torch.cat(components, dim=-1)
        else:
            return torch.zeros(self.num_envs, 0, device=self.device)

    def update_history(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Update history buffers with current observations.

        Args:
            obs_dict: Dictionary of observations to add to history
        """
        if self.history_length == 0:
            return

        for key in self.history_buffers.keys():
            if key in obs_dict:
                current_obs = obs_dict[key]
                buffer = self.history_buffers[key]

                # Shift history: [t-1, t-2, ...] -> [t-2, t-3, ...]
                buffer[:, 1:] = buffer[:, :-1].clone()
                # Add current: [t, t-1, t-2, ...]
                buffer[:, 0] = current_obs.clone()

    def reset_history(self, env_ids: torch.Tensor):
        """
        Reset history buffers for done environments.

        Args:
            env_ids: Indices of environments to reset
        """
        if self.history_length == 0 or len(env_ids) == 0:
            return

        for buffer in self.history_buffers.values():
            buffer[env_ids] = 0.0


def create_student_obs_builder_from_env_cfg(
    env: ManagerBasedRLEnv,
    student_obs_cfg_name: str,
    device: str = 'cuda:0'
) -> Optional[StudentObsBuilder]:
    """
    Create StudentObsBuilder from environment configuration name.

    Args:
        env: The environment
        student_obs_cfg_name: Name of student observation config
                             (e.g., 'policy_no_history', 'policy_history_5')
        device: Device to store buffers

    Returns:
        StudentObsBuilder instance or None if config not found
    """
    # Parse config name
    if 'history' in student_obs_cfg_name:
        # Extract history length from name (e.g., 'policy_history_5' -> 5)
        try:
            history_length = int(student_obs_cfg_name.split('_')[-1])
        except ValueError:
            history_length = 5  # Default
    else:
        history_length = 0

    # Get observation keys from policy group
    policy_obs_group = env.observation_manager.group_obs_term_cfgs.get("policy", None)
    if policy_obs_group is None:
        print("[StudentObsBuilder] Error: No 'policy' observation group found")
        return None

    obs_keys = list(policy_obs_group.keys())

    return StudentObsBuilder(
        env=env,
        obs_keys=obs_keys,
        history_length=history_length,
        device=device
    )
