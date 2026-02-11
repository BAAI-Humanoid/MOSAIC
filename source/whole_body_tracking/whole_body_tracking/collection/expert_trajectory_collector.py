"""
Expert Trajectory Collector for MOSAIC Off-Policy Learning

Collects state-action pairs during policy evaluation and saves them
in a format compatible with RolloutStorage for off-policy learning.

Usage:
    # During evaluation
    collector = ExpertTrajectoryCollector(
        obs_dim_dict={'policy': 154, 'student_policy': 770},
        action_dim=29,
        max_trajectories=1000,
        device='cuda:0'
    )

    # Collect data during rollout
    collector.add_step(obs_dict, actions, action_mean, action_sigma)

    # Mark trajectory ends
    collector.mark_trajectory_end(count=num_done)

    # Save collected data
    collector.save('expert_trajectories.npy')

    # Load for training
    expert_data = ExpertTrajectoryCollector.load('expert_trajectories.npy')
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class ExpertTrajectoryCollector:
    """
    Collects expert trajectories in a format compatible with RolloutStorage.

    Stored data structure:
    {
        'observations': [T, N, obs_dim],           # Teacher observations
        'student_observations': [T, N, obs_dim],   # Student observations (optional)
        'actions': [T, N, action_dim],
        'action_mean': [T, N, action_dim],
        'action_sigma': [T, N, action_dim],
        'valid_mask': [T, N],                      # Optional, True for active envs
        'metadata': {
            'num_trajectories': int,
            'num_steps': int,
            'obs_dim_dict': dict,
            'action_dim': int
        }
    }
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        student_obs_dim: Optional[int] = None,
        max_trajectories: int = 1000,
        device: str = 'cpu'
    ):
        """
        Args:
            obs_dim: Dimension of teacher observation space
            action_dim: Dimension of action space
            student_obs_dim: Dimension of student observation space (optional)
            max_trajectories: Maximum number of trajectories to store
            device: Device to store data on during collection
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.student_obs_dim = student_obs_dim
        self.max_trajectories = max_trajectories
        self.device = device

        # Storage buffers
        self.data_buffer = {
            'observations': [],
            'actions': [],
            'action_mean': [],
            'action_sigma': []
        }
        self.valid_masks = []

        # Add student observations if specified
        if student_obs_dim is not None:
            self.data_buffer['student_observations'] = []

        self.num_steps = 0
        self.num_trajectories = 0

    def add_step(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_mean: Optional[torch.Tensor] = None,
        action_sigma: Optional[torch.Tensor] = None,
        student_observations: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ):
        """
        Add a single step of data.

        Args:
            observations: Teacher observations [N, obs_dim]
            actions: Actions taken [N, action_dim]
            action_mean: Mean of action distribution [N, action_dim] (optional)
            action_sigma: Std of action distribution [N, action_dim] (optional)
            student_observations: Student observations [N, student_obs_dim] (optional)
            valid_mask: Boolean mask [N] to indicate valid envs for this step (optional)
        """
        if self.num_trajectories >= self.max_trajectories:
            print(f"[ExpertTrajectoryCollector] Warning: Reached max trajectories "
                  f"({self.max_trajectories}), skipping new data")
            return

        # Store observations
        self.data_buffer['observations'].append(observations.cpu())

        # Store student observations if provided
        if student_observations is not None and 'student_observations' in self.data_buffer:
            self.data_buffer['student_observations'].append(student_observations.cpu())

        # Store actions
        self.data_buffer['actions'].append(actions.cpu())

        # Store action distributions (use actions if not provided)
        if action_mean is not None:
            self.data_buffer['action_mean'].append(action_mean.cpu())
        else:
            self.data_buffer['action_mean'].append(actions.cpu())

        if action_sigma is not None:
            self.data_buffer['action_sigma'].append(action_sigma.cpu())
        else:
            # Default sigma (for deterministic actions)
            self.data_buffer['action_sigma'].append(
                torch.ones_like(actions.cpu()) * 0.01
            )

        # Track valid mask if provided
        if valid_mask is not None:
            mask_cpu = valid_mask.detach().cpu().bool()
            if mask_cpu.ndim != 1 or mask_cpu.shape[0] != actions.shape[0]:
                raise ValueError("valid_mask must be shape [N] matching actions batch size.")
            self.valid_masks.append(mask_cpu)
            self.num_steps += int(mask_cpu.sum().item())
        else:
            # Increment by batch size (number of parallel environments)
            batch_size = actions.shape[0]
            self.num_steps += batch_size

    def mark_trajectory_end(self, count: int = 1):
        """
        Mark the end of trajectory/trajectories.

        Args:
            count: Number of trajectories that ended
        """
        self.num_trajectories += count
        if self.num_trajectories % 100 == 0:
            print(f"[ExpertTrajectoryCollector] Collected {self.num_trajectories} trajectories, "
                  f"{self.num_steps} total steps")

    def save(self, save_path: str):
        """
        Save collected trajectories to disk.

        Args:
            save_path: Path to save file (*.npy)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Stack all data
        stacked_data = {}
        for key, data_list in self.data_buffer.items():
            if len(data_list) == 0:
                continue
            stacked_data[key] = torch.stack(data_list, dim=0).numpy()  # [T, N, dim]

        if self.valid_masks:
            if len(self.valid_masks) != len(self.data_buffer['observations']):
                raise ValueError("valid_mask count does not match collected steps.")
            stacked_data['valid_mask'] = torch.stack(self.valid_masks, dim=0).numpy()  # [T, N]

        # Add metadata
        obs_dim_dict = {'observations': self.obs_dim}
        if self.student_obs_dim is not None:
            obs_dim_dict['student_observations'] = self.student_obs_dim

        stacked_data['metadata'] = {
            'num_trajectories': self.num_trajectories,
            'num_steps': self.num_steps,
            'obs_dim_dict': obs_dim_dict,
            'action_dim': self.action_dim,
            'has_valid_mask': bool(self.valid_masks),
        }

        # Save as numpy archive
        np.save(save_path, stacked_data, allow_pickle=True)

        print("=" * 80)
        print(f"Expert trajectories saved to: {save_path}")
        print(f"  Total trajectories: {self.num_trajectories}")
        print(f"  Total steps: {self.num_steps}")
        print(f"  Observation dims: {obs_dim_dict}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        print("=" * 80)

    @staticmethod
    def load(load_path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Load expert trajectories from disk.

        Args:
            load_path: Path to load file (*.npy)
            device: Device to load data to

        Returns:
            Dictionary containing expert trajectories as torch tensors
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Expert trajectory file not found: {load_path}")

        # Load numpy archive
        data_dict = np.load(load_path, allow_pickle=True).item()

        # Convert to torch tensors
        expert_data = {}
        for key, value in data_dict.items():
            if key != 'metadata':
                expert_data[key] = torch.from_numpy(value).to(device)

        expert_data['metadata'] = data_dict['metadata']

        print("=" * 80)
        print(f"Loaded expert trajectories from: {load_path}")
        print(f"  Total trajectories: {expert_data['metadata']['num_trajectories']}")
        print(f"  Total steps: {expert_data['metadata']['num_steps']}")
        print(f"  Observation dims: {expert_data['metadata']['obs_dim_dict']}")
        print(f"  Action dim: {expert_data['metadata']['action_dim']}")
        if expert_data['metadata'].get('has_valid_mask'):
            print("  Valid mask: True")
        print("=" * 80)

        return expert_data

    def clear(self):
        """Clear all collected data"""
        for key in self.data_buffer.keys():
            self.data_buffer[key] = []
        self.valid_masks = []
        self.num_steps = 0
        self.num_trajectories = 0
        print("[ExpertTrajectoryCollector] Trajectory collector cleared")

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about collected data"""
        mask_mem = 0
        if self.valid_masks:
            mask_mem = sum(m.element_size() * m.nelement() for m in self.valid_masks)
        return {
            'num_trajectories': self.num_trajectories,
            'num_steps': self.num_steps,
            'avg_steps_per_trajectory': self.num_steps / max(1, self.num_trajectories),
            'obs_dim': self.obs_dim,
            'student_obs_dim': self.student_obs_dim,
            'action_dim': self.action_dim,
            'memory_usage_mb': sum(
                sum(t.element_size() * t.nelement() for t in tensors)
                for tensors in self.data_buffer.values() if tensors
            ) / 1024 / 1024 + (mask_mem / 1024 / 1024)
        }
