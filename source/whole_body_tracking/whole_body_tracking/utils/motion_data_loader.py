"""
Standalone Motion Data Loader

Extracted core data loading logic from MultiMotionLoader, independent of Isaac Lab environment
"""

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from isaaclab.utils.math import quat_rotate_inverse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "isaaclab is required. Install Isaac Lab or add it to PYTHONPATH."
    ) from exc


def _load_single_npz(npz_file, anchor_body_idx=7):
    """
    Load a single npz file (for parallel processing)

    Args:
        npz_file: Path to npz file
        anchor_body_idx: Index of anchor body in body_quat_w/body_lin_vel_w arrays
                         Default: 7 (torso_link for G1)

    Returns:
        tuple: (joint_pos, joint_vel, anchor_quat, anchor_lin_vel)
    """
    data = np.load(npz_file)

    joint_pos = data['joint_pos']  # [T, 29]
    joint_vel = data['joint_vel']  # [T, 29]
    body_quat_w = data['body_quat_w']  # [T, 30, 4]
    body_lin_vel_w = data['body_lin_vel_w']  # [T, 30, 3]

    # Keep only anchor body (e.g., torso_link for G1 at index 7)
    anchor_quat = body_quat_w[:, anchor_body_idx, :]  # [T, 4]
    anchor_lin_vel = body_lin_vel_w[:, anchor_body_idx, :]  # [T, 3]

    return joint_pos, joint_vel, anchor_quat, anchor_lin_vel


class SimpleMotionLoader:
    """
    Simplified Motion Data Loader

    Load motion data from npz files and extract features required for training
    """

    def __init__(self, motion_dir: str, device: str = 'cuda', anchor_body_idx: int = 7):
        """
        Args:
            motion_dir: Motion data directory (will recursively search all subdirectories)
            device: 'cuda' or 'cpu'
            anchor_body_idx: Index of anchor body in body_quat_w/body_lin_vel_w arrays
                             Default: 7 (torso_link for G1)
        """
        self.motion_dir = Path(motion_dir)
        self.device = torch.device(device)
        self.anchor_body_idx = anchor_body_idx

        # Recursively scan all npz files in all subdirectories
        self.motion_files = sorted(list(self.motion_dir.rglob("*.npz")))

        if len(self.motion_files) == 0:
            raise ValueError(f"No .npz files found in {motion_dir} (searched recursively)")

        print(f"Found {len(self.motion_files)} motion files (searched recursively)")
        print(f"  Using anchor body index: {anchor_body_idx}")

        # Print subdirectories found
        subdirs = set(f.parent.relative_to(self.motion_dir) for f in self.motion_files)
        if len(subdirs) > 1 or (len(subdirs) == 1 and list(subdirs)[0] != Path('.')):
            print(f"  From {len(subdirs)} subdirectories:")
            for subdir in sorted(subdirs)[:10]:  # Show first 10
                count = sum(1 for f in self.motion_files if f.parent.relative_to(self.motion_dir) == subdir)
                print(f"    {subdir}: {count} files")
            if len(subdirs) > 10:
                print(f"    ... and {len(subdirs) - 10} more subdirectories")

        # Load all data
        self._load_all_data()

    def _load_all_data(self):
        """Load all npz files using parallel processing"""

        print(f"Loading {len(self.motion_files)} motion files...")

        # Use multiprocessing for parallel loading
        num_workers = min(cpu_count(), 16)  # Use up to 16 workers
        print(f"  Using {num_workers} parallel workers...")

        # Create partial function with anchor_body_idx
        load_fn = partial(_load_single_npz, anchor_body_idx=self.anchor_body_idx)

        with Pool(num_workers) as pool:
            # Use imap for progress tracking (or map for simplicity)
            try:
                from tqdm import tqdm
                results = list(tqdm(
                    pool.imap(load_fn, self.motion_files),
                    total=len(self.motion_files),
                    desc="Loading files"
                ))
            except ImportError:
                # Fallback if tqdm not available
                results = pool.map(load_fn, self.motion_files)

        # Unpack results
        joint_pos_list = []
        joint_vel_list = []
        body_quat_list = []
        body_lin_vel_list = []

        for joint_pos, joint_vel, anchor_quat, anchor_lin_vel in results:
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            body_quat_list.append(anchor_quat)
            body_lin_vel_list.append(anchor_lin_vel)

        # Concatenate all data
        joint_pos_np = np.concatenate(joint_pos_list, axis=0)
        joint_vel_np = np.concatenate(joint_vel_list, axis=0)
        body_quat_np = np.concatenate(body_quat_list, axis=0)
        body_lin_vel_np = np.concatenate(body_lin_vel_list, axis=0)

        # Convert to torch tensor (keep on CPU to save GPU memory)
        # Will transfer to GPU only when needed for computation
        self.joint_pos = torch.from_numpy(joint_pos_np).float()
        self.joint_vel = torch.from_numpy(joint_vel_np).float()
        self.body_quat_w = torch.from_numpy(body_quat_np).float()
        self.body_lin_vel_w = torch.from_numpy(body_lin_vel_np).float()

        self.total_frames = len(self.joint_pos)

        print(f"Loaded {self.total_frames:,} frames from {len(self.motion_files)} files")
        print(f"  Data stored on CPU (will transfer to {self.device} for computation)")

    def __len__(self):
        return len(self.motion_files)


def load_motion_data_for_training(motion_dir: str, device: str = 'cuda',
                                  filter_outliers: bool = True,
                                  vel_threshold: float = 10.0,
                                  history_length: int = 4,
                                  anchor_body_idx: int = 7):
    """
    Load motion data and compute features required for training (with history)

    Args:
        motion_dir: Motion data directory
        device: 'cuda' or 'cpu'
        filter_outliers: Whether to filter outliers
        vel_threshold: Velocity threshold (m/s), samples exceeding this will be filtered
        history_length: Number of history frames
        anchor_body_idx: Index of anchor body in body_quat_w/body_lin_vel_w arrays
                         Default: 7 (torso_link for G1)

    Returns:
        ref_vel_estimator_obs: [N, (29+29+3)*(1+history_length)] Input with history
        ref_anchor_lin_vel_b: [N, 3] Target (anchor body linear velocity in anchor frame)
    """

    print("Loading motion data...")
    print(f"  Using anchor body index: {anchor_body_idx}")

    # 1. Load raw data (on CPU)
    loader = SimpleMotionLoader(motion_dir, device=device, anchor_body_idx=anchor_body_idx)

    # 2. Extract data and transfer to GPU for computation
    print(f"Transferring data to {device} for computation...")
    ref_dof_pos = loader.joint_pos.to(device)  # [N, 29]
    ref_dof_vel = loader.joint_vel.to(device)  # [N, 29]
    anchor_quat_w = loader.body_quat_w.to(device)  # [N, 4] anchor body quaternion
    anchor_lin_vel_w = loader.body_lin_vel_w.to(device)  # [N, 3] anchor body linear velocity

    # 3. Compute projected gravity (using anchor body quaternion)
    gravity_w = torch.zeros(anchor_quat_w.shape[0], 3, device=device)
    gravity_w[:, 2] = -1.0
    ref_projected_gravity = quat_rotate_inverse(anchor_quat_w, gravity_w)

    # 4. Compute ref_anchor_lin_vel_b (linear velocity in anchor frame)
    ref_anchor_lin_vel_b = quat_rotate_inverse(anchor_quat_w, anchor_lin_vel_w)

    # 5. Free GPU memory from temporary data (anchor_quat_w, anchor_lin_vel_w, gravity_w)
    del anchor_quat_w, anchor_lin_vel_w, gravity_w
    if device == 'cuda':
        torch.cuda.empty_cache()

    # 6. Filter outliers
    if filter_outliers:
        # Compute L2 norm of velocity
        vel_norm = torch.norm(ref_anchor_lin_vel_b, dim=-1)

        # Find samples within normal range
        valid_mask = vel_norm < vel_threshold
        num_outliers = (~valid_mask).sum().item()

        if num_outliers > 0:
            print(f"Filtering {num_outliers:,} outliers (vel > {vel_threshold} m/s)")

            # Filter data
            ref_dof_pos = ref_dof_pos[valid_mask]
            ref_dof_vel = ref_dof_vel[valid_mask]
            ref_projected_gravity = ref_projected_gravity[valid_mask]
            ref_anchor_lin_vel_b = ref_anchor_lin_vel_b[valid_mask]

    # 7. Build observations with history - CRITICAL: Match IsaacLab's concatenation order!
    # IsaacLab concatenates: [command_all_history, gravity_all_history]
    # NOT: [frame[t], frame[t-1], ...]
    total_frames = len(ref_dof_pos)
    num_samples = total_frames - history_length

    print(f"\nBuilding history (length={history_length})...")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Output samples: {num_samples:,}")
    print(f"  CRITICAL: Using IsaacLab's concatenation order!")

    # Calculate expected memory usage
    expected_memory_gb = num_samples * (61 * (history_length + 1) + 3) * 4 / 1024**3
    print(f"  Expected output memory: {expected_memory_gb:.2f} GB")

    # Keep features separate (do NOT concatenate yet!)
    # We need to build history for each feature separately, then concatenate
    print(f"  Building separate histories for each feature...")

    # Create indices for all samples
    sample_indices = torch.arange(history_length, total_frames, device=device)  # [num_samples]
    # CRITICAL: Use IsaacLab's history order [t-4, t-3, t-2, t-1, t] (oldest to newest)
    history_offsets = torch.arange(-(history_length), 1, 1, device=device)  # [-4, -3, -2, -1, 0]
    all_indices = sample_indices.unsqueeze(1) + history_offsets.unsqueeze(0)  # [num_samples, 5]

    # Build history for command (joint_pos + joint_vel)
    print(f"  Building command history...")
    command = torch.cat([ref_dof_pos, ref_dof_vel], dim=1)  # [N, 58]
    command_history = command[all_indices]  # [num_samples, 5, 58]
    command_history_flat = command_history.reshape(num_samples, -1)  # [num_samples, 290]

    # Build history for gravity
    print(f"  Building gravity history...")
    gravity_history = ref_projected_gravity[all_indices]  # [num_samples, 5, 3]
    gravity_history_flat = gravity_history.reshape(num_samples, -1)  # [num_samples, 15]

    # Concatenate in IsaacLab's order: [command_all_history, gravity_all_history]
    print(f"  Concatenating in IsaacLab order: [command_history, gravity_history]...")
    ref_vel_estimator_obs = torch.cat([command_history_flat, gravity_history_flat], dim=1)  # [num_samples, 305]

    # Extract targets
    ref_anchor_lin_vel_b = ref_anchor_lin_vel_b[sample_indices]

    # Free memory
    del ref_dof_pos, ref_dof_vel, ref_projected_gravity, command, command_history, gravity_history
    del command_history_flat, gravity_history_flat, all_indices
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"\nData ready: {len(ref_vel_estimator_obs):,} samples")
    print(f"  Input shape: {ref_vel_estimator_obs.shape}")
    print(f"  Target shape: {ref_anchor_lin_vel_b.shape}")
    print(f"  ref_anchor_lin_vel_b mean: [{ref_anchor_lin_vel_b[:, 0].mean():.3f}, "
          f"{ref_anchor_lin_vel_b[:, 1].mean():.3f}, {ref_anchor_lin_vel_b[:, 2].mean():.3f}]")
    print(f"  ref_anchor_lin_vel_b std:  [{ref_anchor_lin_vel_b[:, 0].std():.3f}, "
          f"{ref_anchor_lin_vel_b[:, 1].std():.3f}, {ref_anchor_lin_vel_b[:, 2].std():.3f}]")
    print(f"  ref_anchor_lin_vel_b max:  {ref_anchor_lin_vel_b.abs().max():.3f}")

    return ref_vel_estimator_obs, ref_anchor_lin_vel_b
