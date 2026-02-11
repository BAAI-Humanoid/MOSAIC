# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MOSAIC: Motion Imitation Hybrid Learning
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from itertools import chain
import numpy as np
from typing import Optional, Dict, Any

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class MOSAIC:
    """
    MOSAIC
    Loss Function:
        L = L_PPO + λ_off * L_BC_expert + λ_teacher * L_BC_teacher
    """

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        # PPO parameters
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters (not supported in MOSAIC)
        rnd_cfg: dict | None = None,
        # Symmetry parameters (not supported in MOSAIC)
        symmetry_cfg: dict | None = None,
        # ========== MOSAIC Mode Selection ==========
        hybrid=True,  # True = hybrid mode (_update_hybrid), False = pure BC mode (_update_pure_bc)
        # MOSAIC specific parameters
        use_ppo=True,  # Only used in hybrid mode
        expert_trajectory_path: Optional[str] = None,
        lambda_off_policy=1.0,
        lambda_off_policy_decay=1.0,
        lambda_off_policy_min=0.0,
        off_policy_batch_size=256,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",  # "kl" or "mse"
        expert_normalize_obs=True,  # Whether to normalize expert observations
        expert_update_normalizer=False,  # Whether expert obs should update normalizer statistics
        teacher_policy: Optional[ActorCritic] = None,
        teacher_checkpoint_path: Optional[str | dict] = None,  # Single teacher (str) or multi-teacher (dict: group_name -> path)
        teacher_policy_cfg: Optional[dict] = None,  # For creating teacher with correct dimensions
        teacher_obs_source_mapping: Optional[dict[str, str]] = None,  # Maps teacher group to obs source: "policy", "teacher", "critic"
        teacher_critic_checkpoint_path: Optional[str] = None,  # Separate checkpoint for critic
        teacher_critic_frozen: bool = True,  # Whether to freeze teacher critic (False = allow fine-tuning)
        train_critic_during_distillation: bool = False,  # Whether to train critic when use_ppo=False
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.99,
        lambda_teacher_min=0.01,
        teacher_loss_type="kl",  # "kl" (KL divergence) or "mse" (MSE on action means)
        # Gradient accumulation for stability (similar to Distillation's gradient_length)
        gradient_accumulation_steps=1,  # 1 = no accumulation, >1 = accumulate gradients
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Observation normalizers (for denormalizing student obs when using teacher normalizers)
        obs_normalizer: Optional[torch.nn.Module] = None,
        privileged_obs_normalizer: Optional[torch.nn.Module] = None,
        # Reference Velocity Estimator parameters
        use_estimate_ref_vel: bool = False,
        ref_vel_estimator_checkpoint_path: Optional[str] = None,
        ref_vel_estimator_type: str = "mlp",
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        self.obs_normalizer = obs_normalizer
        self.privileged_obs_normalizer = privileged_obs_normalizer

        self.use_estimate_ref_vel = use_estimate_ref_vel
        self.ref_vel_estimator = None
        if use_estimate_ref_vel:
            if ref_vel_estimator_checkpoint_path is None:
                raise ValueError("ref_vel_estimator_checkpoint_path must be provided when use_estimate_ref_vel=True")

            print(f"[MOSAIC] Loading reference velocity estimator from: {ref_vel_estimator_checkpoint_path}")
            print(f"[MOSAIC] Estimator type: {ref_vel_estimator_type}")

            if ref_vel_estimator_type == "mlp":
                from rsl_rl.modules import VelocityEstimator
                self.ref_vel_estimator = VelocityEstimator.load(ref_vel_estimator_checkpoint_path, device=self.device)
            elif ref_vel_estimator_type == "transformer":
                from rsl_rl.modules import VelocityEstimatorTransformer
                checkpoint = torch.load(ref_vel_estimator_checkpoint_path, map_location=self.device, weights_only=False)
                self.ref_vel_estimator = VelocityEstimatorTransformer(
                    feature_dim=checkpoint.get('feature_dim', 61),
                    history_length=checkpoint.get('history_length', 5),
                    d_model=checkpoint.get('d_model', 128),
                    nhead=checkpoint.get('nhead', 4),
                    num_layers=checkpoint.get('num_layers', 2),
                ).to(self.device)
                self.ref_vel_estimator.load_state_dict(checkpoint['model_state_dict'])
                print(f"[MOSAIC] Transformer estimator loaded successfully")
            else:
                raise ValueError(f"Unknown ref_vel_estimator_type: {ref_vel_estimator_type}. Must be 'mlp' or 'transformer'")

            self.ref_vel_estimator.eval()
            for param in self.ref_vel_estimator.parameters():
                param.requires_grad = False
            print("[MOSAIC] Reference velocity estimator loaded and frozen")

            self.ref_vel_estimator_obs_shape = (self.ref_vel_estimator.num_obs,)
            print(f"[MOSAIC] Estimator observation shape: {self.ref_vel_estimator_obs_shape}")
        else:
            self.ref_vel_estimator_obs_shape = None

        self.teacher_critic_checkpoint_path = teacher_critic_checkpoint_path
        self.teacher_critic_frozen = teacher_critic_frozen
        self.train_critic_during_distillation = train_critic_during_distillation

        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.rnd_optimizer = None

        self.symmetry = None

        self.policy = policy
        self.policy.to(self.device)

        from rsl_rl.modules import ResidualActorCritic
        if isinstance(policy, ResidualActorCritic):
            # Residual learning: only optimize residual network + critic
            # GMT policy is frozen and should not be in optimizer
            trainable_params = list(policy.residual_actor.parameters())
            trainable_params.extend(policy.critic.parameters())
            if hasattr(policy, 'std'):
                trainable_params.append(policy.std)
            elif hasattr(policy, 'log_std'):
                trainable_params.append(policy.log_std)
            self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
            print("[MOSAIC] Residual learning: optimizer updates residual_actor + critic")
        elif not use_ppo:
            if teacher_critic_checkpoint_path is not None and not teacher_critic_frozen:
                self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
                print("[MOSAIC] Pure BC mode with critic fine-tuning: optimizer updates both actor and critic")
            else:
                self.optimizer = optim.Adam(self.policy.actor.parameters(), lr=learning_rate)
                print("[MOSAIC] Pure BC mode: optimizer only updates actor network")
        else:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.storage: RolloutStorage = None
        self.transition = RolloutStorage.Transition()

        self.use_ppo = use_ppo
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.hybrid = hybrid 

        self.expert_trajectory_path = expert_trajectory_path
        self.lambda_off_policy_init = lambda_off_policy
        self.lambda_off_policy_decay = lambda_off_policy_decay
        self.lambda_off_policy_min = lambda_off_policy_min
        self.lambda_off_policy_current = lambda_off_policy
        self.off_policy_batch_size = off_policy_batch_size
        self.expert_allow_repeat_sampling = expert_allow_repeat_sampling
        self.use_off_policy_bc = expert_trajectory_path is not None
        self.expert_loss_type = expert_loss_type
        self.expert_normalize_obs = expert_normalize_obs
        self.expert_update_normalizer = expert_update_normalizer
        if self.expert_loss_type not in ["kl", "mse"]:
            raise ValueError(f"expert_loss_type must be 'kl' or 'mse', got {self.expert_loss_type}")
        self._warned_expert_sigma_missing = False

        self.expert_data: Optional[Dict[str, torch.Tensor]] = None
        self.expert_data_indices = None

        self.use_multi_teacher = isinstance(teacher_checkpoint_path, dict)

        self.teacher_obs_source_mapping = teacher_obs_source_mapping
        if self.use_multi_teacher and self.teacher_obs_source_mapping is None:
            self.teacher_obs_source_mapping = {group_name: "teacher" for group_name in teacher_checkpoint_path.keys()}
            print(f"[MOSAIC] No teacher_obs_source_mapping provided, defaulting all teachers to 'teacher' observations")

        if self.use_multi_teacher:
            print(f"[MOSAIC] Multi-teacher mode enabled with {len(teacher_checkpoint_path)} teachers")
            self.teacher_policies = {}
            self.teacher_normalizers = {}

            for group_name, checkpoint_path in teacher_checkpoint_path.items():
                print(f"[MOSAIC] Loading teacher for group '{group_name}': {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                state_dict = checkpoint["model_state_dict"]

                def _iter_linear_layers(prefix: str):
                    layers = []
                    for key, value in state_dict.items():
                        if not (key.startswith(prefix) and key.endswith(".weight")):
                            continue
                        if value.ndim != 2:
                            continue
                        parts = key.split(".")
                        if len(parts) < 2 or not parts[1].isdigit():
                            continue
                        layers.append((int(parts[1]), value))
                    layers.sort(key=lambda item: item[0])
                    return layers

                actor_layers = _iter_linear_layers("actor.")
                critic_layers = _iter_linear_layers("critic.")
                if not actor_layers or not critic_layers:
                    raise ValueError(
                        f"[MOSAIC] Could not infer teacher architecture from checkpoint: {checkpoint_path}"
                    )

                teacher_actor_input_dim = actor_layers[0][1].shape[1]
                teacher_critic_input_dim = critic_layers[0][1].shape[1]

                if hasattr(self.policy, 'std'):
                    num_actions = self.policy.std.shape[0]
                elif hasattr(self.policy, 'log_std'):
                    num_actions = self.policy.log_std.shape[0]
                else:
                    num_actions = state_dict["std"].shape[0] if "std" in state_dict else state_dict["log_std"].shape[0]

                actor_hidden_dims = [layer.shape[0] for _, layer in actor_layers if layer.shape[0] != num_actions]
                critic_hidden_dims = [layer.shape[0] for _, layer in critic_layers if layer.shape[0] != 1]

                policy_cfg_from_file = None
                try:
                    import os
                    import yaml
                    params_yaml_path = os.path.join(os.path.dirname(checkpoint_path), "params", "agent.yaml")
                    if os.path.exists(params_yaml_path):
                        with open(params_yaml_path, "r") as f:
                            checkpoint_cfg = yaml.safe_load(f)
                        if isinstance(checkpoint_cfg, dict) and "policy" in checkpoint_cfg:
                            policy_cfg_from_file = checkpoint_cfg["policy"]
                except Exception as exc:
                    print(f"[MOSAIC]   WARNING: Failed to load policy config from checkpoint params: {exc}")

                print(f"[MOSAIC]   Group '{group_name}' teacher architecture: actor_input={teacher_actor_input_dim}, critic_input={teacher_critic_input_dim}, actions={num_actions}")
                print(f"[MOSAIC]   Inferred actor_hidden_dims={actor_hidden_dims}, critic_hidden_dims={critic_hidden_dims}")

                current_teacher_policy_cfg = {}
                if policy_cfg_from_file:
                    for key in ("actor_hidden_dims", "critic_hidden_dims", "activation", "noise_std_type", "init_noise_std"):
                        if key in policy_cfg_from_file:
                            current_teacher_policy_cfg[key] = policy_cfg_from_file[key]
                if teacher_policy_cfg is not None:
                    current_teacher_policy_cfg.update(teacher_policy_cfg)
                if "init_noise_std" not in current_teacher_policy_cfg:
                    current_teacher_policy_cfg["init_noise_std"] = (
                        self.policy.std.data[0].item() if hasattr(self.policy, "std") else 1.0
                    )
                if "actor_hidden_dims" not in current_teacher_policy_cfg:
                    current_teacher_policy_cfg["actor_hidden_dims"] = actor_hidden_dims
                if "critic_hidden_dims" not in current_teacher_policy_cfg:
                    current_teacher_policy_cfg["critic_hidden_dims"] = critic_hidden_dims
                if "activation" not in current_teacher_policy_cfg:
                    current_teacher_policy_cfg["activation"] = "elu"
                if "ref_vel_skip_first_layer" not in current_teacher_policy_cfg:
                    current_teacher_policy_cfg["ref_vel_skip_first_layer"] = False

                teacher_policy_instance = ActorCritic(
                    num_actor_obs=teacher_actor_input_dim,
                    num_critic_obs=teacher_critic_input_dim,
                    num_actions=num_actions,
                    **current_teacher_policy_cfg
                ).to(self.device)

                teacher_policy_instance.load_state_dict(state_dict)

                teacher_normalizer_instance = None
                if "obs_norm_state_dict" in checkpoint:
                    from rsl_rl.modules import EmpiricalNormalization
                    teacher_normalizer_instance = EmpiricalNormalization(
                        shape=[teacher_actor_input_dim], until=1.0e8
                    ).to(self.device)
                    teacher_normalizer_instance.load_state_dict(checkpoint["obs_norm_state_dict"])
                    teacher_normalizer_instance.until = 0 
                    teacher_normalizer_instance.eval()
                    print(f"[MOSAIC]   Group '{group_name}' teacher normalizer loaded (frozen)")
                else:
                    print(f"[MOSAIC]   WARNING: No normalizer found for group '{group_name}' teacher!")

                self.teacher_policies[group_name] = teacher_policy_instance
                self.teacher_normalizers[group_name] = teacher_normalizer_instance

                print(f"[MOSAIC]   Group '{group_name}' teacher loaded successfully!")

            self.group_name_to_idx = {name: idx for idx, name in enumerate(sorted(self.teacher_policies.keys()))}
            self.idx_to_group_name = {idx: name for name, idx in self.group_name_to_idx.items()}
            print(f"[MOSAIC] Group name to index mapping: {self.group_name_to_idx}")

            self.env_group_name_to_idx = None

            print(f"[MOSAIC] Teacher observation sources:")
            for group_name in sorted(self.teacher_policies.keys()):
                obs_source = self.teacher_obs_source_mapping.get(group_name, "teacher")
                print(f"[MOSAIC]   - Group '{group_name}': using '{obs_source}' observations")

            self.teacher_policy = None
            self.teacher_normalizer = None

        else:
            self.teacher_normalizer = None 
            if teacher_checkpoint_path is not None and teacher_policy is None:
                print(f"[MOSAIC] Loading teacher policy from: {teacher_checkpoint_path}")
                checkpoint = torch.load(teacher_checkpoint_path, map_location=self.device, weights_only=False)

                state_dict = checkpoint["model_state_dict"]
                teacher_actor_input_dim = state_dict["actor.0.weight"].shape[1]
                teacher_critic_input_dim = state_dict["critic.0.weight"].shape[1]

                if hasattr(self.policy, 'std'):
                    num_actions = self.policy.std.shape[0]
                elif hasattr(self.policy, 'log_std'):
                    num_actions = self.policy.log_std.shape[0]
                else:
                    num_actions = state_dict["std"].shape[0] if "std" in state_dict else state_dict["log_std"].shape[0]

                print(f"[MOSAIC] Teacher architecture: actor_input={teacher_actor_input_dim}, critic_input={teacher_critic_input_dim}, actions={num_actions}")

                if teacher_policy_cfg is None:
                    if hasattr(self.policy, 'ref_vel_skip_first_layer') and self.policy.ref_vel_skip_first_layer:
                        layer1_dim = self.policy.actor_layer1.out_features
                        remaining_dims = [layer.out_features for layer in self.policy.actor_remaining if hasattr(layer, 'out_features')][:-1]
                        actor_hidden_dims = [layer1_dim] + remaining_dims
                    else:
                        actor_hidden_dims = [layer.out_features for layer in self.policy.actor if hasattr(layer, 'out_features')][:-1]

                    teacher_policy_cfg = {
                        "init_noise_std": self.policy.std.data[0].item() if hasattr(self.policy, 'std') else 1.0,
                        "actor_hidden_dims": actor_hidden_dims,
                        "critic_hidden_dims": [layer.out_features for layer in self.policy.critic if hasattr(layer, 'out_features')][:-1],
                        "activation": "elu", 
                        "ref_vel_skip_first_layer": False,
                    }

                self.teacher_policy = ActorCritic(
                    num_actor_obs=teacher_actor_input_dim,
                    num_critic_obs=teacher_critic_input_dim,
                    num_actions=num_actions,
                    **teacher_policy_cfg
                ).to(self.device)

                self.teacher_policy.load_state_dict(state_dict)

                if "obs_norm_state_dict" in checkpoint:
                    from rsl_rl.modules import EmpiricalNormalization
                    self.teacher_normalizer = EmpiricalNormalization(
                        shape=[teacher_actor_input_dim], until=1.0e8
                    ).to(self.device)
                    self.teacher_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
                    self.teacher_normalizer.until = 0 
                    self.teacher_normalizer.eval()
                    print("[MOSAIC] Teacher observation normalizer loaded from checkpoint (frozen)")
                else:
                    print("[MOSAIC] WARNING: No normalizer found in teacher checkpoint. Teacher may produce incorrect actions!")

                print("[MOSAIC] Teacher policy loaded successfully!")
            else:
                self.teacher_policy = teacher_policy

        self.lambda_teacher_init = lambda_teacher_init
        self.lambda_teacher_decay = lambda_teacher_decay
        self.lambda_teacher_min = lambda_teacher_min
        self.lambda_teacher_current = lambda_teacher_init
        self.use_teacher_bc = (self.teacher_policy is not None) or (self.use_multi_teacher and len(self.teacher_policies) > 0)
        self.teacher_loss_type = teacher_loss_type
        if self.teacher_loss_type not in ["kl", "mse"]:
            raise ValueError(f"teacher_loss_type must be 'kl' or 'mse', got {self.teacher_loss_type}")

        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")

        if not self.hybrid and (self.use_ppo or not self.use_teacher_bc):
            raise ValueError(
                f"hybrid=False (pure BC mode) requires use_ppo=False and teacher BC enabled, "
                f"but got use_ppo={self.use_ppo}, teacher_bc={self.use_teacher_bc}"
            )

        if self.use_multi_teacher:
            for group_name, teacher_policy in self.teacher_policies.items():
                teacher_policy.eval()
                for param in teacher_policy.parameters():
                    param.requires_grad = False
            print(f"[MOSAIC] All {len(self.teacher_policies)} teachers frozen")
        elif self.teacher_policy is not None:
            self.teacher_policy.eval()
            for param in self.teacher_policy.parameters():
                param.requires_grad = False
            print("[MOSAIC] Teacher policy frozen")

        self.teacher_critic_normalizer = None
        if self.teacher_critic_checkpoint_path is not None:
            print(f"[MOSAIC] Loading teacher critic from: {self.teacher_critic_checkpoint_path}")
            critic_checkpoint = torch.load(self.teacher_critic_checkpoint_path, map_location=self.device, weights_only=False)

            critic_state_dict = critic_checkpoint["model_state_dict"]
            critic_only_state_dict = {k.replace("critic.", ""): v for k, v in critic_state_dict.items() if k.startswith("critic.")}

            if len(critic_only_state_dict) == 0:
                raise ValueError(f"No critic weights found in checkpoint: {self.teacher_critic_checkpoint_path}")

            critic_input_dim = critic_only_state_dict["0.weight"].shape[1]
            print(f"[MOSAIC] Teacher critic input dim: {critic_input_dim}")

            self.policy.critic.load_state_dict(critic_only_state_dict)
            print("[MOSAIC] Teacher critic weights loaded into student's critic")

            if "privileged_obs_norm_state_dict" in critic_checkpoint:
                from rsl_rl.modules import EmpiricalNormalization
                self.teacher_critic_normalizer = EmpiricalNormalization(
                    shape=[critic_input_dim], until=1.0e8
                ).to(self.device)
                self.teacher_critic_normalizer.load_state_dict(critic_checkpoint["privileged_obs_norm_state_dict"])
                if self.teacher_critic_frozen:
                    self.teacher_critic_normalizer.until = 0
                    self.teacher_critic_normalizer.eval()
                print(f"[MOSAIC] Teacher critic normalizer loaded ({'frozen' if self.teacher_critic_frozen else 'trainable'})")

            if self.teacher_critic_frozen:
                for param in self.policy.critic.parameters():
                    param.requires_grad = False
                print("[MOSAIC] Teacher critic frozen (no fine-tuning)")
            else:
                for param in self.policy.critic.parameters():
                    param.requires_grad = True
                print("[MOSAIC] Teacher critic trainable (fine-tuning enabled)")

            if self.teacher_critic_frozen:
                self.optimizer = optim.Adam(self.policy.actor.parameters(), lr=learning_rate)
                print("[MOSAIC] Optimizer recreated to only update actor")
            else:
                self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
                print("[MOSAIC] Optimizer recreated to update both actor and critic")

        if self.use_off_policy_bc:
            self._load_expert_trajectories()

        self._print_init_summary()

    def _print_init_summary(self):
        """Print initialization summary"""
        print("=" * 80)
        print("MOSAIC: PPO + BC Hybrid Learning")
        print("=" * 80)
        print("Optimizer:")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps} steps{' (DISABLED)' if self.gradient_accumulation_steps == 1 else ''}")
        print()
        print("Off-policy BC:")
        print(f"  Enabled: {self.use_off_policy_bc}")
        if self.use_off_policy_bc:
            print(f"  Lambda: {self.lambda_off_policy_current} (decay: {self.lambda_off_policy_decay}, min: {self.lambda_off_policy_min})")
            print(f"  Loss type: {self.expert_loss_type}")
        print()
        print("Teacher BC:")
        print(f"  Enabled: {self.use_teacher_bc}")
        if self.use_teacher_bc:
            print(f"  Lambda: {self.lambda_teacher_current} (decay: {self.lambda_teacher_decay}, min: {self.lambda_teacher_min})")
            print(f"  Loss type: {self.teacher_loss_type} ({['KL divergence (matches mean+variance)', 'MSE (matches mean only)'][self.teacher_loss_type == 'mse']})")
        print("=" * 80)

    def _load_expert_trajectories(self):
        """Load pre-collected expert trajectories for off-policy BC"""
        if not self.use_off_policy_bc or self.expert_trajectory_path is None:
            return

        print("=" * 80)
        print(f"Loading Expert Trajectories for Off-Policy BC:")
        print(f"  Path: {self.expert_trajectory_path}")
        print(f"  Off-Policy Lambda: {self.lambda_off_policy_init} -> {self.lambda_off_policy_min} (decay: {self.lambda_off_policy_decay})")
        print(f"  Batch size: {self.off_policy_batch_size}")
        print(f"  Allow repeat sampling: {self.expert_allow_repeat_sampling}")

        try:
            if self.expert_trajectory_path.endswith('.npy'):
                loaded_data = np.load(self.expert_trajectory_path, allow_pickle=True).item()
                self.expert_data = {}
                for key, value in loaded_data.items():
                    if isinstance(value, np.ndarray):
                        self.expert_data[key] = torch.from_numpy(value).to(self.device)
                    else:
                        self.expert_data[key] = value
            elif self.expert_trajectory_path.endswith('.pt'):
                loaded_data = torch.load(self.expert_trajectory_path, map_location=self.device)
                self.expert_data = loaded_data
            else:
                raise ValueError(f"Unsupported file format: {self.expert_trajectory_path}")

            # Flatten expert data from [T, N, dim] to [T*N, dim] for efficient sampling
            flattened_data = {}
            total_steps = None

            obs_key = "observations"
            if "student_observations" in self.expert_data:
                obs_key = "student_observations"
                print("  Using student_observations for off-policy BC")

            for key in [obs_key, 'actions', 'action_mean', 'action_sigma']:
                if key in self.expert_data and isinstance(self.expert_data[key], torch.Tensor):
                    original_shape = self.expert_data[key].shape
                    if len(original_shape) >= 2:
                        T, N = original_shape[0], original_shape[1]
                        out_key = "observations" if key == obs_key else key
                        flattened_data[out_key] = self.expert_data[key].view(T * N, -1)
                        if total_steps is None:
                            total_steps = T * N
                            print(f"  Expert data shape: T={T}, N={N}, total_pairs={total_steps:,}")

            self.expert_data = flattened_data

            if total_steps is not None:
                self.expert_data_indices = torch.arange(total_steps, device=self.device)
                print(f"  Total expert pairs: {total_steps:,}")
                print(f"  Expert data ready for sampling")
            else:
                print("  Warning: Could not determine total steps from expert data")
                self.use_off_policy_bc = False

            print("=" * 80)

        except Exception as e:
            print(f"Failed to load expert trajectories: {e}")
            print("Disabling off-policy BC")
            self.use_off_policy_bc = False
            self.expert_data = None

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        teacher_obs_shape=None,
        ref_vel_estimator_obs_shape=None,
        ):
        self.ref_vel_estimator_obs_shape = ref_vel_estimator_obs_shape

        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            None,
            self.device,
            teacher_obs_shape=teacher_obs_shape,
            ref_vel_estimator_obs_shape=ref_vel_estimator_obs_shape,
        )

    def act(self, obs, critic_obs, teacher_obs=None, ref_vel_estimator_obs=None, motion_groups=None):
        """
        Generate actions for the current observations.

        Args:
            obs: Policy observations (student observations in distillation mode)
            critic_obs: Privileged observations used by the critic
            teacher_obs: Teacher observations used for teacher BC (if provided)
            ref_vel_estimator_obs: Observations for reference velocity estimator (optional)
                                   Should contain: [ref_dof_pos, ref_dof_vel, ref_projected_gravity] with history
            motion_groups: Motion group indices for multi-teacher support (optional)
        """
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
            estimator_input = ref_vel_estimator_obs if ref_vel_estimator_obs is not None else obs

            with torch.no_grad():
                estimated_ref_vel = self.ref_vel_estimator(estimator_input)  # [N, 3]

                self.last_estimated_ref_vel = estimated_ref_vel.clone()

                obs_augmented = torch.cat([obs, estimated_ref_vel], dim=-1)  # [N, obs_dim + 3]
        else:
            obs_augmented = obs
            self.last_estimated_ref_vel = None

        if not self.use_ppo and self.use_teacher_bc:
            self.policy.update_distribution(obs_augmented)
            self.policy.distribution.sample()
            self.transition.actions = self.policy.act_inference(obs_augmented).detach()
        else:
            self.transition.actions = self.policy.act(obs_augmented).detach()

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        if teacher_obs is None:
            teacher_obs = critic_obs
        self.transition.teacher_observations = teacher_obs
        self.transition.ref_vel_estimator_observations = ref_vel_estimator_obs
        self.transition.motion_groups = motion_groups
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        if not self.use_ppo and self.use_teacher_bc and not self.train_critic_during_distillation:
            return

        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _sample_expert_batch(self):
        """Sample a batch from expert trajectories

        Supports two modes:
        1. expert_allow_repeat_sampling=False (default): batch_size limited by dataset size
        2. expert_allow_repeat_sampling=True: batch_size can exceed dataset size via sampling with replacement
        """
        if not self.use_off_policy_bc or self.expert_data is None or self.expert_data_indices is None:
            return None

        dataset_size = len(self.expert_data_indices)
        target_batch_size = self.off_policy_batch_size

        if self.expert_allow_repeat_sampling:
            batch_size = target_batch_size
            sampled_indices = torch.randint(0, dataset_size, (batch_size,), device=self.device)
            indices = self.expert_data_indices[sampled_indices]
        else:
            batch_size = min(target_batch_size, dataset_size)
            sampled_indices = torch.randperm(dataset_size, device=self.device)[:batch_size]
            indices = self.expert_data_indices[sampled_indices]

        expert_batch = {}
        for key in ['observations', 'actions', 'action_mean', 'action_sigma']:
            if key in self.expert_data:
                expert_batch[key] = self.expert_data[key][indices]

        return expert_batch

    def _update_teacher_lambda_schedule(self):
        """Update teacher lambda with exponential decay"""
        if self.use_teacher_bc:
            self.lambda_teacher_current *= self.lambda_teacher_decay
            self.lambda_teacher_current = max(self.lambda_teacher_current, self.lambda_teacher_min)

    def _update_off_policy_lambda_schedule(self):
        """Update off-policy BC lambda with exponential decay"""
        if self.use_off_policy_bc:
            self.lambda_off_policy_current *= self.lambda_off_policy_decay
            self.lambda_off_policy_current = max(self.lambda_off_policy_current, self.lambda_off_policy_min)

    def update(self):
        """MOSAIC update: PPO + Off-Policy BC + Optional Teacher BC

        Mode selection (controlled by self.hybrid):
        - hybrid=False: Pure BC mode (_update_pure_bc) - sequential data, gradient accumulation, deterministic actions
        - hybrid=True: Hybrid mode (_update_hybrid) - random mini-batches, per-batch updates, stochastic actions
        """

        if not self.hybrid:
            return self._update_pure_bc()
        else:
            return self._update_hybrid()

    def _update_pure_bc(self):
        """Pure BC update (exactly like Distillation): sequential data traversal + gradient accumulation

        Compute teacher actions on-demand from privileged_obs (same as Distillation.update())
        """
        mean_bc_teacher_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            epoch_loss_sum = 0
            epoch_samples = 0

            if self.policy.is_recurrent:
                self.policy.reset(hidden_states=getattr(self, 'last_hidden_states', None))
                self.policy.detach_hidden_states()
            else:
                self.policy.reset()

            for t in range(self.storage.num_transitions_per_env):
                obs = self.storage.observations[t]
                privileged_obs = self.storage.privileged_observations[t]
                if self.storage.teacher_observations is not None:
                    teacher_obs = self.storage.teacher_observations[t]
                else:
                    teacher_obs = privileged_obs
                if self.storage.ref_vel_estimator_observations is not None:
                    ref_vel_estimator_obs = self.storage.ref_vel_estimator_observations[t]
                else:
                    ref_vel_estimator_obs = None
                dones = self.storage.dones[t]

                if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
                    with torch.no_grad():
                        estimator_input = ref_vel_estimator_obs if ref_vel_estimator_obs is not None else obs
                        estimated_ref_vel = self.ref_vel_estimator(estimator_input)
                        obs_augmented = torch.cat([obs, estimated_ref_vel], dim=-1)
                else:
                    obs_augmented = obs

                student_actions = self.policy.act_inference(obs_augmented)

                with torch.no_grad():
                    teacher_actions = self.teacher_policy.act_inference(teacher_obs)

                behavior_loss = nn.functional.mse_loss(student_actions, teacher_actions)

                loss = loss + behavior_loss
                mean_bc_teacher_loss += behavior_loss.item()
                epoch_loss_sum += behavior_loss.item()
                epoch_samples += 1
                cnt += 1

                if cnt % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    if self.policy.is_recurrent:
                        self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                if self.policy.is_recurrent:
                    self.policy.detach_hidden_states(dones.view(-1))

            avg_epoch_loss = epoch_loss_sum / epoch_samples

        mean_bc_teacher_loss /= max(cnt, 1)

        self.storage.clear()

        if self.policy.is_recurrent:
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()
        else:
            self.last_hidden_states = None

        self._update_teacher_lambda_schedule()

        return {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "bc_off_policy": 0.0,
            "bc_teacher": mean_bc_teacher_loss,
            "lambda_off_policy": self.lambda_off_policy_current,
            "lambda_teacher": self.lambda_teacher_current,
        }

    def _update_hybrid(self):
        """Hybrid update (PPO + BC): random mini-batches"""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_bc_off_policy_loss = 0
        mean_bc_teacher_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
            teacher_obs_batch,
            teacher_mu_batch, 
            teacher_sigma_batch,
            ref_vel_estimator_obs_batch, 
            motion_groups_batch,
        ) in generator:
            original_batch_size = obs_batch.shape[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
                with torch.no_grad():
                    estimator_input = ref_vel_estimator_obs_batch if ref_vel_estimator_obs_batch is not None else obs_batch
                    estimated_ref_vel_batch = self.ref_vel_estimator(estimator_input)
                    obs_batch_augmented = torch.cat([obs_batch, estimated_ref_vel_batch], dim=-1)
            else:
                obs_batch_augmented = obs_batch

            # ========== On-Policy Forward Pass (PPO) ==========
            self.policy.act(obs_batch_augmented, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # ========== Adaptive Learning Rate (KL-based) ==========
            if self.use_ppo and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ========== PPO Losses ==========
            if self.use_ppo:
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
            else:
                # PPO disabled: set surrogate loss to zero
                surrogate_loss = torch.tensor(0.0, device=self.device)
                entropy_batch = torch.tensor(0.0, device=self.device)

                # Value loss: train critic if flag is enabled
                if self.train_critic_during_distillation:
                    # Train critic via value loss (for distillation stage)
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                            -self.clip_param, self.clip_param
                        )
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()
                else:
                    # Don't train critic
                    value_loss = torch.tensor(0.0, device=self.device)

            # ========== Teacher BC Loss ==========
            bc_teacher_loss = torch.tensor(0.0, device=self.device)

            if self.use_teacher_bc:
                if self.use_multi_teacher:
                    if teacher_obs_batch is None:
                        teacher_obs_batch = critic_obs_batch

                    for group_name, teacher_policy in self.teacher_policies.items():
                        if self.env_group_name_to_idx is not None:
                            if group_name not in self.env_group_name_to_idx:
                                continue 
                            group_idx = self.env_group_name_to_idx[group_name]
                        else:
                            if group_name not in self.group_name_to_idx:
                                continue
                            group_idx = self.group_name_to_idx[group_name]

                        group_mask = (motion_groups_batch == group_idx)

                        if group_mask.sum() == 0:
                            continue

                        obs_source = self.teacher_obs_source_mapping.get(group_name, "teacher")
                        if obs_source == "policy":
                            obs_for_teacher = obs_batch
                        elif obs_source == "critic":
                            obs_for_teacher = critic_obs_batch
                        else:  # "teacher" or default
                            obs_for_teacher = teacher_obs_batch

                        # Extract observations and actions for this group
                        teacher_obs_group = obs_for_teacher[group_mask]
                        mu_group = mu_batch[group_mask]
                        sigma_group = sigma_batch[group_mask]

                        if group_name in self.teacher_normalizers and self.teacher_normalizers[group_name] is not None:
                            if obs_source == "teacher":
                                teacher_obs_group = self.teacher_normalizers[group_name](teacher_obs_group)
                            elif obs_source == "policy":
                                if self.obs_normalizer is not None and hasattr(self.obs_normalizer, '_mean') and hasattr(self.obs_normalizer, '_std'):
                                    teacher_obs_group = teacher_obs_group * self.obs_normalizer._std + self.obs_normalizer._mean
                                    teacher_obs_group = self.teacher_normalizers[group_name](teacher_obs_group)
                                else:
                                    teacher_obs_group = self.teacher_normalizers[group_name](teacher_obs_group)
                            elif obs_source == "critic":
                                if self.privileged_obs_normalizer is not None and hasattr(self.privileged_obs_normalizer, '_mean') and hasattr(self.privileged_obs_normalizer, '_std'):
                                    teacher_obs_group = teacher_obs_group * self.privileged_obs_normalizer._std + self.privileged_obs_normalizer._mean
                                    teacher_obs_group = self.teacher_normalizers[group_name](teacher_obs_group)
                                else:
                                    teacher_obs_group = self.teacher_normalizers[group_name](teacher_obs_group)

                        with torch.no_grad():
                            teacher_mu_group = teacher_policy.act_inference(teacher_obs_group)
                            if self.teacher_loss_type == "kl" and hasattr(teacher_policy, 'std'):
                                teacher_sigma_group = teacher_policy.std.expand_as(teacher_mu_group)
                            else:
                                teacher_sigma_group = None

                        if self.teacher_loss_type == "kl" and teacher_sigma_group is not None:
                            student_dist = dist.Normal(mu_group, sigma_group)
                            teacher_dist = dist.Normal(teacher_mu_group, teacher_sigma_group)
                            group_loss = dist.kl_divergence(student_dist, teacher_dist).mean()
                        elif self.teacher_loss_type == "mse":
                            group_loss = nn.functional.mse_loss(mu_group, teacher_mu_group)
                        else:
                            group_loss = torch.tensor(0.0, device=self.device)

                        group_weight = group_mask.sum().float() / len(motion_groups_batch)
                        bc_teacher_loss += group_loss * group_weight

                        if self.current_learning_iteration % self.log_interval == 0:
                            with torch.no_grad():
                                action_diff = mu_group - teacher_mu_group
                                action_diff_l2 = torch.norm(action_diff, dim=-1).mean().item()

                                if hasattr(self, "writer"):
                                    self.writer.add_scalar(
                                        f"BC_Loss/{group_name}_loss",
                                        group_loss.item(),
                                        self.current_learning_iteration
                                    )
                                    self.writer.add_scalar(
                                        f"BC_Loss/{group_name}_weight",
                                        group_weight.item(),
                                        self.current_learning_iteration
                                    )
                                    self.writer.add_scalar(
                                        f"BC_Loss/{group_name}_weighted_loss",
                                        (group_loss * group_weight).item(),
                                        self.current_learning_iteration
                                    )
                                    self.writer.add_scalar(
                                        f"BC_Loss/{group_name}_action_diff_l2",
                                        action_diff_l2,
                                        self.current_learning_iteration
                                    )
                                    self.writer.add_scalar(
                                        f"BC_Loss/{group_name}_num_samples",
                                        group_mask.sum().item(),
                                        self.current_learning_iteration
                                    )

                    if self.current_learning_iteration % self.log_interval == 0:
                        if hasattr(self.policy, 'last_gmt_actions') and hasattr(self.policy, 'last_residual_actions'):
                            with torch.no_grad():
                                gmt_actions_batch = self.policy.last_gmt_actions
                                residual_actions_batch = self.policy.last_residual_actions

                                for group_name in self.teacher_policies.keys():
                                    if self.env_group_name_to_idx is not None:
                                        if group_name not in self.env_group_name_to_idx:
                                            continue
                                        group_idx = self.env_group_name_to_idx[group_name]
                                    else:
                                        if group_name not in self.group_name_to_idx:
                                            continue
                                        group_idx = self.group_name_to_idx[group_name]

                                    group_mask = (motion_groups_batch == group_idx)
                                    if group_mask.sum() == 0:
                                        continue

                                    gmt_group = gmt_actions_batch[group_mask]
                                    residual_group = residual_actions_batch[group_mask]

                                    gmt_l2 = torch.norm(gmt_group, dim=-1).mean().item()
                                    residual_l2 = torch.norm(residual_group, dim=-1).mean().item()
                                    residual_ratio = residual_l2 / (gmt_l2 + 1e-8)

                                    if hasattr(self, "writer"):
                                        self.writer.add_scalar(
                                            f"Residual/{group_name}_gmt_l2",
                                            gmt_l2,
                                            self.current_learning_iteration
                                        )
                                        self.writer.add_scalar(
                                            f"Residual/{group_name}_residual_l2",
                                            residual_l2,
                                            self.current_learning_iteration
                                        )
                                        self.writer.add_scalar(
                                            f"Residual/{group_name}_residual_ratio",
                                            residual_ratio,
                                            self.current_learning_iteration
                                        )

                elif self.teacher_policy is not None:
                    if teacher_obs_batch is None:
                        teacher_obs_batch = critic_obs_batch
                    with torch.no_grad():
                        teacher_mu_batch = self.teacher_policy.act_inference(teacher_obs_batch)
                        if self.teacher_loss_type == "kl" and hasattr(self.teacher_policy, 'std'):
                            teacher_sigma_batch = self.teacher_policy.std.expand_as(teacher_mu_batch)
                        else:
                            teacher_sigma_batch = None

                    if self.teacher_loss_type == "kl" and teacher_sigma_batch is not None:
                        student_dist = dist.Normal(mu_batch, sigma_batch)
                        teacher_dist = dist.Normal(teacher_mu_batch, teacher_sigma_batch)
                        bc_teacher_loss = dist.kl_divergence(student_dist, teacher_dist).mean()
                    elif self.teacher_loss_type == "mse":
                        bc_teacher_loss = nn.functional.mse_loss(mu_batch, teacher_mu_batch)

            # ========== Off-Policy Expert BC Loss ==========
            bc_off_policy_loss = torch.tensor(0.0, device=self.device)
            if self.use_off_policy_bc and self.lambda_off_policy_current > 0:
                expert_batch = self._sample_expert_batch()
                if expert_batch is not None and 'observations' in expert_batch:
                    expert_obs = expert_batch['observations'].detach()

                    if self.expert_normalize_obs and self.obs_normalizer is not None:
                        if not self.expert_update_normalizer:
                            was_training = self.obs_normalizer.training
                            self.obs_normalizer.eval()
                            expert_obs = self.obs_normalizer(expert_obs)
                            if was_training:
                                self.obs_normalizer.train()
                        else:
                            expert_obs = self.obs_normalizer(expert_obs)

                    if self.use_estimate_ref_vel and self.ref_vel_estimator is not None:
                        with torch.no_grad():
                            estimated_ref_vel_expert = self.ref_vel_estimator(expert_obs)
                            expert_obs_augmented = torch.cat([expert_obs, estimated_ref_vel_expert], dim=-1)
                    else:
                        expert_obs_augmented = expert_obs

                    student_mu_batch = self.policy.act_inference(expert_obs_augmented)
                    student_sigma_batch = None
                    if self.expert_loss_type == "kl":
                        if self.policy.noise_std_type == "scalar":
                            student_sigma_batch = self.policy.std.expand_as(student_mu_batch)
                        elif self.policy.noise_std_type == "log":
                            student_sigma_batch = torch.exp(self.policy.log_std).expand_as(student_mu_batch)
                        else:
                            raise ValueError(
                                f"Unknown standard deviation type: {self.policy.noise_std_type}. "
                                "Should be 'scalar' or 'log'"
                            )

                    expert_mu_batch = expert_batch.get('action_mean')
                    expert_sigma_batch = expert_batch.get('action_sigma')
                    expert_actions = expert_batch.get('actions')

                    if self.expert_loss_type == "kl" and student_sigma_batch is not None:
                        if expert_mu_batch is not None and expert_sigma_batch is not None:
                            expert_mu_batch = expert_mu_batch.detach()
                            expert_sigma_batch = expert_sigma_batch.detach()
                            student_dist = dist.Normal(student_mu_batch, student_sigma_batch)
                            expert_dist = dist.Normal(expert_mu_batch, expert_sigma_batch)
                            bc_off_policy_loss = dist.kl_divergence(student_dist, expert_dist).mean()
                        else:
                            if not self._warned_expert_sigma_missing:
                                print("[MOSAIC] WARNING: expert_loss_type='kl' but expert sigma missing; falling back to MSE.")
                                self._warned_expert_sigma_missing = True
                            target_actions = expert_mu_batch if expert_mu_batch is not None else expert_actions
                            if target_actions is not None:
                                bc_off_policy_loss = nn.functional.mse_loss(
                                    student_mu_batch, target_actions.detach()
                                )
                    else:
                        target_actions = expert_mu_batch if expert_mu_batch is not None else expert_actions
                        if target_actions is not None:
                            bc_off_policy_loss = nn.functional.mse_loss(
                                student_mu_batch, target_actions.detach()
                            )

            # ========== MOSAIC Combined Loss ==========
            loss = torch.tensor(0.0, device=self.device)

            if self.use_ppo:
                loss = loss + surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            loss = loss + self.lambda_off_policy_current * bc_off_policy_loss + self.lambda_teacher_current * bc_teacher_loss

            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # ========== Record Losses ==========
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_bc_off_policy_loss += bc_off_policy_loss.item()
            mean_bc_teacher_loss += bc_teacher_loss.item()

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_bc_off_policy_loss /= num_updates
        mean_bc_teacher_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Update lambda schedules
        self._update_teacher_lambda_schedule()
        self._update_off_policy_lambda_schedule()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_off_policy": mean_bc_off_policy_loss,
            "bc_teacher": mean_bc_teacher_loss,
            "lambda_off_policy": self.lambda_off_policy_current,
            "lambda_teacher": self.lambda_teacher_current,
        }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
