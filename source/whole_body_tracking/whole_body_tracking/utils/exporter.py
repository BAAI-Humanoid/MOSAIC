# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from rsl_rl.networks import FSQEncoder, TemporalTransformer, VQEncoder


def export_motion_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    ref_vel_estimator: object | None = None,
    ref_vel_estimator_obs_dim: int | None = None,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(
        actor_critic, normalizer, verbose, ref_vel_estimator, ref_vel_estimator_obs_dim
    )
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, actor_critic, normalizer=None, verbose=False, ref_vel_estimator=None, ref_vel_estimator_obs_dim=None):
        super().__init__(actor_critic, normalizer, verbose)
        self._encoder_type: str = "mlp"
        self.actor_encoder = None

        # Handle both standard Sequential actor and ActorWithSkip
        if hasattr(self.actor, 'actor_layer1'):
            # ActorWithSkip mode: input = policy_obs + ref_vel
            self._policy_input_dim = self.actor.actor_layer1.in_features + actor_critic.ref_vel_dim
        else:
            # Standard Sequential mode
            self._policy_input_dim = self.actor[0].in_features

        self.num_actor_proprio = getattr(actor_critic, "num_actor_proprio", 0)

        # Velocity estimator support (use deepcopy to avoid moving original model to CPU during export)
        self.ref_vel_estimator = copy.deepcopy(ref_vel_estimator) if ref_vel_estimator is not None else None
        self.ref_vel_estimator_obs_dim = ref_vel_estimator_obs_dim

        # Get policy obs dimension from normalizer
        if normalizer is not None and hasattr(normalizer, '_mean'):
            # Use shape[-1] to get the last dimension (works for both (N,) and (1, N) shapes)
            # Normalizer normalizes policy_obs (without estimated_ref_vel)
            self.policy_obs_dim = normalizer._mean.shape[-1]
        else:
            self.policy_obs_dim = self._policy_input_dim

        encoder_src = getattr(actor_critic, "actor_encoder", None)

        has_attention = (
            hasattr(actor_critic, "reference_encoder")
            and hasattr(actor_critic, "query_encoder")
            and hasattr(actor_critic, "attention")
        )

        if has_attention:
            self._encoder_type = "attention"
            self.reference_encoder = copy.deepcopy(actor_critic.reference_encoder)
            self.query_encoder = copy.deepcopy(actor_critic.query_encoder)
            self.attention = copy.deepcopy(actor_critic.attention)
            if hasattr(actor_critic, "num_actor_obs"):
                self._policy_input_dim = actor_critic.num_actor_obs
            elif (
                hasattr(self.reference_encoder, "net")
                and len(self.reference_encoder.net) > 0
                and hasattr(self.reference_encoder.net[0], "in_features")
            ):
                self._policy_input_dim = self.reference_encoder.net[0].in_features + self.num_actor_proprio
        elif isinstance(encoder_src, TemporalTransformer):
            self._encoder_type = "transformer"
            self.seq_len = actor_critic.seq_len
            self.actor_obs_per_step = actor_critic.actor_obs_per_step
            self._policy_input_dim = self.seq_len * self.actor_obs_per_step
            self.actor_encoder = copy.deepcopy(encoder_src)
        elif isinstance(encoder_src, FSQEncoder):
            self._encoder_type = "fsq"
            self.actor_encoder = copy.deepcopy(encoder_src)
            if hasattr(self.actor_encoder, "mlp") and hasattr(self.actor_encoder.mlp, "net"):
                self._policy_input_dim = self.actor_encoder.mlp.net[0].in_features + self.num_actor_proprio
        elif isinstance(encoder_src, VQEncoder):
            self._encoder_type = "vq"
            vq_module = getattr(encoder_src, "vq", None)
            backup: dict[str, torch.Tensor | None] = {}
            self.actor_encoder = copy.deepcopy(encoder_src)
            if vq_module is not None:
                for attr, value in backup.items():
                    if hasattr(vq_module, attr):
                        setattr(vq_module, attr, value)
            if hasattr(self.actor_encoder, "mlp") and hasattr(self.actor_encoder.mlp, "net"):
                self._policy_input_dim = self.actor_encoder.mlp.net[0].in_features + self.num_actor_proprio
        else:
            self.actor_encoder = None

    def _compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        # Handle velocity estimator if present
        if self.ref_vel_estimator is not None and self.ref_vel_estimator_obs_dim is not None:
            # Split input: policy_obs + ref_vel_estimator_obs
            policy_obs = obs[:, :self.policy_obs_dim]
            ref_vel_estimator_obs = obs[:, self.policy_obs_dim:self.policy_obs_dim + self.ref_vel_estimator_obs_dim]

            # Normalize policy obs first (matches training logic)
            policy_obs = self.normalizer(policy_obs)

            # Estimate reference velocity (no normalization for estimator input)
            with torch.no_grad():
                estimated_ref_vel = self.ref_vel_estimator(ref_vel_estimator_obs)

            # Optimize for ActorWithSkip: avoid redundant concat->slice->concat
            if hasattr(self.actor, 'actor_layer1'):
                # ActorWithSkip mode: directly call internal components to avoid redundant ops
                # Pass policy_obs through first layer
                layer1_out = self.actor.actor_activation(self.actor.actor_layer1(policy_obs))
                # Concatenate layer1_out with estimated ref_vel (skip connection)
                layer2_input = torch.cat([layer1_out, estimated_ref_vel], dim=-1)
                # Pass through remaining layers
                return self.actor.actor_remaining(layer2_input)
            else:
                # Standard mode: concatenate and pass to actor
                obs = torch.cat([policy_obs, estimated_ref_vel], dim=-1)
        else:
            # Original behavior: normalize all obs
            obs = self.normalizer(obs)

        if self._encoder_type != "mlp":
            if not torch.jit.is_tracing() and obs.shape[-1] != self._policy_input_dim:
                raise ValueError(
                    f"Observation dim ({obs.shape[-1]}) does not match expected encoder input ({self._policy_input_dim})"
                )

        if self._encoder_type == "attention":
            if self.num_actor_proprio:
                ref_input = obs[:, :-self.num_actor_proprio]
                proprio = obs[:, -self.num_actor_proprio :]
            else:
                ref_input = obs
                proprio = obs[:, :0]
            key = self.reference_encoder(ref_input)
            query = self.query_encoder(proprio) if self.num_actor_proprio else key
            context, _ = self.attention(query.unsqueeze(1), key.unsqueeze(1), key.unsqueeze(1))
            obs = torch.cat([context.squeeze(1), proprio], dim=-1)
        elif self._encoder_type == "transformer":
            obs = obs.view(obs.shape[0], self.seq_len, self.actor_obs_per_step)
            obs = self.actor_encoder(obs)
        elif self._encoder_type == "fsq":
            latent, *_ = self.actor_encoder(obs[:, :-self.num_actor_proprio])
            if self.num_actor_proprio:
                proprio = obs[:, -self.num_actor_proprio :]
                obs = torch.cat([latent, proprio], dim=-1)
            else:
                obs = latent
        elif self._encoder_type == "vq":
            latent, *_ = self.actor_encoder(obs[:, :-self.num_actor_proprio])
            if self.num_actor_proprio:
                proprio = obs[:, -self.num_actor_proprio :]
                obs = torch.cat([latent, proprio], dim=-1)
            else:
                obs = latent

        return self.actor(obs)

    def forward(self, x):
        return self._compute_actions(x)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()

        # Determine input dimension based on whether velocity estimator is used
        if self.ref_vel_estimator is not None and self.ref_vel_estimator_obs_dim is not None:
            # Input = policy_obs + ref_vel_estimator_obs
            input_dim = self.policy_obs_dim + self.ref_vel_estimator_obs_dim
        else:
            # Original behavior
            input_dim = self._policy_input_dim

        obs = torch.zeros(1, input_dim)
        opset_version = 18
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=[
                "actions",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
