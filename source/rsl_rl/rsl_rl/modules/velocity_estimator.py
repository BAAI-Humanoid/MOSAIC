# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class VelocityEstimator(nn.Module):
    """
    MLP-based velocity estimator for reference motion tracking.

    Predicts ref_base_lin_vel_b from historical observations:
    - ref_dof_pos
    - ref_dof_vel
    - ref_projected_gravity

    Args:
        num_obs: Input observation dimension (includes history)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        use_skip_connections: Enable residual connections between layers (default: False)
        use_layer_norm: Enable layer normalization (default: False)
        dropout: Dropout probability for regularization (default: 0.0, disabled)
        use_input_skip: Enable direct input-to-output skip connection (default: False)
    """

    def __init__(
        self,
        num_obs: int,
        hidden_dims: list[int] = [256, 128, 64],
        activation: str = "elu",
        use_skip_connections: bool = False,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        use_input_skip: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "VelocityEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_obs = num_obs
        self.hidden_dims = hidden_dims
        self.use_skip_connections = use_skip_connections
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout
        self.use_input_skip = use_input_skip

        activation_fn = resolve_nn_activation(activation)

        # Build network based on architecture choice
        if use_skip_connections:
            # Build residual network with skip connections
            self._build_residual_network(activation_fn)
        else:
            # Build standard MLP (backward compatible)
            self._build_standard_mlp(activation_fn)

        print(f"VelocityEstimator: skip_conn={use_skip_connections}, layer_norm={use_layer_norm}, "
              f"dropout={dropout}, input_skip={use_input_skip}")

    def _build_standard_mlp(self, activation_fn):
        """Build standard MLP (original architecture)."""
        layers: list[nn.Module] = []
        prev_dim = self.num_obs

        if self.hidden_dims:
            layers.append(nn.Linear(prev_dim, self.hidden_dims[0]))
            layers.append(activation_fn)

            for layer_index in range(len(self.hidden_dims)):
                in_dim = self.hidden_dims[layer_index]
                if layer_index == len(self.hidden_dims) - 1:
                    # Output layer: 3D velocity
                    layers.append(nn.Linear(in_dim, 3))
                else:
                    out_dim = self.hidden_dims[layer_index + 1]
                    layers.append(nn.Linear(in_dim, out_dim))
                    layers.append(activation_fn)
        else:
            layers.append(nn.Linear(prev_dim, 3))

        self.net = nn.Sequential(*layers)

    def _build_residual_network(self, activation_fn):
        """Build residual network with skip connections."""
        self.input_proj = nn.Linear(self.num_obs, self.hidden_dims[0]) if self.hidden_dims else None

        # Build residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = self._make_residual_block(
                self.hidden_dims[i],
                self.hidden_dims[i + 1],
                activation_fn
            )
            self.residual_blocks.append(block)

        # Output projection
        final_dim = self.hidden_dims[-1] if self.hidden_dims else self.num_obs
        self.output_proj = nn.Linear(final_dim, 3)

        # Optional: input-to-output skip connection
        if self.use_input_skip:
            self.input_to_output = nn.Linear(self.num_obs, 3)

    def _make_residual_block(self, in_dim, out_dim, activation_fn):
        """Create a residual block with optional normalization and dropout."""
        layers = []

        # Layer normalization (before linear layer)
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(in_dim))

        # Linear transformation
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fn)

        # Dropout for regularization
        if self.dropout_prob > 0:
            layers.append(nn.Dropout(self.dropout_prob))

        block = nn.Sequential(*layers)

        # Skip connection (if dimensions match)
        if in_dim == out_dim:
            return ResidualBlock(block, skip=True)
        else:
            # Use projection for dimension mismatch
            projection = nn.Linear(in_dim, out_dim)
            return ResidualBlock(block, skip=True, projection=projection)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity from observations.

        Args:
            observations: [N, num_obs] Input observations with history

        Returns:
            velocity: [N, 3] Predicted ref_base_lin_vel_b
        """
        if self.use_skip_connections:
            # Residual network forward pass
            x = observations

            # Input projection
            if self.input_proj is not None:
                x = self.input_proj(x)

            # Pass through residual blocks
            for block in self.residual_blocks:
                x = block(x)

            # Output projection
            out = self.output_proj(x)

            # Optional: add input-to-output skip connection
            if self.use_input_skip:
                out = out + self.input_to_output(observations)

            return out
        else:
            # Standard MLP forward pass
            return self.net(observations)

    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Alias for forward (for compatibility)."""
        return self.forward(observations)

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_obs': self.num_obs,
            'hidden_dims': self.hidden_dims,
            'use_skip_connections': self.use_skip_connections,
            'use_layer_norm': self.use_layer_norm,
            'dropout': self.dropout_prob,
            'use_input_skip': self.use_input_skip,
        }
        torch.save(checkpoint, path)
        print(f"VelocityEstimator saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            num_obs=checkpoint['num_obs'],
            hidden_dims=checkpoint['hidden_dims'],
            use_skip_connections=checkpoint.get('use_skip_connections', False),
            use_layer_norm=checkpoint.get('use_layer_norm', False),
            dropout=checkpoint.get('dropout', 0.0),
            use_input_skip=checkpoint.get('use_input_skip', False),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        print(f"VelocityEstimator loaded from: {path}")
        print(f"  num_obs: {checkpoint['num_obs']}")
        print(f"  hidden_dims: {checkpoint['hidden_dims']}")

        return model

    def export_onnx(self, path: str, input_dim: int | None = None):
        """
        Export model to ONNX format.

        Args:
            path: Output path for ONNX model
            input_dim: Input dimension (if None, auto-detect from model)
        """
        if input_dim is None:
            input_dim = self.num_obs

        # Create dummy input
        dummy_input = torch.randn(1, input_dim, device=next(self.parameters()).device)

        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['observations'],
            output_names=['velocity'],
            dynamic_axes={
                'observations': {0: 'batch_size'},
                'velocity': {0: 'batch_size'}
            }
        )

        print(f"VelocityEstimator exported to ONNX: {path}")
        print(f"  Input shape: [batch_size, {input_dim}]")
        print(f"  Output shape: [batch_size, 3]")


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, block, skip=True, projection=None):
        super().__init__()
        self.block = block
        self.skip = skip
        self.projection = projection

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip:
            if self.projection is not None:
                identity = self.projection(identity)
            out = out + identity

        return out
