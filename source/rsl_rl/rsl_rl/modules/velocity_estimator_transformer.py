# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer (supports batch_first)."""

    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        if batch_first:
            # [1, max_len, d_model] for batch_first format
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            # [max_len, 1, d_model] for seq_first format
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] if batch_first else [seq_len, batch, d_model]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return x


class VelocityEstimatorTransformer(nn.Module):
    """
    Transformer-based velocity estimator for reference motion tracking.

    Uses self-attention to model temporal dependencies in historical observations.

    Args:
        feature_dim: Dimension of features per timestep (e.g., 61 = 29+29+3)
        history_length: Number of historical timesteps (including current)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
    """

    def __init__(
        self,
        feature_dim: int = 61,
        history_length: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "VelocityEstimatorTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.feature_dim = feature_dim
        self.history_length = history_length
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=history_length, batch_first=True)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Better inference performance
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # 3D velocity output
        )

        print(f"VelocityEstimatorTransformer:")
        print(f"  feature_dim={feature_dim}, history_length={history_length}")
        print(f"  d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity from observations.

        Args:
            observations: [N, feature_dim * history_length]
                         e.g., [N, 61*5=305]

        Returns:
            velocity: [N, 3] Predicted ref_base_lin_vel_b
        """
        batch_size = observations.shape[0]

        # Reshape: [batch, feature_dim * history_length] -> [batch, seq_len, feature_dim]
        x = observations.view(batch_size, self.history_length, self.feature_dim)

        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]

        # Use the last timestep (current frame) for prediction
        x = x[:, -1, :]  # [batch, d_model]

        # Output prediction
        velocity = self.output_head(x)  # [batch, 3]

        return velocity

    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Alias for forward (for compatibility)."""
        return self.forward(observations)

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'feature_dim': self.feature_dim,
            'history_length': self.history_length,
            'd_model': self.d_model,
            'config': {
                'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
                'num_layers': len(self.transformer_encoder.layers),
            }
        }
        torch.save(checkpoint, path)
        print(f"VelocityEstimatorTransformer saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            feature_dim=checkpoint['feature_dim'],
            history_length=checkpoint['history_length'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['config']['nhead'],
            num_layers=checkpoint['config']['num_layers'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        print(f"VelocityEstimatorTransformer loaded from: {path}")
        print(f"  feature_dim: {checkpoint['feature_dim']}")
        print(f"  history_length: {checkpoint['history_length']}")
        print(f"  d_model: {checkpoint['d_model']}")

        return model

    def export_onnx(self, path: str):
        """
        Export model to ONNX format.

        Args:
            path: Output path for ONNX model
        """
        input_dim = self.feature_dim * self.history_length

        # Create dummy input
        dummy_input = torch.randn(1, input_dim, device=next(self.parameters()).device)

        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=13,  # Use opset 13 for unflatten support
            do_constant_folding=True,
            input_names=['observations'],
            output_names=['velocity'],
            dynamic_axes={
                'observations': {0: 'batch_size'},
                'velocity': {0: 'batch_size'}
            }
        )

        print(f"VelocityEstimatorTransformer exported to ONNX: {path}")
        print(f"  Input shape: [batch_size, {input_dim}]")
        print(f"  Output shape: [batch_size, 3]")
