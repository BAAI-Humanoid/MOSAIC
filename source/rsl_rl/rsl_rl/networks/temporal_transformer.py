import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    """Temporal Transformer module for processing temporal observations.
    input: (batch_size, seq_len, num_observations)
    output: (batch_size, num_latent)

    """
    def __init__(
        self,
        obs_dim_per_step: int, 
        seq_len: int,
        d_model: int = 256,
        nhead: int = 4, 
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.obs_dim_per_step = obs_dim_per_step
        self.seq_len = seq_len
        self.d_model = d_model

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        self.input_proj = nn.Linear(obs_dim_per_step, d_model)

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 3, f"expected [B, T, D], got {x.shape}"
        B, T, D = x.shape
        if not torch.jit.is_tracing():
            assert T == self.seq_len, f"expected seq_len={self.seq_len}, got {T}"
            assert D == self.obs_dim_per_step, f"expected obs_dim_per_step={self.obs_dim_per_step}, got {D}"

        h = self.input_proj(x)  # [B, T, d_model]
        h = h + self.pos_embedding  # broadcast: [1, T, d_model] -> [B, T, d_model]

        # src_key_padding_mask
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        return h[:, -1, :]  # [B, d_model]


