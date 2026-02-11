import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from typing import List, Tuple, Optional

from rsl_rl.utils import resolve_nn_activation


class MLP(nn.Module):
    """
    Simple MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "elu",
        last_activation: bool = False,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        act = resolve_nn_activation(activation)

        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)

        layers.append(nn.Linear(dims[-1], output_dim))
        if last_activation:
            layers.append(act)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSQ(nn.Module):
    """
    FSQ: Finite Scalar Quantization
    """
    def __init__(self, d_model: int, num_levels: int):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.scale = (num_levels - 1) / 2.0

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_bounded = self.scale * torch.tanh(z)
        z_rounded = torch.round(z_bounded)
        z_q = z_bounded + (z_rounded - z_bounded).detach()
        codes = (z_rounded + self.scale).long().clamp(0, self.num_levels - 1)
        return z_q, codes
    

class FSQEncoder(nn.Module):
    """
    FSQ Encoder.
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        num_levels: int,
        activation: str = "elu",
    ):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, latent_dim, activation)
        self.fsq = FSQ(latent_dim, num_levels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_cont = self.mlp(x)
        z_q, codes = self.fsq(z_cont)
        return z_q, codes, z_cont


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer.
    """
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int,
        commitment_weight: float = 0.25,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # flatten z_e
        flat_z_e = z_e.view(-1, self.embedding_dim)
        with torch.no_grad():
            if self.amp_enabled:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    flat_z_e_amp = flat_z_e.to(self.amp_dtype)
                    embedding_weight_amp = self.embedding.weight.to(self.amp_dtype)

                    z_sq = torch.sum(flat_z_e_amp**2, dim=1, keepdim=True)
                    e_sq = torch.sum(embedding_weight_amp**2, dim=1)
                    ze = flat_z_e_amp @ embedding_weight_amp.t()
                    distances = z_sq + e_sq - 2 * ze
                    encoding_indices = torch.argmin(distances, dim=1)
            else:
                e_sq = torch.sum(self.embedding.weight**2, dim=1)
                z_sq = torch.sum(flat_z_e**2, dim=1, keepdim=True)
                ze = flat_z_e @ self.embedding.weight.t()
                # calculate distances between flat_z_e and the embedding
                distances = z_sq + e_sq - 2 * ze
                encoding_indices = torch.argmin(distances, dim=1)
        # Quantize
        quantized_flat = self.embedding(encoding_indices)
        quantized = quantized_flat.view(z_e.shape)
        # loss
        z_e = z_e.to(torch.float32)
        quantized = quantized.to(torch.float32)
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        q_latent_loss = F.mse_loss(quantized, z_e.detach())
        loss = q_latent_loss + self.commitment_weight * e_latent_loss

        quantized = z_e + (quantized - z_e).detach()
        if torch.onnx.is_in_onnx_export():
            perplexity = torch.zeros(1)
        else:
            encodings_count = torch.bincount(encoding_indices, minlength=self.num_embeddings).float()
            probabilities = encodings_count / encodings_count.sum()
            perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities + 1e-10)))
        return quantized, loss, perplexity, encoding_indices


class VQEncoder(nn.Module):
    """
    VQ Encoder.
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        activation: str = "elu",
    ):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, output_dim, activation)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_cont = self.mlp(x)
        z_q, loss, perplexity, encodings = self.vq(z_cont)
        return z_q, loss, perplexity, encodings, z_cont