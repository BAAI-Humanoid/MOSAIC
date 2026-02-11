import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP


class NStepRiskHead(nn.Module):
    """
    predict next n steps of the done signal
    output the logits (non sigmoided), training loss is BCEWithLogitsLoss
    """
    def __init__(
        self,
        input_dim: int,
        horizons: list[int],
        hidden_dims: list[int],
        activation: str = "elu",
    ):
        super(NStepRiskHead, self).__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        output_dim = len(horizons)
        self.risk_heads = MLP(input_dim, hidden_dims, output_dim, activation)

        nn.init.constant_(self.risk_heads.net[-1].bias, -2.0)

    def forward(self, x):
        return self.risk_head(x)