import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class ActorCriticAttention(ActorCritic):
    """

    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_actor_proprio,
        encoder_hidden_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "elu",
        activation_attn: str = "gelu",
        attention_dim: int = 256,
        nhead: int = 4,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticAttention.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        
        self.num_actor_proprio = num_actor_proprio
        self.num_actor_obs = num_actor_obs
        actor_input_dim = attention_dim + num_actor_proprio

        super().__init__(
            num_actor_obs=actor_input_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.reference_encoder = MLP(
            input_dim=num_actor_obs - num_actor_proprio,
            hidden_dims=encoder_hidden_dims,
            output_dim=attention_dim,
            activation=activation_attn,
        )
        self.query_encoder = MLP(
            input_dim=num_actor_proprio,
            hidden_dims=encoder_hidden_dims,
            output_dim=attention_dim,
            activation=activation_attn,
        )

        assert attention_dim % nhead == 0, "attention_dim must be divisible by nhead"

        self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=nhead, batch_first=True)

        print(f"Actor Reference Encoder: {self.reference_encoder}")
        print(f"Actor Query Encoder: {self.query_encoder}")
        print(f"Actor Attention layer: {self.attention}")
    
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        ref_input = observations[:, :-self.num_actor_proprio]
        proprio_input = observations[:, -self.num_actor_proprio:]

        key = self.reference_encoder(ref_input)
        query = self.query_encoder(proprio_input)

        Q = query.unsqueeze(1)       # [B, 1, D]
        K = key.unsqueeze(1)         # [B, 1, D]
        V = key.unsqueeze(1)         # [B, 1, D]
        context, _ = self.attention(Q, K, V)
        context = context.squeeze(1)
        actor_input = torch.cat([context, proprio_input], dim=1) 

        return super().act(actor_input, **kwargs)
    
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        ref_input = observations[:, :-self.num_actor_proprio]
        proprio_input = observations[:, -self.num_actor_proprio:]
        key = self.reference_encoder(ref_input)
        query = self.query_encoder(proprio_input)
        Q = query.unsqueeze(1);  K = key.unsqueeze(1);  V = key.unsqueeze(1)
        context, _ = self.attention(Q, K, V)
        context = context.squeeze(1)
        actor_input = torch.cat([context, proprio_input], dim=1)
        return super().act_inference(actor_input)