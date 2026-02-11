import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import TemporalTransformer


class ActorCriticTransformer(ActorCritic):
    """

    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        seq_len,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "elu",
        activation_transformer: str = "gelu",
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        
        self.seq_len = seq_len
        assert num_actor_obs % seq_len == 0, f"num_actor_obs ({num_actor_obs}) must be divisible by seq_len ({seq_len})"

        self.actor_obs_per_step = num_actor_obs // seq_len

        super().__init__(
            num_actor_obs=d_model,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.actor_encoder = TemporalTransformer(
            obs_dim_per_step=self.actor_obs_per_step,
            seq_len=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            activation=activation_transformer,
        )

        print(f"Actor Encoder: {self.actor_encoder}")
    
    def _obs_to_seq(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.dim() == 2, f"expected [B, D], got {obs.shape}"
        B, D = obs.shape
        return obs.view(B, self.seq_len, -1)
    
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        obs = self._obs_to_seq(observations)
        actor_hidden = self.actor_encoder(obs)
        return super().act(actor_hidden, **kwargs)
    
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        obs = self._obs_to_seq(observations)
        actor_hidden = self.actor_encoder(obs)
        return super().act_inference(actor_hidden)

