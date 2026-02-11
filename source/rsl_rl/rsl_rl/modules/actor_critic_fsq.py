import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import FSQEncoder


class ActorCriticFSQ(ActorCritic):
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
        activation_fsq: str = "gelu",
        latent_dim: int = 256,
        num_levels: int = 256,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticFSQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        
        actor_input_dim = latent_dim + num_actor_proprio
        self.num_actor_proprio = num_actor_proprio

        super().__init__(
            num_actor_obs=actor_input_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.actor_encoder = FSQEncoder(
            input_dim=num_actor_obs - num_actor_proprio,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            num_levels=num_levels,
            activation=activation_fsq,
        )

        print(f"Actor Encoder: {self.actor_encoder}")
    
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        latent, codes, z_cont = self.actor_encoder(observations[:, :-self.num_actor_proprio])
        decoder_input = torch.cat([latent, observations[:, -self.num_actor_proprio:]], dim=1)
        return super().act(decoder_input, **kwargs)
    
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        latent, codes, z_cont = self.actor_encoder(observations[:, :-self.num_actor_proprio])
        decoder_input = torch.cat([latent, observations[:, -self.num_actor_proprio:]], dim=1)
        return super().act_inference(decoder_input)