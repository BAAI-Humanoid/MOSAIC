import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import VQEncoder


class ActorCriticVQ(ActorCritic):
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
        encoder_output_dim: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "elu",
        activation_vq: str = "gelu",
        num_embeddings: int = 256,
        embedding_dim: int = 256,
        commitment_weight: float = 0.25,
        vq_loss_coef: float = 0.1,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticVQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        
        actor_input_dim = encoder_output_dim + num_actor_proprio
        self.num_actor_proprio = num_actor_proprio
        self.vq_loss_coef = vq_loss_coef
        self._vq_perplexity = None
        self._vq_loss = None

        encoder_input_dim = num_actor_obs - num_actor_proprio

        super().__init__(
            num_actor_obs=actor_input_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.actor_encoder = VQEncoder(
            input_dim=encoder_input_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=encoder_output_dim,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_weight=commitment_weight,
            activation=activation_vq,
        )

        print(f"Actor Encoder: {self.actor_encoder}")
    
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        latent, self._vq_loss, self._vq_perplexity, encodings, z_cont = self.actor_encoder(observations[:, :-self.num_actor_proprio])
        decoder_input = torch.cat([latent, observations[:, -self.num_actor_proprio:]], dim=1)
        return super().act(decoder_input, **kwargs)
    
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        latent, self._vq_loss, self._vq_perplexity, encodings, z_cont = self.actor_encoder(observations[:, :-self.num_actor_proprio])
        decoder_input = torch.cat([latent, observations[:, -self.num_actor_proprio:]], dim=1)
        return super().act_inference(decoder_input)
    
    @property
    def vq_loss(self) -> torch.Tensor:
        return self.vq_loss_coef * self._vq_loss

    @property
    def vq_perplexity(self) -> torch.Tensor:
        return self._vq_perplexity