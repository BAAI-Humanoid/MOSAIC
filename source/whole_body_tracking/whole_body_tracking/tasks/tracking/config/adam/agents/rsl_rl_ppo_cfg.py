from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from whole_body_tracking.utils.rsl_rl_cfg import RslRlPpoActorCriticTransformerCfg, RslRlPpoActorCriticFSQCfg, RslRlPpoActorCriticVQCfg

@configclass
class AdamFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "adam_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        # actor_hidden_dims=[1024, 512, 256, 128],
        # critic_hidden_dims=[768, 256, 128],
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
    )
    # policy=  RslRlPpoActorCriticTransformerCfg(
    #     class_name="ActorCriticTransformer",
    #     init_noise_std=1.0,
    #     seq_len=5,
    #     d_model=256,
    #     nhead=2,
    #     num_layers=1,
    #     actor_hidden_dims=[512, 256],
    #     critic_hidden_dims=[1024, 1024, 512, 256],
    #     activation="elu",
    #     activation_transformer="gelu",
    # )
    # policy = RslRlPpoActorCriticFSQCfg(
    #     class_name="ActorCriticFSQ",
    #     init_noise_std=1.0,
    #     encoder_hidden_dims=[512, 128],
    #     actor_hidden_dims=[512, 256],
    #     critic_hidden_dims=[1024, 1024, 512, 256],
    #     activation="elu",
    #     num_actor_proprio=96,
    #     latent_dim=16,
    #     num_levels=7,
    # )
    # policy = RslRlPpoActorCriticVQCfg(
    #     class_name="ActorCriticVQ",
    #     init_noise_std=1.0,
    #     encoder_hidden_dims=[512],
    #     encoder_output_dim=256,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[1024, 1024, 512, 256],
    #     activation="elu",
    #     activation_vq="elu",
    #     num_embeddings=1024,
    #     embedding_dim=8,
    #     commitment_weight=0.25,
    #     vq_loss_coef=1.0,
    #     num_actor_proprio=96,
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class AdamFlatLowFreqPPORunnerCfg(AdamFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
