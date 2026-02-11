from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from whole_body_tracking.utils.rsl_rl_cfg import (
    RslRlMOSAICAlgorithmCfg,
    RslRlResidualActorCriticCfg,
    RslRlPpoActorCriticWithRefVelSkipCfg,
)

@configclass
class G1FlatMOSAICRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "g1_flat_mosaic"
    empirical_normalization = True

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        ref_vel_skip_first_layer=False, 
        ref_vel_dim=3, 
    )

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True, 

        # PPO parameters
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
        # MOSAIC specific parameters
        use_ppo=True,
        expert_trajectory_path=None, 
        lambda_off_policy=0.3,
        lambda_off_policy_decay=0.995,
        lambda_off_policy_min=0.01,
        off_policy_batch_size=256,
        expert_allow_repeat_sampling=False,
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=0.1,
    )


@configclass
class G1FlatMOSAICHybridRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC Hybrid Mode Configuration.
    """
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        ref_vel_skip_first_layer=False,
        ref_vel_dim=3,
    )

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,  # Hybrid mode

        # ========== PPO Configuration ==========
        use_ppo=True,

        # PPO hyperparameters
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

        # ========== Teacher BC Configuration ==========
        teacher_checkpoint_path=None, 
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.9995,
        lambda_teacher_min=1.0,
        teacher_loss_type="mse", 

        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False, 

        # ========== Pure BC only ==========
        gradient_accumulation_steps=15,  # For Pure BC mode

        # ========== Expert BC Configuration (Offline Data Distillation) ==========
        expert_trajectory_path=None,
        lambda_off_policy=0.1,
        lambda_off_policy_decay=0.99,
        lambda_off_policy_min=0.1,
        off_policy_batch_size=100000,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",  
        expert_normalize_obs=True,  
        expert_update_normalizer=False, 
        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=False, 
        ref_vel_estimator_checkpoint_path=None,
        ref_vel_estimator_type="transformer", 
    )


@configclass
class G1FlatMOSAICPureDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC Pure Distillation Configuration.
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=0.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        ref_vel_skip_first_layer=True, 
        ref_vel_dim=3,
    )

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,

        # ========== PPO Configuration ==========
        use_ppo=False,

        # PPO hyperparameters
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

        # ========== Teacher BC Configuration ==========
        teacher_checkpoint_path="path/to/teacher_checkpoint.pt",  
        lambda_teacher_init=1.0,
        lambda_teacher_decay=1.0,
        lambda_teacher_min=1.0,
        teacher_loss_type="mse",

        # ========== Teacher Critic Configuration ==========
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False,
        train_critic_during_distillation=True, 

        # ========== Gradient Accumulation ==========
        gradient_accumulation_steps=1,

        # ========== Expert BC Configuration ==========
        expert_trajectory_path=None,
        lambda_off_policy=1.0, 
        lambda_off_policy_decay=1.0,
        lambda_off_policy_min=1.0,
        off_policy_batch_size=100000,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",
        expert_normalize_obs=True,
        expert_update_normalizer=False,

        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=True, 
        ref_vel_estimator_checkpoint_path="path/to/ref_vel_estimator_checkpoint.pt",  # Optional checkpoint for pre-trained estimator
        ref_vel_estimator_type="mlp",  # "mlp" or "transformer"
    )


@configclass
class G1FlatMOSAICRLContinueRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC RL Continue Configuration.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid" 
    empirical_normalization = True

    # ========== Student Checkpoint Configuration ==========
    student_checkpoint_path="path/to/distilled_checkpoint.pt",

    # ========== Noise Std Reset Configuration ==========
    reset_noise_std_on_resume = True

    # ========== Normalizer Freeze Configuration ==========
    freeze_normalizer_on_resume = False

    policy = RslRlPpoActorCriticWithRefVelSkipCfg(
        class_name="ActorCritic",
        init_noise_std=0.5, 
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,  # Hybrid mode

        # ========== PPO Configuration (ENABLED) ==========
        use_ppo=True,  # Enable PPO for RL training

        # PPO hyperparameters
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

        # ========== Teacher BC Configuration ==========
        teacher_checkpoint_path=None,
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=0.1,
        teacher_loss_type="mse",

        # ========== Expert BC Configuration ==========
        expert_trajectory_path=None,
        lambda_off_policy=1.0,  
        lambda_off_policy_decay=0.995,  
        lambda_off_policy_min=1.0, 
        off_policy_batch_size=100000, 
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",
        expert_normalize_obs=True,
        expert_update_normalizer=False,

        # ========== Gradient Accumulation ==========
        gradient_accumulation_steps=1,

        # ========== Teacher Critic Configuration ==========
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False, 

        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=False,
        ref_vel_estimator_checkpoint_path=None,
        ref_vel_estimator_type="transformer",  # "mlp" or "transformer"
    )


@configclass
class G1FlatMOSAICRLContinueResidualRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC RL Continue with Residual Learning Configuration.
    """
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True 

    policy = RslRlResidualActorCriticCfg(
        # Residual network configuration
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,

        # GMT configuration
        gmt_checkpoint_path="path/to/gmt_checkpoint.pt",
        critic_hidden_dims=[1024, 1024, 512, 256],
        init_critic_from_gmt=True,

        # Standard ActorCritic parameters
        init_noise_std=0.8,
        noise_std_type="scalar",
        activation="elu",
    )

    algorithm = RslRlMOSAICAlgorithmCfg(
        # ========== Mode Selection ==========
        hybrid=True,
        use_ppo=True,

        # ========== PPO Configuration ==========
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        # ========== Teacher BC Configuration (DISABLED) ======
        teacher_checkpoint_path="path/to/teacher_checkpoint.pt",
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=0.1,
        teacher_loss_type="mse",

        # ========== Expert BC Configuration (Regularization) ==========
        expert_trajectory_path=None,
        lambda_off_policy=1.0,
        lambda_off_policy_decay=0.99,
        lambda_off_policy_min=1.0,
        off_policy_batch_size=100000,
        expert_allow_repeat_sampling=False,
        expert_loss_type="mse",
        expert_normalize_obs=True,
        expert_update_normalizer=False,

        # ========== Teacher Critic Configuration ==========
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False,

        # ========== Gradient Accumulation ==========
        gradient_accumulation_steps=1, 

        # ========== Reference Velocity Estimator Configuration ==========
        use_estimate_ref_vel=False, 
        ref_vel_estimator_checkpoint_path=None,
        ref_vel_estimator_type="transformer",  # "mlp" or "transformer"
    )


@configclass
class G1FlatMOSAICMultiTeacherResidualRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    MOSAIC Multi-Teacher Residual BC Configuration.
    """
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    empirical_normalization = True

    policy = RslRlResidualActorCriticCfg(
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,
        gmt_checkpoint_path="path/to/gmt_checkpoint.pt",
        critic_hidden_dims=[1024, 1024, 512, 256],
        init_critic_from_gmt=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        activation="elu",
        num_ref_vel_estimator_obs=305,
        ref_vel_estimator_checkpoint_path="path/to/ref_vel_estimator_checkpoint.pt",
        ref_vel_estimator_type="mlp",
    )

    algorithm = RslRlMOSAICAlgorithmCfg(
        hybrid=True,
        use_ppo=False,  # Pure BC mode
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
        # Multi-teacher BC configuration
        teacher_checkpoint_path={
            "motions": "path/to/gmt_checkpoint.pt", # same as gmt_checkpoint_path="path/to/gmt_checkpoint.pt when using one stage rl training for gmt",
            "teleop_motions": "path/to/finetuned_gmt_checkpoint.pt",
        },
        # Teacher observation source mapping: which observations each teacher uses
        teacher_obs_source_mapping={
            "motions": "teacher",  # Use teacher observations
            "teleop_motions": "policy",     # Use policy observations
        },
        lambda_teacher_init=1.0,
        lambda_teacher_decay=0.995,
        lambda_teacher_min=1.0,
        teacher_loss_type="mse",

        use_estimate_ref_vel=False,  # Enable estimator to track velocity error
        ref_vel_estimator_checkpoint_path="path/to/ref_vel_estimator_checkpoint.pt",
        ref_vel_estimator_type="mlp",

        expert_trajectory_path=None,
        lambda_off_policy=1.0,
        lambda_off_policy_decay=0.995,
        lambda_off_policy_min=0.01,
        off_policy_batch_size=100000,

        gradient_accumulation_steps=1,
        teacher_critic_checkpoint_path=None,
        teacher_critic_frozen=False,
        train_critic_during_distillation=False,
    )