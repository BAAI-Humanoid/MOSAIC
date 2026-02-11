"""
Reference Velocity Estimator Training Configuration
"""

from isaaclab.utils import configclass


@configclass
class RefVelEstimatorBaseCfg:
    """
    Base configuration for reference velocity estimator training.
    """

    # Data configuration
    motion_data_path: str = "path/to/motion/data"
    val_split: float = 0.1 
    history_length: int = 4

    # Training configuration
    epochs: int = 500
    batch_size: int = 4096
    learning_rate: float = 5e-4
    save_interval: int = 100

    # Optimizer configuration
    optimizer: str = "adam"
    weight_decay: float = 0.0 

    # Learning rate scheduler configuration
    scheduler: str = "plateau"
    warmup_epochs: int = 10  

    # Device configuration
    device: str = "cuda"

    # Output configuration
    output_dir: str = "logs/ref_vel_estimator"
    experiment_name: str = "ref_vel_estimator"


@configclass
class RefVelEstimatorMLPCfg(RefVelEstimatorBaseCfg):
    """
    MLP-based Reference Velocity Estimator Configuration.
    """

    # Model configuration
    model_type: str = "mlp"
    hidden_dims: list = [256, 128, 64]
    activation: str = "elu"

    # Architecture 
    use_skip_connections: bool = False 
    use_layer_norm: bool = False  
    dropout: float = 0.0  
    use_input_skip: bool = False

    # Output configuration
    output_dir: str = "logs/ref_vel_estimator"
    experiment_name: str = "ref_vel_estimator_mlp"


@configclass
class RefVelEstimatorTransformerCfg(RefVelEstimatorBaseCfg):
    """
    Transformer-based Reference Velocity Estimator Configuration.
    """

    # Model configuration
    model_type: str = "transformer"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    feature_dim: int = 61 
    dropout: float = 0.1

    # Output configuration
    output_dir: str = "logs/ref_vel_estimator_transformer"
    experiment_name: str = "ref_vel_estimator_transformer"


@configclass
class RefVelEstimatorTransformerLargeCfg(RefVelEstimatorTransformerCfg):
    """
    Transformer-based Reference Velocity Estimator Configuration.
    """

    # Model configuration
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4

    # Training configuration 
    batch_size: int = 2048 
    learning_rate: float = 3e-4 

    # Output configuration
    output_dir: str = "logs/ref_vel_estimator_transformer_large"
    experiment_name: str = "ref_vel_estimator_transformer_large"


@configclass
class RefVelEstimatorMLPOptimizedCfg(RefVelEstimatorMLPCfg):
    """
    Optimized MLP-based Reference Velocity Estimator Configuration.
    """

    # Model configuration
    hidden_dims: list = [512, 256, 128]

    # Training configuration
    epochs: int = 1000
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 10

    # Output configuration
    output_dir: str = "logs/ref_vel_estimator_mlp_optimized"
    experiment_name: str = "ref_vel_estimator_mlp_optimized"


@configclass
class RefVelEstimatorMLPResidualCfg(RefVelEstimatorMLPCfg):
    """
    Residual MLP-based Reference Velocity Estimator Configuration.
    """

    # Model configuration
    hidden_dims: list = [512, 256, 128]

    # Architecture improvements
    use_skip_connections: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.1
    use_input_skip: bool = False

    # Training configuration
    epochs: int = 1000
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 10

    # Output configuration
    output_dir: str = "logs/ref_vel_estimator_mlp_residual"
    experiment_name: str = "ref_vel_estimator_mlp_residual"


__all__ = [
    "RefVelEstimatorBaseCfg",
    "RefVelEstimatorMLPCfg",
    "RefVelEstimatorMLPOptimizedCfg",
    "RefVelEstimatorMLPResidualCfg",
    "RefVelEstimatorTransformerCfg",
    "RefVelEstimatorTransformerLargeCfg",
]
