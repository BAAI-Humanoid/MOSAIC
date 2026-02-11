#!/usr/bin/env python3
"""
Training for Reference Velocity Estimator
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).parent.parent / "source" / "whole_body_tracking"
utils_path = project_root / "whole_body_tracking" / "utils"
sys.path.insert(0, str(utils_path))

# Add rsl_rl to path
rsl_rl_path = Path(__file__).parent.parent / "source" / "rsl_rl"
sys.path.insert(0, str(rsl_rl_path))

# Add config path
config_path = project_root / "whole_body_tracking" / "tasks" / "tracking" / "config" / "g1" / "agents"
sys.path.insert(0, str(config_path))

from rsl_rl.modules import VelocityEstimator, VelocityEstimatorTransformer
from motion_data_loader import load_motion_data_for_training
from estimator_cfg import (
    RefVelEstimatorMLPCfg,
    RefVelEstimatorMLPOptimizedCfg,
    RefVelEstimatorMLPResidualCfg,
    RefVelEstimatorTransformerCfg,
    RefVelEstimatorTransformerLargeCfg,
)


def create_dataloaders(ref_vel_estimator_obs, ref_base_lin_vel_b,
                       batch_size=2048, val_split=0.1, device='cuda', seed=42):
    """Create training and validation DataLoaders"""

    total_samples = len(ref_vel_estimator_obs)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    # Random split with fixed seed for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = torch.randperm(total_samples, device=device, generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"\nDataset split:")
    print(f"  Train: {train_size:,} samples")
    print(f"  Val: {val_size:,} samples")

    # Create simple batch generator (memory-efficient: no data copying)
    class SimpleDataset:
        def __init__(self, obs, vel_target, indices):
            # Store references only, no copying
            self.obs = obs
            self.vel_target = vel_target
            self.indices = indices
            self.size = len(indices)

        def get_batches(self, batch_size, shuffle=True):
            n_batches = (self.size + batch_size - 1) // batch_size
            order = torch.randperm(self.size, device=self.indices.device) if shuffle else torch.arange(self.size, device=self.indices.device)

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, self.size)
                local_idx = order[start:end]
                # Map to real indices in the full dataset
                real_idx = self.indices[local_idx]

                yield {
                    'obs': self.obs[real_idx],
                    'target': self.vel_target[real_idx]
                }

    train_dataset = SimpleDataset(ref_vel_estimator_obs, ref_base_lin_vel_b, train_indices)
    val_dataset = SimpleDataset(ref_vel_estimator_obs, ref_base_lin_vel_b, val_indices)

    return train_dataset, val_dataset


def train_epoch(estimator, dataset, batch_size, optimizer, criterion, epoch, writer, device):
    """Train one epoch"""
    estimator.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataset.get_batches(batch_size, shuffle=True):
        # Forward pass
        obs = batch['obs'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        pred_vel = estimator(obs)

        # Compute loss
        loss = criterion(pred_vel, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    if writer is not None:
        writer.add_scalar('Train/loss', avg_loss, epoch)

    return avg_loss


def validate(estimator, dataset, batch_size, criterion, epoch, writer, device):
    """Validate the model"""
    estimator.eval()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataset.get_batches(batch_size, shuffle=False):
            obs = batch['obs'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            pred_vel = estimator(obs)

            loss = criterion(pred_vel, target)
            mae = (pred_vel - target).abs().mean()

            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches

    if writer is not None:
        writer.add_scalar('Val/loss', avg_loss, epoch)
        writer.add_scalar('Val/mae', avg_mae, epoch)

    return avg_loss, avg_mae


def main():
    # First parse to get config selection
    parser = argparse.ArgumentParser(description="Train Reference Velocity Estimator")
    parser.add_argument('--config', type=str, default=None,
                        choices=['mlp', 'mlp_optimized', 'mlp_residual', 'transformer', 'transformer_large'],
                        help='Configuration preset to use')

    # Parse known args to get config first
    args_config, remaining = parser.parse_known_args()

    # Load configuration if specified
    if args_config.config is not None:
        config_map = {
            'mlp': RefVelEstimatorMLPCfg(),
            'mlp_optimized': RefVelEstimatorMLPOptimizedCfg(),
            'mlp_residual': RefVelEstimatorMLPResidualCfg(),
            'transformer': RefVelEstimatorTransformerCfg(),
            'transformer_large': RefVelEstimatorTransformerLargeCfg(),
        }
        cfg = config_map[args_config.config]
        print(f"Using configuration: {args_config.config}")
    else:
        # Use default values (backward compatibility)
        cfg = None
        print("Using legacy command-line arguments (no config file)")

    # Now parse all arguments with defaults from config
    parser.add_argument('--motion', type=str,
                        default=cfg.motion_data_path if cfg else '/path/to/motions',
                        help='data directory')
    parser.add_argument('--output_dir', type=str,
                        default=cfg.output_dir if cfg else 'logs/ref_vel_estimator',
                        help='Output directory')
    parser.add_argument('--epochs', type=int,
                        default=cfg.epochs if cfg else 50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=cfg.batch_size if cfg else 2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=cfg.learning_rate if cfg else 1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                        default=cfg.hidden_dims if (cfg and hasattr(cfg, 'hidden_dims')) else [256, 128, 64],
                        help='Hidden layer dimensions (MLP only)')
    parser.add_argument('--model_type', type=str,
                        default=cfg.model_type if cfg else 'mlp',
                        choices=['mlp', 'transformer'],
                        help='Model type: mlp or transformer')

    # Architecture improvements (MLP only)
    parser.add_argument('--use_skip_connections', action='store_true',
                        default=cfg.use_skip_connections if (cfg and hasattr(cfg, 'use_skip_connections')) else False,
                        help='Enable residual connections between layers')
    parser.add_argument('--use_layer_norm', action='store_true',
                        default=cfg.use_layer_norm if (cfg and hasattr(cfg, 'use_layer_norm')) else False,
                        help='Enable layer normalization')
    parser.add_argument('--dropout', type=float,
                        default=cfg.dropout if (cfg and hasattr(cfg, 'dropout')) else 0.0,
                        help='Dropout probability (0.0 = disabled)')
    parser.add_argument('--use_input_skip', action='store_true',
                        default=cfg.use_input_skip if (cfg and hasattr(cfg, 'use_input_skip')) else False,
                        help='Enable direct input-to-output skip connection')
    parser.add_argument('--d_model', type=int,
                        default=cfg.d_model if (cfg and hasattr(cfg, 'd_model')) else 128,
                        help='Transformer d_model dimension')
    parser.add_argument('--nhead', type=int,
                        default=cfg.nhead if (cfg and hasattr(cfg, 'nhead')) else 4,
                        help='Number of transformer attention heads')
    parser.add_argument('--num_layers', type=int,
                        default=cfg.num_layers if (cfg and hasattr(cfg, 'num_layers')) else 2,
                        help='Number of transformer layers')
    parser.add_argument('--device', type=str,
                        default=cfg.device if cfg else 'cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--data_device', type=str,
                        default=cfg.data_device if (cfg and hasattr(cfg, 'data_device')) else None,
                        help='Device for building training data (cuda or cpu). Defaults to --device.')
    parser.add_argument('--val_split', type=float,
                        default=cfg.val_split if cfg else 0.1,
                        help='Validation split ratio')
    parser.add_argument('--save_interval', type=int,
                        default=cfg.save_interval if cfg else 10,
                        help='Checkpoint save interval (epochs)')

    # Optimizer options
    parser.add_argument('--optimizer', type=str,
                        default=cfg.optimizer if (cfg and hasattr(cfg, 'optimizer')) else 'adam',
                        choices=['adam', 'adamw'],
                        help='Optimizer type (adam or adamw)')
    parser.add_argument('--weight_decay', type=float,
                        default=cfg.weight_decay if (cfg and hasattr(cfg, 'weight_decay')) else 0.0,
                        help='Weight decay for L2 regularization (0 to disable)')

    # Learning rate scheduler options
    parser.add_argument('--scheduler', type=str,
                        default=cfg.scheduler if (cfg and hasattr(cfg, 'scheduler')) else 'plateau',
                        choices=['plateau', 'cosine', 'step'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int,
                        default=cfg.warmup_epochs if (cfg and hasattr(cfg, 'warmup_epochs')) else 10,
                        help='Warmup epochs for cosine scheduler')

    args = parser.parse_args()
    if args.data_device is None:
        args.data_device = args.device

    print("=" * 80)
    print("Training Reference Velocity Estimator")
    print("=" * 80)
    if args_config.config:
        print(f"Config: {args_config.config}")
    print(f"Motion: {args.motion}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_type}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    if args.model_type == 'mlp':
        print(f"Hidden dims: {args.hidden_dims}")
    else:
        print(f"d_model: {args.d_model}, nhead: {args.nhead}, layers: {args.num_layers}")
    print(f"Device: {args.device}")
    print(f"Data device: {args.data_device}")
    print(f"Val split: {args.val_split}")
    print(f"Optimizer: {args.optimizer.upper()}" + (f" (weight_decay={args.weight_decay})" if args.weight_decay > 0 else ""))
    print(f"Scheduler: {args.scheduler}" + (f" (warmup={args.warmup_epochs})" if args.scheduler == 'cosine' else ""))
    print("=" * 80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=output_dir / "tensorboard") 
    anchor_body_idx = 9  # G1: torso_link is at index 9 in NPZ file (robot model)

    print("=" * 80)
    print(f"Using anchor_body_idx={anchor_body_idx} for training")
    print("If you see high error, run: python scripts/get_body_indices.py")
    print("to verify the correct index")
    print("=" * 80)

    # 1. Load data (using standalone data loader with history)
    ref_vel_estimator_obs, ref_base_lin_vel_b = load_motion_data_for_training(
        args.motion, args.data_device, history_length=4, anchor_body_idx=anchor_body_idx
    )

    # 2. Create datasets
    train_dataset, val_dataset = create_dataloaders(
        ref_vel_estimator_obs, ref_base_lin_vel_b,
        batch_size=args.batch_size,
        val_split=args.val_split,
        device=args.data_device
    )

    # 3. Create model
    print("Creating model...")
    input_dim = ref_vel_estimator_obs.shape[1]  # Automatically get input dimension

    if args.model_type == 'mlp':
        estimator = VelocityEstimator(
            num_obs=input_dim,
            hidden_dims=args.hidden_dims,
            activation='elu',
            use_skip_connections=args.use_skip_connections,
            use_layer_norm=args.use_layer_norm,
            dropout=args.dropout,
            use_input_skip=args.use_input_skip,
        )
        arch_info = f"hidden_dims={args.hidden_dims}"
        if args.use_skip_connections:
            arch_info += f", skip_conn=True, layer_norm={args.use_layer_norm}, dropout={args.dropout}"
            if args.use_input_skip:
                arch_info += ", input_skip=True"
        print(f"MLP Estimator: {arch_info}")
    elif args.model_type == 'transformer':
        feature_dim = 61  # 29 + 29 + 3
        history_length = 5  # Current frame + 4 history frames
        estimator = VelocityEstimatorTransformer(
            feature_dim=feature_dim,
            history_length=history_length,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
        )
        print(f"Transformer Estimator: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Model parameters: {sum(p.numel() for p in estimator.parameters()):,}\n")

    # Move model to device
    estimator = estimator.to(args.device)
    train_device = next(estimator.parameters()).device

    # 4. Create optimizer and scheduler
    # Optimizer selection
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using AdamW optimizer (lr={args.lr}, weight_decay={args.weight_decay})")
    else:
        optimizer = optim.Adam(estimator.parameters(), lr=args.lr)
        print(f"Using Adam optimizer (lr={args.lr})")

    criterion = nn.MSELoss()

    # Scheduler selection
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
        )
        print(f"Using CosineAnnealingLR scheduler (T_max={args.epochs - args.warmup_epochs}, warmup={args.warmup_epochs})")
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.5
        )
        print(f"Using StepLR scheduler (step_size={args.epochs // 3}, gamma=0.5)")
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        print(f"Using ReduceLROnPlateau scheduler (patience=5, factor=0.5)")

    print()

    # 5. Training loop
    print("Starting training...\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", end=" ")

        start_time = time.time()

        # Train
        train_loss = train_epoch(estimator, train_dataset, args.batch_size, optimizer, criterion, epoch, writer, train_device)

        # Validate
        val_loss, val_mae = validate(estimator, val_dataset, args.batch_size, criterion, epoch, writer, train_device)

        # Update learning rate based on scheduler type
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler == 'cosine':
            # Warmup phase: linearly increase LR
            if epoch <= args.warmup_epochs:
                warmup_lr = args.lr * epoch / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                # After warmup, use cosine annealing
                scheduler.step()
        else:  # step scheduler
            scheduler.step()

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('Train/lr', current_lr, epoch)

        epoch_time = time.time() - start_time

        # Print statistics (single line)
        print(f"| Train: {train_loss:.6f} | Val: {val_loss:.6f} | MAE: {val_mae:.6f} | {epoch_time:.1f}s", end="")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = output_dir / "best_model.pt"
            estimator.save(str(best_model_path))
            print(" | ✓ Best", end="")

        # Periodically save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            estimator.save(str(checkpoint_path))
            print(" | ✓ Saved", end="")

        print()  # New line

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    estimator.save(str(final_model_path))

    # Export best model to ONNX
    print("\n" + "=" * 80)
    print("Exporting best model to ONNX...")
    print("=" * 80)

    try:
        # Load best model for export
        best_model_path = output_dir / "best_model.pt"
        if args.model_type == 'mlp':
            best_estimator = VelocityEstimator.load(str(best_model_path), device=args.device)
        elif args.model_type == 'transformer':
            best_estimator = VelocityEstimatorTransformer.load(str(best_model_path), device=args.device)

        best_estimator.eval()

        # Export to ONNX using built-in method
        onnx_path = output_dir / "best_model.onnx"
        if args.model_type == 'mlp':
            best_estimator.export_onnx(str(onnx_path), input_dim=input_dim)
        elif args.model_type == 'transformer':
            best_estimator.export_onnx(str(onnx_path))

    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        print("PyTorch model is still available.")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best epoch: {best_epoch} | Best val loss: {best_val_loss:.6f}")
    print(f"PyTorch model: {output_dir / 'best_model.pt'}")
    print(f"ONNX model: {output_dir / 'best_model.onnx'}")
    print("=" * 80)

    writer.close()


if __name__ == "__main__":
    main()
