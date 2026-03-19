#!/usr/bin/env python3
"""
Easy launcher for different MAE-DEAP configurations
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mae_deap import train_with_monitoring
from configs_recommended import (
    Config_MAE_DEAP_Light,
    Config_MAE_DEAP_Small,
    Config_MAE_DEAP_Balanced,
    Config_MAE_DEAP_FinePatch
)
from config_deap import Config_MAE_DEAP
import torch
from torch.utils.data import DataLoader
from dataset_deap import DEAPPretrainDataset, deap_transform
from mae_for_eeg import MAEforEEG
from trainer import NativeScalerWithGradNormCount as NativeScaler


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Add weight decay to optimizer, skip biases and norms"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def main():
    parser = argparse.ArgumentParser('MAE-DEAP Launcher')
    parser.add_argument('--config', type=str, default='light',
                       choices=['original', 'light', 'small', 'balanced', 'finepatch'],
                       help='Configuration to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--mask_ratio', type=float, default=None,
                       help='Override mask ratio')
    
    args = parser.parse_args()
    
    # Select configuration
    print(f"\n{'='*80}")
    print(f"MAE-DEAP Training Launcher")
    print(f"{'='*80}\n")
    
    if args.config == 'original':
        config = Config_MAE_DEAP()
        print("Using: Original configuration (config_deap.py)")
    elif args.config == 'light':
        config = Config_MAE_DEAP_Light()
        print("Using: Light configuration (recommended starting point)")
    elif args.config == 'small':
        config = Config_MAE_DEAP_Small()
        print("Using: Small configuration (fastest, good for debugging)")
    elif args.config == 'balanced':
        config = Config_MAE_DEAP_Balanced()
        print("Using: Balanced configuration (recommended for production)")
    elif args.config == 'finepatch':
        config = Config_MAE_DEAP_FinePatch()
        print("Using: Fine-patch configuration (better temporal resolution)")
    
    # Override parameters if specified
    if args.epochs is not None:
        config.num_epoch = args.epochs
        print(f"  → Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"  → Overriding batch_size: {args.batch_size}")
    
    if args.lr is not None:
        config.lr = args.lr
        print(f"  → Overriding learning rate: {args.lr}")
    
    if args.mask_ratio is not None:
        config.mask_ratio = args.mask_ratio
        print(f"  → Overriding mask_ratio: {args.mask_ratio}")
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Min LR: {config.min_lr}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.num_epoch}")
    print(f"  Warmup Epochs: {config.warmup_epochs}")
    print(f"  Mask Ratio: {config.mask_ratio}")
    print(f"  Patch Size: {config.patch_size}")
    print(f"  Embed Dim: {config.embed_dim}")
    print(f"  Depth: {config.depth}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Output Path: {config.output_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from: {config.data_path}")
    train_dataset = DEAPPretrainDataset(
        data_path=config.data_path,
        split='train',
        time_len=config.time_len,
        num_channels=config.num_channels,
        transform=lambda x: deap_transform(x, sparse_rate=config.sparse_rate)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model
    print(f"\nCreating model...")
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        focus_range=config.focus_range,
        focus_rate=config.focus_rate,
        img_recon_weight=config.img_recon_weight,
        use_nature_img_loss=config.use_nature_img_loss
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1e6:.1f} MB (fp32)")
    
    # Setup optimizer
    param_groups = add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    # Start training
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    best_cor, best_epoch = train_with_monitoring(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        config=config
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Best Correlation: {best_cor:.4f} (epoch {best_epoch})")
    print(f"Results saved to: {config.output_path}")
    
    if best_cor > 0.6:
        print("\n✓ EXCELLENT - Ready for downstream tasks!")
    elif best_cor > 0.45:
        print("\n✓ GOOD - Can be used for STAD")
    else:
        print("\n⚠️  Consider retraining with different hyperparameters")
        print("   Try: python launcher.py --config balanced")


if __name__ == '__main__':
    main()
