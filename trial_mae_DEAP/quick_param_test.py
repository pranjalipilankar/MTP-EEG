#!/usr/bin/env python3
"""
Quick parameter testing - manually test specific configurations
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from config_deap import Config_MAE_DEAP
from dataset_deap import DEAPPretrainDataset, deap_transform
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler


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


def test_config(config_params, epochs=50, config_name="test"):
    """Test a specific configuration"""
    
    print(f"\n{'='*80}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*80}")
    
    # Create config
    config = Config_MAE_DEAP()
    
    # Update parameters
    for key, value in config_params.items():
        setattr(config, key, value)
    
    # Auto-adjust related parameters
    if 'embed_dim' in config_params:
        config.decoder_embed_dim = config.embed_dim // 2
        config.num_heads = config.embed_dim // 64
        config.decoder_num_heads = config.decoder_embed_dim // 64
    
    if 'depth' in config_params:
        config.decoder_depth = max(4, config.depth // 3)
    
    # Print configuration
    print("\nKey Parameters:")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Mask Ratio: {config.mask_ratio}")
    print(f"  Patch Size: {config.patch_size}")
    print(f"  Embed Dim: {config.embed_dim}")
    print(f"  Depth: {config.depth}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Warmup Epochs: {config.warmup_epochs}")
    print(f"  Weight Decay: {config.weight_decay}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Setup seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load data
    print("\nLoading data...")
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
    
    # Create model
    print("\nCreating model...")
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
    
    # Setup optimizer
    param_groups = add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*80}\n")
    
    best_cor = 0.0
    losses = []
    correlations = []
    
    for epoch in range(epochs):
        loss, cor = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            config=config,
            model_without_ddp=model
        )
        
        losses.append(loss)
        correlations.append(cor)
        
        lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        if cor > best_cor:
            best_cor = cor
            marker = " ✓ BEST"
        else:
            marker = ""
        
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Cor: {cor:.4f} | LR: {lr:.2e}{marker}")
        
        # Early stopping if clearly not working
        if epoch > 20 and cor < 0.05:
            print(f"\n⚠️  Correlation too low after {epoch} epochs, stopping early")
            break
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Results Summary:")
    print(f"{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"Best Correlation: {best_cor:.4f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Correlation: {correlations[-1]:.4f}")
    
    if best_cor > 0.6:
        print("✓ EXCELLENT - Great configuration!")
    elif best_cor > 0.45:
        print("✓ GOOD - Usable configuration")
    elif best_cor > 0.2:
        print("⚠️  MODERATE - May need further tuning")
    else:
        print("✗ POOR - Try different parameters")
    
    # Cleanup
    del model, optimizer, loss_scaler, train_loader, train_dataset
    torch.cuda.empty_cache()
    
    return {
        'best_cor': best_cor,
        'final_loss': losses[-1],
        'final_cor': correlations[-1],
        'all_losses': losses,
        'all_cors': correlations
    }


def main():
    """Test multiple promising configurations"""
    
    # Configuration 1: Lighter model, higher LR, less masking
    config1 = {
        'lr': 2e-3,
        'mask_ratio': 0.5,
        'patch_size': 16,
        'embed_dim': 512,
        'depth': 12,
        'batch_size': 64,
        'warmup_epochs': 10,
        'weight_decay': 0.03,
        'min_lr': 1e-5
    }
    
    # Configuration 2: Smaller patches, moderate masking
    config2 = {
        'lr': 1e-3,
        'mask_ratio': 0.6,
        'patch_size': 8,
        'embed_dim': 512,
        'depth': 12,
        'batch_size': 64,
        'warmup_epochs': 10,
        'weight_decay': 0.03,
        'min_lr': 1e-5
    }
    
    # Configuration 3: Very light model for baseline
    config3 = {
        'lr': 3e-3,
        'mask_ratio': 0.4,
        'patch_size': 32,
        'embed_dim': 256,
        'depth': 8,
        'batch_size': 64,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
        'min_lr': 1e-5
    }
    
    # Configuration 4: Balanced approach
    config4 = {
        'lr': 1.5e-3,
        'mask_ratio': 0.55,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 10,
        'batch_size': 64,
        'warmup_epochs': 8,
        'weight_decay': 0.02,
        'min_lr': 1e-5
    }
    
    configs = [
        (config1, "Lighter Model + Higher LR"),
        (config2, "Small Patches + Moderate Mask"),
        (config3, "Very Light Baseline"),
        (config4, "Balanced Approach")
    ]
    
    # Test each configuration
    results = []
    for config_params, name in configs:
        result = test_config(config_params, epochs=50, config_name=name)
        results.append((name, result))
        
        print(f"\n{'='*80}\n")
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON")
    print(f"{'='*80}\n")
    
    for name, result in results:
        print(f"{name:40s} | Best Cor: {result['best_cor']:.4f} | Final Loss: {result['final_loss']:.4f}")
    
    # Find best
    best_idx = max(range(len(results)), key=lambda i: results[i][1]['best_cor'])
    best_name, best_result = results[best_idx]
    
    print(f"\n🏆 Best Configuration: {best_name}")
    print(f"   Correlation: {best_result['best_cor']:.4f}")


if __name__ == '__main__':
    main()
