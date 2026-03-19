#!/usr/bin/env python3
"""
Diagnostic script to identify training issues
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config_deap import Config_MAE_DEAP
from dataset_deap import DEAPPretrainDataset, deap_transform
from mae_for_eeg import MAEforEEG


def check_data():
    """Check if data loads correctly"""
    print("="*80)
    print("1. DATA CHECK")
    print("="*80)
    
    config = Config_MAE_DEAP()
    
    # Check if file exists
    if not os.path.exists(config.data_path):
        print(f"❌ Data file NOT found: {config.data_path}")
        print(f"\nPlease update the path in config_deap.py")
        return False
    else:
        print(f"✓ Data file found: {config.data_path}")
    
    # Load data
    try:
        dataset = DEAPPretrainDataset(
            data_path=config.data_path,
            split='train',
            time_len=config.time_len,
            num_channels=config.num_channels
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Check a sample
        sample = dataset[0]
        eeg = sample['eeg']
        print(f"✓ Sample shape: {eeg.shape}")
        print(f"✓ Sample range: [{eeg.min():.3f}, {eeg.max():.3f}]")
        print(f"✓ Sample mean: {eeg.mean():.3f}, std: {eeg.std():.3f}")
        
        # Check for NaNs
        if torch.isnan(eeg).any():
            print(f"❌ WARNING: NaN values in data!")
            return False
        else:
            print(f"✓ No NaN values")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def check_model():
    """Check if model initializes correctly"""
    print("\n" + "="*80)
    print("2. MODEL CHECK")
    print("="*80)
    
    config = Config_MAE_DEAP()
    
    try:
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
        )
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created: {n_params:,} parameters")
        
        # Check patch configuration
        num_patches = config.time_len // config.patch_size
        print(f"✓ Patches: {num_patches} (time_len={config.time_len}, patch_size={config.patch_size})")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False, None


def check_forward_pass(model):
    """Check if forward pass works"""
    print("\n" + "="*80)
    print("3. FORWARD PASS CHECK")
    print("="*80)
    
    config = Config_MAE_DEAP()
    
    try:
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, config.num_channels, config.time_len)
        print(f"✓ Input shape: {x.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            loss, pred, mask = model(x, mask_ratio=config.mask_ratio)
        
        print(f"✓ Forward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Mask shape: {mask.shape}")
        
        # Check outputs
        if torch.isnan(loss):
            print(f"❌ WARNING: Loss is NaN!")
            return False
        
        if torch.isnan(pred).any():
            print(f"❌ WARNING: Predictions contain NaN!")
            return False
        
        print(f"✓ No NaN values in outputs")
        
        # Check reconstruction correlation
        pred_unpatched = model.unpatchify(pred)
        print(f"✓ Unpatchified shape: {pred_unpatched.shape}")
        
        # Calculate correlation
        correlations = []
        for i in range(batch_size):
            for c in range(config.num_channels):
                if torch.std(pred_unpatched[i, c]) > 1e-6 and torch.std(x[i, c]) > 1e-6:
                    corr = torch.corrcoef(torch.stack([pred_unpatched[i, c], x[i, c]]))[0, 1]
                    if not torch.isnan(corr):
                        correlations.append(corr.item())
        
        if correlations:
            avg_cor = np.mean(correlations)
            print(f"  Initial reconstruction correlation: {avg_cor:.4f}")
            print(f"  (This should be near 0 for untrained model)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_configuration():
    """Check configuration parameters"""
    print("\n" + "="*80)
    print("4. CONFIGURATION CHECK")
    print("="*80)
    
    config = Config_MAE_DEAP()
    
    issues = []
    warnings = []
    
    # Check critical parameters
    if config.mask_ratio > 0.7:
        warnings.append(f"Mask ratio very high: {config.mask_ratio} (recommend 0.4-0.6 for EEG)")
    
    if config.depth > 16:
        warnings.append(f"Model very deep: {config.depth} layers (recommend 8-12 for EEG)")
    
    if config.embed_dim > 768:
        warnings.append(f"Embedding very large: {config.embed_dim} (recommend 256-512 for EEG)")
    
    if config.min_lr == 0:
        issues.append(f"Min LR is 0 - learning rate will collapse to 0!")
    
    if config.lr < 5e-4:
        warnings.append(f"Learning rate quite low: {config.lr} (try 1e-3 to 3e-3)")
    
    # Print configuration
    print("\nCurrent Configuration:")
    print(f"  LR: {config.lr} → {config.min_lr}")
    print(f"  Mask Ratio: {config.mask_ratio}")
    print(f"  Patch Size: {config.patch_size}")
    print(f"  Depth: {config.depth}")
    print(f"  Embed Dim: {config.embed_dim}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Warmup: {config.warmup_epochs} epochs")
    
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✓ Configuration looks reasonable")
    
    return len(issues) == 0


def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC: Identifying Training Issues")
    print("="*80 + "\n")
    
    # Run checks
    data_ok = check_data()
    model_ok, model = check_model()
    
    if model_ok and model is not None:
        forward_ok = check_forward_pass(model)
    else:
        forward_ok = False
    
    config_ok = check_configuration()
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if data_ok and model_ok and forward_ok and config_ok:
        print("\n✓ All checks passed!")
        print("\nRecommendations:")
        print("1. Run with optimized config: python3 launcher.py --config light")
        print("2. Or test multiple configs: python3 quick_param_test.py")
        print("3. Check training after 30-50 epochs (correlation builds slowly)")
    else:
        print("\n❌ Issues found:")
        if not data_ok:
            print("  - Data loading problems")
        if not model_ok:
            print("  - Model initialization problems")
        if not forward_ok:
            print("  - Forward pass problems")
        if not config_ok:
            print("  - Configuration problems")
        
        print("\nPlease fix these issues before training.")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
