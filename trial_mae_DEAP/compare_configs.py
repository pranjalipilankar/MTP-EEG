#!/usr/bin/env python3
"""
Compare different configurations without training
Shows model size, memory usage, and parameter counts
"""

import torch
from configs_recommended import (
    Config_MAE_DEAP_Light,
    Config_MAE_DEAP_Small,
    Config_MAE_DEAP_Balanced,
    Config_MAE_DEAP_FinePatch
)
from config_deap import Config_MAE_DEAP
from mae_for_eeg import MAEforEEG


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory(model, batch_size, time_len, num_channels):
    """Estimate GPU memory usage (rough)"""
    # Model parameters
    param_memory = count_parameters(model) * 4  # 4 bytes per float32
    
    # Activations (rough estimate)
    # Input: batch_size × channels × time_len
    input_memory = batch_size * num_channels * time_len * 4
    
    # Intermediate activations (very rough: ~10x model size during training)
    activation_memory = param_memory * 10
    
    # Gradients (same as parameters)
    grad_memory = param_memory
    
    # Optimizer states (Adam: 2x parameters for momentum + variance)
    optimizer_memory = param_memory * 2
    
    total_mb = (param_memory + input_memory + activation_memory + 
                grad_memory + optimizer_memory) / 1e6
    
    return total_mb


def analyze_config(config, name):
    """Analyze a configuration"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    # Create model
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
    
    # Count parameters
    n_params = count_parameters(model)
    
    # Calculate patches
    num_patches = config.time_len // config.patch_size
    patch_dim = config.num_channels * config.patch_size
    
    # Estimate memory
    memory_mb = estimate_memory(model, config.batch_size, 
                                config.time_len, config.num_channels)
    
    # Print summary
    print(f"\nModel Architecture:")
    print(f"  Encoder Depth: {config.depth}")
    print(f"  Encoder Dim: {config.embed_dim}")
    print(f"  Encoder Heads: {config.num_heads}")
    print(f"  Decoder Depth: {config.decoder_depth}")
    print(f"  Decoder Dim: {config.decoder_embed_dim}")
    print(f"  Decoder Heads: {config.decoder_num_heads}")
    
    print(f"\nPatch Configuration:")
    print(f"  Patch Size: {config.patch_size} samples")
    print(f"  Number of Patches: {num_patches}")
    print(f"  Patch Dimension: {patch_dim} ({config.num_channels} channels × {config.patch_size})")
    print(f"  Sequence Length: {num_patches + 1} (patches + CLS token)")
    
    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Min LR: {config.min_lr}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Warmup Epochs: {config.warmup_epochs}")
    print(f"  Total Epochs: {config.num_epoch}")
    print(f"  Mask Ratio: {config.mask_ratio} ({int(num_patches * config.mask_ratio)} / {num_patches} masked)")
    print(f"  Weight Decay: {config.weight_decay}")
    
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {n_params:,}")
    print(f"  Model Size: {n_params * 4 / 1e6:.1f} MB")
    print(f"  Estimated GPU Memory: ~{memory_mb:.0f} MB ({memory_mb/1024:.1f} GB)")
    
    print(f"\nTraining Time Estimate:")
    # Rough estimate: depends on GPU, but give relative comparison
    relative_speed = (n_params / 1e6) * (num_patches / 100)
    print(f"  Relative Speed Factor: {relative_speed:.1f}x baseline")
    print(f"  (Lower is faster)")
    
    return {
        'name': name,
        'params': n_params,
        'memory_mb': memory_mb,
        'patches': num_patches,
        'depth': config.depth,
        'embed_dim': config.embed_dim,
        'lr': config.lr,
        'mask_ratio': config.mask_ratio,
        'relative_speed': relative_speed
    }


def main():
    """Compare all configurations"""
    
    configs = [
        (Config_MAE_DEAP(), "Original Configuration"),
        (Config_MAE_DEAP_Light(), "Light Configuration (Recommended)"),
        (Config_MAE_DEAP_Small(), "Small Configuration (Fastest)"),
        (Config_MAE_DEAP_Balanced(), "Balanced Configuration (Production)"),
        (Config_MAE_DEAP_FinePatch(), "Fine-Patch Configuration (High Resolution)")
    ]
    
    results = []
    
    # Analyze each config
    for config, name in configs:
        result = analyze_config(config, name)
        results.append(result)
    
    # Comparison table
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Configuration':<40} {'Params':<12} {'Memory':<12} {'Speed':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        params_str = f"{r['params']/1e6:.1f}M"
        memory_str = f"{r['memory_mb']/1024:.1f}GB"
        speed_str = f"{r['relative_speed']:.1f}x"
        
        # Truncate name if too long
        name = r['name'][:38]
        print(f"{name:<40} {params_str:<12} {memory_str:<12} {speed_str:<10}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    print("For Quick Testing:")
    print("  → Use 'Small' configuration (fastest, good for debugging)")
    print("  → Command: python launcher.py --config small --epochs 50")
    
    print("\nFor Best Performance:")
    print("  → Start with 'Light' configuration")
    print("  → Command: python launcher.py --config light")
    
    print("\nFor Production:")
    print("  → Use 'Balanced' configuration")
    print("  → Command: python launcher.py --config balanced")
    
    print("\nFor Fine Temporal Detail:")
    print("  → Use 'Fine-Patch' configuration")
    print("  → Command: python launcher.py --config finepatch")
    
    print("\nFor Hyperparameter Tuning:")
    print("  → Quick test: python quick_param_test.py")
    print("  → Full search: python hyperparam_tuning.py --strategy focused --n_trials 30")
    
    print(f"\n{'='*80}")
    print("Key Differences from Original:")
    print(f"{'='*80}\n")
    
    orig = results[0]
    light = results[1]
    
    print(f"Original → Light:")
    print(f"  Parameters: {orig['params']/1e6:.1f}M → {light['params']/1e6:.1f}M ({(1-light['params']/orig['params'])*100:.0f}% reduction)")
    print(f"  Depth: {orig['depth']} → {light['depth']} layers")
    print(f"  Embed Dim: {orig['embed_dim']} → {light['embed_dim']}")
    print(f"  Learning Rate: {orig['lr']} → {light['lr']}")
    print(f"  Mask Ratio: {orig['mask_ratio']} → {light['mask_ratio']}")
    print(f"  Speed: {light['relative_speed']/orig['relative_speed']:.1f}x faster")
    
    print("\nWhy these changes?")
    print("  ✓ EEG has simpler structure than images → fewer parameters needed")
    print("  ✓ EEG has lower SNR → less aggressive masking")
    print("  ✓ Smaller dataset → can use higher learning rate")
    print("  ✓ Lighter model → faster convergence, less overfitting")


if __name__ == '__main__':
    main()
