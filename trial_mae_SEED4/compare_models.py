#!/usr/bin/env python3
"""
Compare MAE models trained on RAW vs PREPROCESSED SEED-IV data
Generates comprehensive comparison metrics and visualizations
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json

from mae_for_eeg import MAEforEEG
from dataset_seed4 import SEED4PretrainDataset
from dataset_seed4_raw import SEED4RawDataset
from config_seed4 import Config_MAE_SEED4


def evaluate_model(model, test_loader, device, data_source, mask_ratio=0.75, num_batches=None):
    """Evaluate a single model"""
    model.eval()
    
    all_losses = []
    all_pcc = []
    all_mse = []
    all_snr = []
    
    sample_original = None
    sample_reconstructed = None
    
    print(f"\n🔬 Evaluating {data_source.upper()} model...")
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_loader):
            if num_batches and batch_idx >= num_batches:
                break
            
            eeg = batch_dict['eeg'].to(device)
            loss, pred, mask = model(eeg, mask_ratio=mask_ratio)
            
            eeg_np = eeg.cpu().numpy()
            pred_np = pred.cpu().numpy()
            
            # Compute metrics
            batch_size = eeg_np.shape[0]
            for i in range(batch_size):
                orig_flat = eeg_np[i].flatten()
                pred_flat = pred_np[i].flatten()
                
                pcc, _ = pearsonr(orig_flat, pred_flat)
                all_pcc.append(pcc)
                
                mse = np.mean((orig_flat - pred_flat) ** 2)
                all_mse.append(mse)
                
                signal_power = np.mean(orig_flat ** 2)
                noise_power = np.mean((orig_flat - pred_flat) ** 2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                all_snr.append(snr)
            
            all_losses.append(loss.item())
            
            # Save first batch samples
            if batch_idx == 0:
                sample_original = eeg_np[:4]
                sample_reconstructed = pred_np[:4]
    
    metrics = {
        'data_source': data_source,
        'loss': np.mean(all_losses),
        'pcc_mean': np.mean(all_pcc),
        'pcc_std': np.std(all_pcc),
        'pcc_all': np.array(all_pcc),
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'snr_mean': np.mean(all_snr),
        'snr_std': np.std(all_snr)
    }
    
    samples = {
        'original': sample_original,
        'reconstructed': sample_reconstructed
    }
    
    print(f"   ✓ PCC: {metrics['pcc_mean']:.4f} ± {metrics['pcc_std']:.4f}")
    print(f"   ✓ MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"   ✓ SNR: {metrics['snr_mean']:.2f} dB ± {metrics['snr_std']:.2f}")
    
    return metrics, samples


def plot_comparison(metrics_raw, metrics_prep, samples_raw, samples_prep, output_dir):
    """Generate comprehensive comparison plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Side-by-side reconstruction examples
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    
    channel_indices = np.linspace(0, 61, 8, dtype=int)
    
    for sample_idx in range(4):
        for ch_idx, ch in enumerate(channel_indices):
            ax = axes[sample_idx, ch_idx]
            
            # Plot RAW
            if sample_idx < 2:
                orig = samples_raw['original'][sample_idx, ch, :]
                recon = samples_raw['reconstructed'][sample_idx, ch, :]
                title_prefix = "RAW"
                color = 'blue'
            else:
                orig = samples_prep['original'][sample_idx-2, ch, :]
                recon = samples_prep['reconstructed'][sample_idx-2, ch, :]
                title_prefix = "PREP"
                color = 'green'
            
            ax.plot(orig, 'k-', alpha=0.5, linewidth=0.8, label='Original')
            ax.plot(recon, color=color, linestyle='--', alpha=0.7, linewidth=0.8, label='Recon')
            
            pcc, _ = pearsonr(orig, recon)
            
            if sample_idx == 0 or sample_idx == 2:
                ax.set_title(f'{title_prefix} Ch{ch}\nPCC={pcc:.3f}', fontsize=8)
            else:
                ax.set_title(f'PCC={pcc:.3f}', fontsize=8)
            
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)
            
            if ch_idx == 0:
                ax.set_ylabel(f'Sample {sample_idx+1}', fontsize=8)
            
            if sample_idx == 0 and ch_idx == 7:
                ax.legend(fontsize=6, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: reconstruction_comparison.png")
    
    # 2. Metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # PCC histograms
    axes[0, 0].hist(metrics_raw['pcc_all'], bins=50, alpha=0.6, color='blue', label='RAW', edgecolor='black')
    axes[0, 0].hist(metrics_prep['pcc_all'], bins=50, alpha=0.6, color='green', label='PREP', edgecolor='black')
    axes[0, 0].axvline(metrics_raw['pcc_mean'], color='blue', linestyle='--', linewidth=2)
    axes[0, 0].axvline(metrics_prep['pcc_mean'], color='green', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('PCC Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCC box plot
    axes[0, 1].boxplot([metrics_raw['pcc_all'], metrics_prep['pcc_all']],
                       labels=['RAW', 'PREPROCESSED'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.6),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_ylabel('Pearson Correlation')
    axes[0, 1].set_title('PCC Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean metrics bar plot
    metrics_names = ['PCC', 'Loss', 'SNR (dB)']
    raw_vals = [metrics_raw['pcc_mean'], metrics_raw['loss'], metrics_raw['snr_mean']]
    prep_vals = [metrics_prep['pcc_mean'], metrics_prep['loss'], metrics_prep['snr_mean']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # Normalize for visualization
    axes[0, 2].bar(x - width/2, [raw_vals[0], raw_vals[1]/10, raw_vals[2]/20], width,
                   label='RAW', alpha=0.8, color='blue')
    axes[0, 2].bar(x + width/2, [prep_vals[0], prep_vals[1]/10, prep_vals[2]/20], width,
                   label='PREP', alpha=0.8, color='green')
    axes[0, 2].set_ylabel('Normalized Value')
    axes[0, 2].set_title('Mean Metrics Comparison\n(PCC, Loss/10, SNR/20)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(metrics_names)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Per-sample PCC comparison
    axes[1, 0].plot(metrics_raw['pcc_all'], 'o', markersize=2, alpha=0.5, color='blue', label='RAW')
    axes[1, 0].plot(metrics_prep['pcc_all'], 'o', markersize=2, alpha=0.5, color='green', label='PREP')
    axes[1, 0].axhline(metrics_raw['pcc_mean'], color='blue', linestyle='--', linewidth=1.5)
    axes[1, 0].axhline(metrics_prep['pcc_mean'], color='green', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('PCC')
    axes[1, 0].set_title('Per-Sample Reconstruction Quality')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: RAW vs PREP PCC
    min_len = min(len(metrics_raw['pcc_all']), len(metrics_prep['pcc_all']))
    axes[1, 1].scatter(metrics_raw['pcc_all'][:min_len], metrics_prep['pcc_all'][:min_len],
                      alpha=0.3, s=10, color='purple')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
    axes[1, 1].set_xlabel('RAW PCC')
    axes[1, 1].set_ylabel('PREPROCESSED PCC')
    axes[1, 1].set_title('Per-Sample PCC: RAW vs PREP')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal', adjustable='box')
    
    # Summary statistics table
    axes[1, 2].axis('off')
    summary_text = f"""
    COMPARISON SUMMARY
    {'='*40}
    
    RAW Data:
      PCC:  {metrics_raw['pcc_mean']:.4f} ± {metrics_raw['pcc_std']:.4f}
      MSE:  {metrics_raw['mse_mean']:.6f}
      SNR:  {metrics_raw['snr_mean']:.2f} dB
      Loss: {metrics_raw['loss']:.6f}
    
    PREPROCESSED Data:
      PCC:  {metrics_prep['pcc_mean']:.4f} ± {metrics_prep['pcc_std']:.4f}
      MSE:  {metrics_prep['mse_mean']:.6f}
      SNR:  {metrics_prep['snr_mean']:.2f} dB
      Loss: {metrics_prep['loss']:.6f}
    
    Improvement (PREP vs RAW):
      PCC:  {(metrics_prep['pcc_mean'] - metrics_raw['pcc_mean']):.4f} ({(metrics_prep['pcc_mean'] / metrics_raw['pcc_mean'] - 1)*100:+.1f}%)
      SNR:  {(metrics_prep['snr_mean'] - metrics_raw['snr_mean']):.2f} dB
    
    Winner: {'PREPROCESSED' if metrics_prep['pcc_mean'] > metrics_raw['pcc_mean'] else 'RAW'}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: metrics_comparison.png")
    
    # 3. Save comparison JSON
    comparison_results = {
        'raw': {
            'pcc_mean': float(metrics_raw['pcc_mean']),
            'pcc_std': float(metrics_raw['pcc_std']),
            'mse_mean': float(metrics_raw['mse_mean']),
            'snr_mean': float(metrics_raw['snr_mean']),
            'loss': float(metrics_raw['loss'])
        },
        'preprocessed': {
            'pcc_mean': float(metrics_prep['pcc_mean']),
            'pcc_std': float(metrics_prep['pcc_std']),
            'mse_mean': float(metrics_prep['mse_mean']),
            'snr_mean': float(metrics_prep['snr_mean']),
            'loss': float(metrics_prep['loss'])
        },
        'improvement': {
            'pcc_absolute': float(metrics_prep['pcc_mean'] - metrics_raw['pcc_mean']),
            'pcc_relative_pct': float((metrics_prep['pcc_mean'] / metrics_raw['pcc_mean'] - 1) * 100),
            'snr_improvement_db': float(metrics_prep['snr_mean'] - metrics_raw['snr_mean']),
            'winner': 'preprocessed' if metrics_prep['pcc_mean'] > metrics_raw['pcc_mean'] else 'raw'
        }
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"   Saved: comparison_results.json")


def main():
    parser = argparse.ArgumentParser('Compare RAW vs PREPROCESSED SEED-IV MAE models')
    parser.add_argument('--raw_checkpoint', type=str, required=True, help='Path to RAW model checkpoint')
    parser.add_argument('--prep_checkpoint', type=str, required=True, help='Path to PREPROCESSED model checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default='results/comparison', help='Output directory')
    parser.add_argument('--num_batches', type=int, default=None, help='Number of test batches')
    args = parser.parse_args()
    
    config = Config_MAE_SEED4()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 SEED-IV MAE: RAW vs PREPROCESSED Comparison")
    print("="*80)
    
    # Load RAW test data
    print("\n📦 Loading RAW test dataset...")
    raw_data_path = Path(config.data_path).parent / 'eeg_raw_data'
    raw_test_dataset = SEED4RawDataset(
        data_path=raw_data_path,
        sessions=config.test_sessions,
        split='test',
        val_split=0,
        downsample_to=config.sampling_rate,
        window_length=config.time_len,
        seed=config.seed,
        verbose=True
    )
    raw_test_loader = DataLoader(raw_test_dataset, batch_size=config.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
    
    # Load PREPROCESSED test data
    print("\n📦 Loading PREPROCESSED test dataset...")
    prep_test_dataset = SEED4PretrainDataset(
        data_path=config.data_path,
        sessions=config.test_sessions,
        split='test',
        val_split=0,
        seed=config.seed,
        verbose=True
    )
    prep_test_loader = DataLoader(prep_test_dataset, batch_size=config.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)
    
    # Build and load RAW model
    print("\n🏗️  Loading RAW model...")
    raw_model = MAEforEEG(
        time_len=config.time_len, patch_size=config.patch_size,
        embed_dim=config.embed_dim, in_chans=config.num_channels,
        depth=config.depth, num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    )
    raw_checkpoint = torch.load(args.raw_checkpoint, map_location='cpu')
    raw_model.load_state_dict(raw_checkpoint['model'])
    raw_model = raw_model.to(device)
    print(f"   Loaded from epoch {raw_checkpoint['epoch']}")
    
    # Build and load PREPROCESSED model
    print("\n🏗️  Loading PREPROCESSED model...")
    prep_model = MAEforEEG(
        time_len=config.time_len, patch_size=config.patch_size,
        embed_dim=config.embed_dim, in_chans=config.num_channels,
        depth=config.depth, num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    )
    prep_checkpoint = torch.load(args.prep_checkpoint, map_location='cpu')
    prep_model.load_state_dict(prep_checkpoint['model'])
    prep_model = prep_model.to(device)
    print(f"   Loaded from epoch {prep_checkpoint['epoch']}")
    
    # Evaluate RAW model
    metrics_raw, samples_raw = evaluate_model(
        raw_model, raw_test_loader, device, 'raw',
        mask_ratio=config.mask_ratio, num_batches=args.num_batches
    )
    
    # Evaluate PREPROCESSED model
    metrics_prep, samples_prep = evaluate_model(
        prep_model, prep_test_loader, device, 'preprocessed',
        mask_ratio=config.mask_ratio, num_batches=args.num_batches
    )
    
    # Generate comparison plots
    print("\n📈 Generating comparison plots...")
    plot_comparison(metrics_raw, metrics_prep, samples_raw, samples_prep, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'RAW':<20} {'PREPROCESSED':<20} {'Difference':<15}")
    print("-"*80)
    print(f"{'PCC Mean':<25} {metrics_raw['pcc_mean']:<20.4f} {metrics_prep['pcc_mean']:<20.4f} {(metrics_prep['pcc_mean'] - metrics_raw['pcc_mean']):+.4f}")
    print(f"{'MSE Mean':<25} {metrics_raw['mse_mean']:<20.6f} {metrics_prep['mse_mean']:<20.6f} {(metrics_prep['mse_mean'] - metrics_raw['mse_mean']):+.6f}")
    print(f"{'SNR Mean (dB)':<25} {metrics_raw['snr_mean']:<20.2f} {metrics_prep['snr_mean']:<20.2f} {(metrics_prep['snr_mean'] - metrics_raw['snr_mean']):+.2f}")
    print(f"{'Loss':<25} {metrics_raw['loss']:<20.6f} {metrics_prep['loss']:<20.6f} {(metrics_prep['loss'] - metrics_raw['loss']):+.6f}")
    print("="*80)
    
    if metrics_prep['pcc_mean'] > metrics_raw['pcc_mean']:
        improvement = (metrics_prep['pcc_mean'] / metrics_raw['pcc_mean'] - 1) * 100
        print(f"\n🏆 WINNER: PREPROCESSED data ({improvement:+.2f}% better PCC)")
    else:
        improvement = (metrics_raw['pcc_mean'] / metrics_prep['pcc_mean'] - 1) * 100
        print(f"\n🏆 WINNER: RAW data ({improvement:+.2f}% better PCC)")
    
    print(f"\n✅ Comparison complete! Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
