#!/usr/bin/env python3
"""
Evaluation Script for SEED MAE
Tests reconstruction quality on validation set
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from current directory
from mae_for_eeg import MAEforEEG
from dataset_seed import SEEDPretrainDataset
from config_seed import Config_MAE_SEED

def evaluate_mae_seed(checkpoint_path, config, device='cuda', num_batches=20):
    """
    Evaluate MAE reconstruction quality
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Config_MAE_SEED instance
        device: 'cuda' or 'cpu'
        num_batches: Number of validation batches to evaluate
    """
    print("="*80)
    print("🔬 EVALUATING SEED MAE RECONSTRUCTION")
    print("="*80)
    
    # Load model
    print("\n🏗️  Building model...")
    model = MAEforEEG(
        time_len=config.segment_length,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=8,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    )
    
    # Load checkpoint
    print(f"\n📥 Loading checkpoint from CPU...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    print(f"✅ Loaded checkpoint:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Training correlation: {checkpoint['correlation']:.4f}")
    print(f"   Training loss: {checkpoint['loss']:.4f}")
    
    # Move to device
    print(f"\n🚀 Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Load validation data
    print("\n📦 Loading validation dataset...")
    val_dataset = SEEDPretrainDataset(
        data_path=config.data_path,
        split='val',
        num_channels=config.num_channels,
        segment_length=config.segment_length,
        segment_overlap=config.segment_overlap,
        transform=None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Validation segments: {len(val_dataset)}")
    print(f"   Segment shape: (62, {config.segment_length})")
    
    # Metrics storage
    all_pcc = []
    all_snr = []
    all_nmse = []
    all_mse = []
    
    print(f"\n🔄 Computing reconstruction metrics on {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            
            eeg = batch_dict['eeg'].to(device)
            
            # Full reconstruction (no masking)
            # Use mask_ratio=0 to keep all patches
            latent, _, ids_restore = model.forward_encoder(eeg, mask_ratio=0.0)
            pred_patches = model.forward_decoder(latent, ids_restore)
            recon = model.unpatchify(pred_patches)
            
            # Move to CPU for metrics
            eeg_np = eeg.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            # Align shapes (in case of mismatch)
            min_len = min(eeg_np.shape[2], recon_np.shape[2])
            eeg_np = eeg_np[:, :, :min_len]
            recon_np = recon_np[:, :, :min_len]
            
            # Channel-wise PCC
            B, C, T = eeg_np.shape
            for b in range(B):
                for ch in range(C):
                    gt = eeg_np[b, ch]
                    pred = recon_np[b, ch]
                    
                    if np.std(gt) > 1e-6 and np.std(pred) > 1e-6:
                        pcc, _ = pearsonr(gt, pred)
                        if not np.isnan(pcc):
                            all_pcc.append(pcc)
            
            # Global metrics
            mse = np.mean((recon_np - eeg_np) ** 2)
            signal_power = np.mean(eeg_np ** 2)
            nmse = mse / (signal_power + 1e-10)
            snr = 10 * np.log10((signal_power + 1e-10) / (mse + 1e-10))
            
            all_nmse.append(nmse)
            all_snr.append(snr)
            all_mse.append(mse)
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{num_batches}...")
    
    # Compute statistics
    mean_pcc = np.mean(all_pcc)
    std_pcc = np.std(all_pcc)
    mean_snr = np.mean(all_snr)
    mean_nmse = np.mean(all_nmse)
    mean_mse = np.mean(all_mse)
    
    # Print results
    print("\n" + "="*80)
    print("📊 RECONSTRUCTION QUALITY REPORT - SEED MAE")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'Value':<20} {'Target':<15} {'Status'}")
    print("-" * 80)
    print(f"{'PCC (mean±std)':<20} {mean_pcc:.4f} ± {std_pcc:.4f}    {'>0.60':<15} {'✅' if mean_pcc > 0.60 else '⚠️' if mean_pcc > 0.45 else '❌'}")
    print(f"{'SNR (dB)':<20} {mean_snr:.2f}              {'>10.0':<15} {'✅' if mean_snr > 10 else '⚠️' if mean_snr > 5 else '❌'}")
    print(f"{'NMSE':<20} {mean_nmse:.4f}            {'<0.15':<15} {'✅' if mean_nmse < 0.15 else '⚠️' if mean_nmse < 0.30 else '❌'}")
    print(f"{'MSE':<20} {mean_mse:.6f}          {'<0.05':<15} {'✅' if mean_mse < 0.05 else '⚠️' if mean_mse < 0.10 else '❌'}")
    
    print(f"\nPCC Distribution:")
    print(f"  Min:  {np.min(all_pcc):.4f}")
    print(f"  25%:  {np.percentile(all_pcc, 25):.4f}")
    print(f"  50%:  {np.percentile(all_pcc, 50):.4f}")
    print(f"  75%:  {np.percentile(all_pcc, 75):.4f}")
    print(f"  Max:  {np.max(all_pcc):.4f}")
    
    # Comparison with training metric
    print(f"\nTraining vs Validation:")
    print(f"  Training PCC:   {checkpoint['correlation']:.4f}")
    print(f"  Validation PCC: {mean_pcc:.4f}")
    print(f"  Difference:     {abs(checkpoint['correlation'] - mean_pcc):.4f}")
    
    if abs(checkpoint['correlation'] - mean_pcc) < 0.05:
        print("  ✅ Good generalization (train ≈ val)")
    elif abs(checkpoint['correlation'] - mean_pcc) < 0.10:
        print("  ⚠️  Some overfitting detected")
    else:
        print("  ❌ Significant overfitting!")
    
    # Assessment
    print("\n" + "="*80)
    if mean_pcc > 0.70 and mean_snr > 12:
        print("✅ MAE QUALITY: EXCELLENT - Ideal for STAD training!")
        recommendation = "proceed"
    elif mean_pcc > 0.60 and mean_snr > 10:
        print("✅ MAE QUALITY: GOOD - Ready for STAD training")
        recommendation = "proceed"
    elif mean_pcc > 0.45:
        print("⚠️  MAE QUALITY: ACCEPTABLE - Can proceed with monitoring")
        recommendation = "monitor"
    else:
        print("❌ MAE QUALITY: NEEDS IMPROVEMENT")
        recommendation = "retrain"
    print("="*80)
    
    # Visualization
    print("\n📈 Generating visualization...")
    
    # Get one batch for plotting
    batch_dict = next(iter(val_loader))
    eeg_batch = batch_dict['eeg'].to(device)
    
    with torch.no_grad():
        latent, _, ids_restore = model.forward_encoder(eeg_batch, mask_ratio=0.0)
        pred_patches = model.forward_decoder(latent, ids_restore)
        recon_batch = model.unpatchify(pred_patches)
    
    eeg_np = eeg_batch[0].cpu().numpy()
    recon_np = recon_batch[0].cpu().numpy()
    
    # Align lengths
    min_len = min(eeg_np.shape[1], recon_np.shape[1])
    eeg_np = eeg_np[:, :min_len]
    recon_np = recon_np[:, :min_len]
    
    # Plot 8 representative channels
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    channel_indices = [0, 7, 15, 23, 31, 39, 47, 55]  # Sample across 62 channels
    
    for i, ch_idx in enumerate(channel_indices):
        gt = eeg_np[ch_idx]
        pred = recon_np[ch_idx]
        
        axes[i].plot(gt, label='Ground Truth', alpha=0.8, linewidth=1.5, color='blue')
        axes[i].plot(pred, label='Reconstruction', alpha=0.8, linewidth=1.5, color='red', linestyle='--')
        
        pcc, _ = pearsonr(gt, pred)
        mse_ch = np.mean((gt - pred) ** 2)
        
        axes[i].set_title(f'Channel {ch_idx+1}/62 | PCC={pcc:.3f}, MSE={mse_ch:.4f}', fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude (normalized)')
    
    plt.suptitle(f'SEED MAE Reconstruction | Avg PCC={mean_pcc:.3f}, SNR={mean_snr:.1f}dB | Epoch {checkpoint["epoch"]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'mae_seed_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    # Save metrics
    metrics_dict = {
        'pcc_mean': mean_pcc,
        'pcc_std': std_pcc,
        'snr': mean_snr,
        'nmse': mean_nmse,
        'mse': mean_mse,
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_correlation': checkpoint['correlation'],
        'recommendation': recommendation
    }
    
    np.save('mae_seed_metrics.npy', metrics_dict)
    print(f"✅ Saved: mae_seed_metrics.npy\n")
    
    return metrics_dict

def print_final_verdict(metrics):
    """Print final assessment and next steps"""
    print("="*80)
    print("🎯 FINAL VERDICT - SEED MAE")
    print("="*80)
    
    if metrics['recommendation'] == 'proceed':
        print("\n✅ Your SEED MAE is READY for STAD training!")
        print("\n📋 Next steps:")
        print("  1. Use this checkpoint for STAD (31→62 channel super-resolution)")
        print("  2. Expected STAD reconstruction PCC: 0.55-0.75")
        print("  3. SEED has 62 channels vs DEAP's 32, so slightly harder task")
        print("  4. Freeze MAE for first 50 epochs, then fine-tune")
        
    elif metrics['recommendation'] == 'monitor':
        print("\n⚠️  SEED MAE is acceptable but not optimal")
        print("\n  Option A: Proceed with STAD (expect slower convergence)")
        print("  Option B: Train MAE 50-100 more epochs")
        print("  Option C: Increase decoder depth from 8 to 12")
        
    else:
        print("\n❌ SEED MAE needs more training")
        print("\n  Action items:")
        print("  1. Continue training for 100-200 more epochs")
        print("  2. Target: PCC > 0.60, SNR > 10 dB")
        print("  3. Check learning rate hasn't decayed too much")
    
    print("\n📊 Quick comparison:")
    print(f"  Your SEED MAE PCC: {metrics['pcc_mean']:.4f}")
    print(f"  Typical good MAE:  0.70-0.80")
    print(f"  Minimum for STAD:  0.45")
    
    print("="*80)

if __name__ == '__main__':
    # Configuration
    config = Config_MAE_SEED()
    
    # Checkpoint path
    checkpoint_path = 'results/best_checkpoint.pth'
    
    # Run evaluation
    print("\n🚀 Starting SEED MAE evaluation...\n")
    
    metrics = evaluate_mae_seed(
        checkpoint_path=checkpoint_path,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_batches=20
    )
    
    # Final verdict
    print_final_verdict(metrics)
    
    print("\n✅ Evaluation complete!")
