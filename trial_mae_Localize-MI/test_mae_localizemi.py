#!/usr/bin/env python3
"""
Evaluation Script for Localize-MI MAE
Tests reconstruction quality on validation set for HD-EEG (128 or 256 channels)
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from current directory
from mae_for_eeg import MAEforEEG
from dataset_localizemi import LocalizeMIPretrainDataset
from config_localizemi_128ch import Config_MAE_LocalizeMI_128ch

def evaluate_mae_localizemi(checkpoint_path, config, device='cuda', num_batches=20):
    """
    Evaluate MAE reconstruction quality for Localize-MI dataset
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Config_MAE_LocalizeMI instance
        device: 'cuda' or 'cpu'
        num_batches: Number of validation batches to evaluate
    """
    print("="*80)
    print(f"🔬 EVALUATING LOCALIZE-MI HD-EEG MAE RECONSTRUCTION ({config.num_channels} channels)")
    print("="*80)
    
    # Load checkpoint first to verify architecture
    print(f"\n📥 Loading checkpoint from CPU...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"✅ Loaded checkpoint:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Training correlation: {checkpoint['correlation']:.4f}")
    print(f"   Training loss: {checkpoint['loss']:.4f}")
    
    # Load model
    print("\n🏗️  Building model...")
    model = MAEforEEG(
        time_len=config.time_len,
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
    
    # Load state dict
    model.load_state_dict(checkpoint['model'])
    
    # Move to device
    print(f"\n🚀 Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Load validation data
    print("\n📦 Loading validation dataset...")
    from torch.utils.data import Subset
    from dataset_localizemi import split_dataset
    
    # Load full dataset with correct channel count
    print(f"   Requesting {config.num_channels} channels from dataset...")
    full_dataset = LocalizeMIPretrainDataset(
        data_path=config.data_path,
        subjects='all',
        num_channels=config.num_channels,
        time_len=config.time_len,
        transform=None
    )
    
    # Verify dataset output shape
    sample = full_dataset[0]
    actual_channels = sample['eeg'].shape[0]
    print(f"   Dataset returns: ({actual_channels}, {sample['eeg'].shape[1]}) - {'✅' if actual_channels == config.num_channels else '❌ MISMATCH!'}")
    
    # Handle channel mismatch with downsampling workaround
    if actual_channels != config.num_channels:
        print(f"\n⚠️  WARNING: Dataset returns {actual_channels} channels but model expects {config.num_channels}")
        if actual_channels > config.num_channels:
            print(f"   Applying uniform downsampling: {actual_channels} → {config.num_channels} channels")
            # Calculate uniform sampling indices
            step = actual_channels / config.num_channels
            channel_indices = [int(i * step) for i in range(config.num_channels)]
            print(f"   Selected channel indices (uniform): {channel_indices[:5]}...{channel_indices[-3:]}")
        else:
            print(f"   ERROR: Cannot upsample from {actual_channels} to {config.num_channels}")
            raise ValueError(f"Channel mismatch: dataset has {actual_channels}, model expects {config.num_channels}")
    else:
        channel_indices = None
    
    # Get validation split
    _, val_idx, _ = split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.15, seed=42)
    val_dataset = Subset(full_dataset, val_idx)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,  # Smaller batch for HD-EEG
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Validation epochs: {len(val_dataset)}")
    print(f"   Epoch shape: ({config.num_channels}, {config.time_len})")
    print(f"   High density: {config.num_channels} channels @ 8000 Hz")
    
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
            
            # Apply channel downsampling if needed
            if channel_indices is not None:
                eeg = eeg[:, channel_indices, :]
            
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
    print(f"📊 RECONSTRUCTION QUALITY REPORT - LOCALIZE-MI HD-EEG MAE ({config.num_channels}ch)")
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
        print("✅ MAE QUALITY: EXCELLENT - Ideal for source localization tasks!")
        recommendation = "proceed"
    elif mean_pcc > 0.60 and mean_snr > 10:
        print("✅ MAE QUALITY: GOOD - Ready for HD-EEG applications")
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
    
    # Apply channel downsampling if needed
    if channel_indices is not None:
        eeg_batch = eeg_batch[:, channel_indices, :]
    
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
    
    # Plot 8 representative channels across the HD grid
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Sample from different regions of the HD grid
    step = config.num_channels // 8
    channel_indices = [i * step for i in range(8)]
    
    for i, ch_idx in enumerate(channel_indices):
        gt = eeg_np[ch_idx]
        pred = recon_np[ch_idx]
        
        axes[i].plot(gt, label='Ground Truth', alpha=0.8, linewidth=1.5, color='blue')
        axes[i].plot(pred, label='Reconstruction', alpha=0.8, linewidth=1.5, color='red', linestyle='--')
        
        pcc, _ = pearsonr(gt, pred)
        mse_ch = np.mean((gt - pred) ** 2)
        
        axes[i].set_title(f'HD Channel {ch_idx+1}/{config.num_channels} | PCC={pcc:.3f}, MSE={mse_ch:.4f}', fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
        axes[i].set_xlabel('Time (samples @ 8000Hz)')
        axes[i].set_ylabel('Amplitude (normalized)')
    
    plt.suptitle(f'Localize-MI HD-EEG MAE ({config.num_channels}ch) | Avg PCC={mean_pcc:.3f}, SNR={mean_snr:.1f}dB | Epoch {checkpoint["epoch"]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'mae_localizemi_evaluation.png'
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
        'recommendation': recommendation,
        'num_channels': config.num_channels,
        'sampling_rate': 8000
    }
    
    np.save('mae_localizemi_metrics.npy', metrics_dict)
    print(f"✅ Saved: mae_localizemi_metrics.npy\n")
    
    return metrics_dict

def print_final_verdict(metrics):
    """Print final assessment and next steps"""
    print("="*80)
    print(f"🎯 FINAL VERDICT - LOCALIZE-MI HD-EEG MAE ({metrics['num_channels']}ch)")
    print("="*80)
    
    if metrics['recommendation'] == 'proceed':
        print("\n✅ Your Localize-MI HD-EEG MAE is READY!")
        print("\n📋 Next steps:")
        print("  1. Use for HD-EEG source localization tasks")
        if metrics['num_channels'] == 128:
            print("  2. Good balance of density and computational efficiency")
        else:
            print("  2. Use for 128→256 channel super-resolution")
        print("  3. Excellent for intracranial stimulation analysis")
        print(f"  4. {metrics['num_channels']} channels provide high spatial resolution")
        print("  5. 8000 Hz provides ultra-high temporal resolution")
        
    elif metrics['recommendation'] == 'monitor':
        print("\n⚠️  Localize-MI MAE is acceptable but not optimal")
        print("\n  Option A: Proceed with applications (expect moderate performance)")
        print("  Option B: Train MAE 50-100 more epochs")
        print("  Option C: Increase decoder depth from 8 to 12")
        print("\n  Note: 256 channels is very challenging - current results may be sufficient")
        
    else:
        print("\n❌ Localize-MI MAE needs more training")
        print("\n  Action items:")
        print("  1. Continue training for 100-200 more epochs")
        print("  2. Target: PCC > 0.60, SNR > 10 dB")
        print("  3. Consider: 256 channels is extremely challenging")
        print("  4. May need larger model (embed_dim > 1024)")
    
    print("\n📊 HD-EEG Context:")
    print(f"  Your PCC:         {metrics['pcc_mean']:.4f}")
    print(f"  {metrics['num_channels']} channels:     {metrics['num_channels']//62}x more than SEED (62)")
    print(f"  8000 Hz:          40x faster than DEAP (128Hz)")
    density = "VERY HIGH" if metrics['num_channels'] == 256 else "HIGH"
    print(f"  Challenge level:  {density} (ultra-dense HD-EEG)")
    
    print("\n💡 Use cases:")
    print("  - EEG source localization (ground truth from intracranial)")
    print("  - Forward/inverse modeling validation")
    print("  - HD electrode selection and optimization")
    if metrics['num_channels'] == 128:
        print("  - Good starting point for HD-EEG analysis")
    else:
        print("  - Super-resolution: 128→256 channels")
    
    print("="*80)

if __name__ == '__main__':
    # Configuration - using 128 channels to match checkpoint
    config = Config_MAE_LocalizeMI_128ch()
    
    # Checkpoint path
    checkpoint_path = 'results_128ch/best_checkpoint.pth'
    
    # Run evaluation
    print(f"\n🚀 Starting Localize-MI HD-EEG MAE evaluation ({config.num_channels} channels)...\n")
    
    metrics = evaluate_mae_localizemi(
        checkpoint_path=checkpoint_path,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_batches=20
    )
    
    # Final verdict
    print_final_verdict(metrics)
    
    print("\n✅ Evaluation complete!")
