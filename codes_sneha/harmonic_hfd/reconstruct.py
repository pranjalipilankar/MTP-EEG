#!/usr/bin/env python3
"""
EEG Signal Reconstruction Script for STAD (FIXED)
Handles checkpoint compatibility issues
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from train_stad_complete import STAD, STADDataset, get_channel_positions, get_beta_schedule, get_diffusion_params

def load_model_flexible(model, checkpoint_path, device):
    """
    Load checkpoint with flexible key matching
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # Find mismatches
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"⚠️  Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")
    
    # Load with strict=False to ignore mismatches
    model.load_state_dict(state_dict, strict=False)
    
    return checkpoint

def reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50):
    """
    Reconstruction that exactly matches training diffusion math
    """
    model.eval()
    with torch.no_grad():
        B = x_lr.shape[0]
        lr_pos = get_channel_positions(16, device, B)

        # Start from pure noise (B, 100, 256)
        zt = torch.randn(B, 100, 256, device=device)

        timesteps = torch.linspace(999, 0, steps, dtype=torch.long, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)

            # Conditioning
            cond_tokens, cond_pooled = model.stc(x_lr, lr_pos, t_batch)

            # Predict noise εθ
            pred_noise = model.mtd(zt, t_batch, cond_tokens, cond_pooled)

            alpha_t = diff_params['sqrt_alphas_cumprod'][t] ** 2

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = diff_params['sqrt_alphas_cumprod'][t_prev] ** 2
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # Predict x0
            pred_x0 = (zt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

            # SAME update as training
            if i < len(timesteps) - 1:
                zt = torch.sqrt(alpha_t_prev) * pred_x0 + \
                     torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                zt = pred_x0

        # Decode
        cls_token = model.mae.cls_token.expand(B, -1, -1)
        zt_with_cls = torch.cat([cls_token, zt], dim=1)
        pred_patches = model.mae.decode_full(zt_with_cls)
        sr_eeg = model.mae.unpatchify(pred_patches)

        return sr_eeg

def compute_metrics(sr_eeg, hr_eeg):
    """
    Compute comprehensive reconstruction metrics
    
    Returns:
        dict with PCC, RMSE, SNR, NMSE, MAE per channel and averaged
    """
    sr_np = sr_eeg.cpu().numpy()
    hr_np = hr_eeg.cpu().numpy()
    
    # Ensure same length
    min_len = min(sr_np.shape[2], hr_np.shape[2])
    sr_np = sr_np[:, :, :min_len]
    hr_np = hr_np[:, :, :min_len]
    
    metrics = {
        'PCC_per_channel': [],
        'RMSE_per_channel': [],
        'SNR_per_channel': [],
        'NMSE_per_channel': [],
        'MAE_per_channel': []
    }
    
    B, C, T = sr_np.shape
    
    for b in range(B):
        for ch in range(C):
            sr_sig = sr_np[b, ch]
            hr_sig = hr_np[b, ch]
            
            # Pearson Correlation
            if np.std(sr_sig) > 1e-6 and np.std(hr_sig) > 1e-6:
                pcc, _ = pearsonr(sr_sig, hr_sig)
                if not np.isnan(pcc):
                    metrics['PCC_per_channel'].append(pcc)
            
            # RMSE
            rmse = np.sqrt(np.mean((sr_sig - hr_sig) ** 2))
            metrics['RMSE_per_channel'].append(rmse)
            
            # SNR (dB)
            signal_power = np.mean(hr_sig ** 2)
            noise_power = np.mean((sr_sig - hr_sig) ** 2)
            snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
            metrics['SNR_per_channel'].append(snr)
            
            # NMSE
            nmse = noise_power / (signal_power + 1e-10)
            metrics['NMSE_per_channel'].append(nmse)
            
            # MAE
            mae = np.mean(np.abs(sr_sig - hr_sig))
            metrics['MAE_per_channel'].append(mae)
    
    # Average metrics
    avg_metrics = {
        'PCC': np.mean(metrics['PCC_per_channel']) if metrics['PCC_per_channel'] else 0.0,
        'RMSE': np.mean(metrics['RMSE_per_channel']),
        'SNR': np.mean(metrics['SNR_per_channel']),
        'NMSE': np.mean(metrics['NMSE_per_channel']),
        'MAE': np.mean(metrics['MAE_per_channel'])
    }
    
    return avg_metrics, metrics

def visualize_reconstruction(lr_eeg, sr_eeg, hr_eeg, sample_idx=0, channel_idx=0, save_path='reconstruction.png'):
    lr_np = lr_eeg[sample_idx, channel_idx].cpu().numpy()
    sr_np = sr_eeg[sample_idx, channel_idx].cpu().numpy()
    hr_np = hr_eeg[sample_idx, channel_idx].cpu().numpy()
    
    # Interpolate LR to match HR length for visualization
    from scipy.interpolate import interp1d
    lr_interp = interp1d(np.linspace(0, 1, len(lr_np)), lr_np, kind='cubic')
    lr_upsampled = lr_interp(np.linspace(0, 1, len(hr_np)))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    time_axis = np.arange(len(hr_np)) / 128.0  # Assuming 128 Hz
    
    # LR (upsampled for comparison)
    axes[0].plot(time_axis, lr_upsampled, 'b-', alpha=0.7, linewidth=1.5)
    axes[0].set_title(f'Low-Resolution EEG (16 channels → interpolated)', fontsize=12)
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].grid(True, alpha=0.3)
    
    # SR
    axes[1].plot(time_axis, sr_np, 'g-', alpha=0.7, linewidth=1.5)
    axes[1].set_title(f'Super-Resolved EEG (STAD Output)', fontsize=12)
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].grid(True, alpha=0.3)
    
    # HR (ground truth)
    axes[2].plot(time_axis, hr_np, 'r-', alpha=0.7, linewidth=1.5)
    axes[2].set_title(f'High-Resolution EEG (Ground Truth)', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {save_path}")
    plt.close()

def visualize_all_channels(sr_eeg, hr_eeg, sample_idx=0, save_path='all_channels.png'):
    """
    Visualize all 32 channels comparison
    """
    sr_np = sr_eeg[sample_idx].cpu().numpy()
    hr_np = hr_eeg[sample_idx].cpu().numpy()
    
    fig, axes = plt.subplots(8, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for ch in range(32):
        time_axis = np.arange(sr_np.shape[1]) / 128.0
        
        axes[ch].plot(time_axis, hr_np[ch], 'r-', alpha=0.5, linewidth=1, label='HR')
        axes[ch].plot(time_axis, sr_np[ch], 'g-', alpha=0.7, linewidth=1, label='SR')
        axes[ch].set_title(f'Ch {ch+1}', fontsize=8)
        axes[ch].tick_params(labelsize=6)
        axes[ch].grid(True, alpha=0.2)
        if ch == 0:
            axes[ch].legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved all-channel visualization to {save_path}")
    plt.close()

def visualize_spectrum_comparison(sr_eeg, hr_eeg, sample_idx=0, channel_idx=0, save_path='spectrum.png'):
    """
    Compare frequency spectra of SR and HR
    """
    from scipy.signal import welch
    
    sr_np = sr_eeg[sample_idx, channel_idx].cpu().numpy()
    hr_np = hr_eeg[sample_idx, channel_idx].cpu().numpy()
    
    fs = 128  # Sampling frequency
    
    # Compute power spectral density
    freqs_sr, psd_sr = welch(sr_np, fs=fs, nperseg=min(256, len(sr_np)))
    freqs_hr, psd_hr = welch(hr_np, fs=fs, nperseg=min(256, len(hr_np)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(freqs_hr, psd_hr, 'r-', alpha=0.7, linewidth=2, label='HR (Ground Truth)')
    ax.semilogy(freqs_sr, psd_sr, 'g-', alpha=0.7, linewidth=2, label='SR (Reconstructed)')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.set_title(f'Frequency Spectrum Comparison - Channel {channel_idx+1}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])  # Focus on EEG relevant frequencies
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved spectrum comparison to {save_path}")
    plt.close()

def main_reconstruct(
    checkpoint_path='best_stad_DIMENSION_FIXED2.pt',
    dataset_path='/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
    num_samples=10,
    ddim_steps=50,
    output_dir='reconstruction_results'
):
    """
    Main reconstruction pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # Load model
    print("📥 Loading model...")
    model = STAD(
        lr_channels=16,
        hr_channels=32,
        seq_len=400,
        latent_dim=256,
        n_harmonics=8
    ).to(device)
    
    # Load checkpoint with flexible matching
    checkpoint = load_model_flexible(model, checkpoint_path, device)
    model.eval()
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})")
    
    # Load diffusion params
    betas = get_beta_schedule(1000).to(device)
    diff_params = get_diffusion_params(betas)
    
    # Load test data
    print("📊 Loading test dataset...")
    test_dataset = STADDataset(dataset_path, split='test', window_size=400)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Reconstruct and evaluate
    all_metrics = []
    
    print(f"\n🚀 Reconstructing {num_samples} samples with {ddim_steps} DDIM steps...")
    
    with torch.no_grad():
        for batch_idx, (x_lr, y_hr) in enumerate(test_loader):
            if batch_idx * test_loader.batch_size >= num_samples:
                break
            
            x_lr = x_lr.to(device)
            y_hr = y_hr.to(device)
            
            print(f"Processing batch {batch_idx+1}... ", end='')
            
            # Reconstruct
            sr_eeg = reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=ddim_steps)
            
            # Compute metrics
            avg_metrics, detailed_metrics = compute_metrics(sr_eeg, y_hr)
            all_metrics.append(avg_metrics)
            
            print(f"PCC={avg_metrics['PCC']:.3f}, SNR={avg_metrics['SNR']:.1f}dB, RMSE={avg_metrics['RMSE']:.4f}")
            
            # Visualize first sample of first batch
            if batch_idx == 0:
                print("📊 Creating visualizations...")
                visualize_reconstruction(x_lr, sr_eeg, y_hr, sample_idx=0, channel_idx=15,
                                        save_path=os.path.join(output_dir, 'sample_reconstruction.png'))
                visualize_all_channels(sr_eeg, y_hr, sample_idx=0,
                                      save_path=os.path.join(output_dir, 'all_channels.png'))
                visualize_spectrum_comparison(sr_eeg, y_hr, sample_idx=0, channel_idx=15,
                                             save_path=os.path.join(output_dir, 'spectrum_comparison.png'))
                
                # Save raw signals for further analysis
                np.savez(os.path.join(output_dir, 'reconstructed_signals.npz'),
                        lr_eeg=x_lr.cpu().numpy(),
                        sr_eeg=sr_eeg.cpu().numpy(),
                        hr_eeg=y_hr.cpu().numpy())
                print("✅ Saved raw signals to NPZ")
    
    # Aggregate metrics
    print("\n" + "="*60)
    print("📊 FINAL RECONSTRUCTION METRICS (averaged over all samples)")
    print("="*60)
    
    final_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    print(f"PCC (Pearson):        {final_metrics['PCC']:.4f}")
    print(f"RMSE:                 {final_metrics['RMSE']:.4f} μV")
    print(f"SNR:                  {final_metrics['SNR']:.2f} dB")
    print(f"NMSE:                 {final_metrics['NMSE']:.4f}")
    print(f"MAE:                  {final_metrics['MAE']:.4f} μV")
    print("="*60)
    
    # Save metrics
    np.save(os.path.join(output_dir, 'metrics.npy'), final_metrics)
    
    # Save detailed report
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write("STAD EEG Reconstruction Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
        f.write(f"Val Loss: {checkpoint['val_loss']:.6f}\n")
        f.write(f"DDIM Steps: {ddim_steps}\n")
        f.write(f"Num Samples: {num_samples}\n\n")
        f.write("Metrics:\n")
        f.write("-"*60 + "\n")
        for k, v in final_metrics.items():
            f.write(f"{k:20s}: {v:.4f}\n")
    
    print(f"\n✅ All results saved to {output_dir}/")
    print(f"   - sample_reconstruction.png")
    print(f"   - all_channels.png")
    print(f"   - spectrum_comparison.png")
    print(f"   - reconstructed_signals.npz")
    print(f"   - metrics.npy")
    print(f"   - report.txt")

if __name__ == '__main__':
    main_reconstruct(
        checkpoint_path='best_stad_DIMENSION_FIXED2.pt',
        dataset_path='/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
        num_samples=50,  # Number of samples to reconstruct
        ddim_steps=50,   # More steps = better quality (50-100 recommended)
        output_dir='reconstruction_results'
    )