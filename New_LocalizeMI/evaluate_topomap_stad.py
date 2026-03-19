#!/usr/bin/env python3
"""
Qualitative Evaluation: Topographic Maps for STAD Super-Resolution
Generates scalp topographies across frequency bands (δ, θ, α, β, γ)
for different channel resolutions (64, 128, 256-SR)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from mne.time_frequency import psd_array_multitaper
from pathlib import Path
from scipy.signal import butter, filtfilt

# Import model components directly
sys.path.insert(0, '/home/ab_students/EEG-MTP/New_LocalizeMI')
from mae_for_eeg import MAEforEEG
from spatio_temporal_condition import SpatioTemporalConditionModule
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule

# Frequency bands (Hz) - matching STAD paper
FREQ_BANDS = {
    'δ (0.5-4 Hz)': (0.5, 4),
    'θ (4-8 Hz)': (4, 8),
    'α (8-13 Hz)': (8, 13),
    'β (13-30 Hz)': (13, 30),
    'γ (30-45 Hz)': (30, 45)
}

# ============================================================================
# Helper functions
# ============================================================================

def get_beta_schedule(timesteps=1000):
    """Cosine beta schedule for diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_diffusion_params(betas):
    """Compute diffusion parameters"""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
    }

def get_channel_positions(n_channels, device='cpu', batch_size=1):
    """Get channel positions for Localize-MI HD-EEG dataset"""
    if n_channels == 256:
        positions = []
        grid_size = 16
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size/2) / (grid_size/2)
                y = (j - grid_size/2) / (grid_size/2)
                if x**2 + y**2 <= 1.0:
                    positions.append([x, y])
        positions = np.array(positions[:256], dtype=np.float32)
        while len(positions) < 256:
            positions = np.vstack([positions, positions[-1]])
    elif n_channels == 128:
        full_pos = get_channel_positions(256, 'cpu', 1).squeeze(0).numpy()
        indices = np.linspace(0, 255, 128, dtype=int)
        positions = full_pos[indices]
    elif n_channels == 64:
        full_pos = get_channel_positions(256, 'cpu', 1).squeeze(0).numpy()
        indices = np.linspace(0, 255, 64, dtype=int)
        positions = full_pos[indices]
    else:
        raise ValueError(f"Unsupported channel count: {n_channels}")
    return torch.tensor(positions, device=device).unsqueeze(0).expand(batch_size, -1, -1)

class STAD_LocalizeMI(nn.Module):
    """STAD model for Localize-MI"""
    def __init__(self, lr_channels=64, hr_channels=128, sr_channels=256, 
                 seq_len=2800, latent_dim=1024, n_harmonics=8):
        super().__init__()
        patch_size = 16
        num_patches = seq_len // patch_size
        
        self.mae = MAEforEEG(
            time_len=seq_len, patch_size=patch_size, embed_dim=1024,
            in_chans=hr_channels, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=1.0, norm_layer=nn.LayerNorm
        )
        
        self.stc = SpatioTemporalConditionModule(
            lr_channels, seq_len, embed_dim=latent_dim, n_harmonics=n_harmonics,
            patch_size=32, n_transformer_layers=6, n_heads=16
        )
        
        # ✅ FIX: Add missing parameter to match training
        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches, 
            latent_dim=latent_dim, 
            n_layers=8, 
            n_heads=16,
            use_multiscale_conv=True  # ← CRITICAL: Must match training!
        )
        
        # Super-resolution upsampling (MUST match training!)
        self.sr_upsample = nn.Sequential(
            nn.Linear(hr_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, sr_channels)
        )
        
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.sr_channels = sr_channels
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels  # ← Add for consistency
    
    def encode_hr(self, hr_eeg):
        latent, _, _ = self.mae.forward_encoder(hr_eeg, mask_ratio=0.0)
        return latent[:, 1:, :]
    
    # ✅ ADD: Decoder method (copy from training code)
    def decode_latent_to_sr(self, latent, lr_eeg=None):
        """
        Decode latent to 256ch SR using LEARNED upsampling.
        MUST match training decoder exactly!
        """
        B = latent.size(0)
        target_len = lr_eeg.size(-1) if lr_eeg is not None else 2800
        
        # Decode latent → 128ch HR via MAE decoder
        cls_token = self.mae.cls_token.expand(B, -1, -1)
        latent_with_cls = torch.cat([cls_token, latent], dim=1)
        hr_patches = self.mae.forward_decoder(
            latent_with_cls, 
            torch.zeros(B, self.num_patches, dtype=torch.long, device=latent.device)
        )
        
        # Unpatchify: patches → full 128ch signal
        hr_eeg = self.mae.unpatchify(hr_patches)  # (B, 128, T')
        
        # Resize to target temporal length if needed
        if hr_eeg.size(-1) != target_len:
            hr_eeg = torch.nn.functional.interpolate(
                hr_eeg.unsqueeze(1),
                size=(self.hr_channels, target_len),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # ✅ CRITICAL: Use LEARNED sr_upsample (not random interpolation!)
        hr_eeg_t = hr_eeg.transpose(1, 2)         # (B, T, 128)
        sr_eeg_t = self.sr_upsample(hr_eeg_t)     # (B, T, 256) - learned mapping
        sr_eeg = sr_eeg_t.transpose(1, 2)         # (B, 256, T)
        
        return sr_eeg
    
    def forward(self, lr_eeg, zt, t_steps, lr_chan_pos):
        cond_tokens, cond_pooled = self.stc(lr_eeg, lr_chan_pos, t_steps)
        return self.mtd(zt, t_steps, cond_tokens, cond_pooled)

def load_test_samples_multireso(data_path, n_samples=100):
    """Load test samples at available resolutions (64, 128, 256 channels)"""
    subject_dirs = sorted(Path(data_path).glob('sub-*/eeg'))
    
    all_epochs_256 = []
    for subj_dir in subject_dirs:
        if not subj_dir.exists():
            continue
        for epoch_file in sorted(subj_dir.glob('*_epochs.npy')):
            data = np.load(epoch_file)
            for epoch_idx in range(data.shape[0]):
                epoch = data[epoch_idx][:, :2800]
                if epoch.shape[1] < 2800:
                    epoch = np.pad(epoch, ((0,0), (0, 2800-epoch.shape[1])), mode='edge')
                all_epochs_256.append(epoch)
    
    all_epochs_256 = np.array(all_epochs_256[-n_samples:])
    epochs_64 = all_epochs_256[:, ::4, :]
    epochs_128 = all_epochs_256[:, ::2, :]
    
    def preprocess(epochs_list):
        processed = []
        for epochs in epochs_list:
            def apply_filters(data, fs=8000):
                nyquist = 0.5 * fs
                low = max(0.1/nyquist, 0.00001)
                high = min(100.0/nyquist, 0.99)
                b_band, a_band = butter(4, [low, high], 'band')
                return filtfilt(b_band, a_band, data, axis=-1)
            
            epochs_filtered = np.array([apply_filters(e) for e in epochs])
            
            for i in range(len(epochs_filtered)):
                for ch in range(epochs_filtered.shape[1]):
                    mean = epochs_filtered[i, ch].mean()
                    std = epochs_filtered[i, ch].std() + 1e-6
                    epochs_filtered[i, ch] = (epochs_filtered[i, ch] - mean) / std
            
            processed.append(epochs_filtered.astype(np.float32))
        return processed
    
    # ✅ FIX: Also preprocess 256ch ground truth for fair comparison
    all_processed = preprocess([epochs_64, epochs_128, all_epochs_256])
    
    return {
        '64ch': all_processed[0], 
        '128ch': all_processed[1],
        '256ch_GT': all_processed[2]  # ✅ Ground truth 256ch
    }

def compute_psd_multitaper(eeg_data, fs=8000, fmin=0.5, fmax=45):
    """Compute PSD using Multitaper method"""
    psds, freqs = psd_array_multitaper(
        eeg_data, sfreq=fs, fmin=fmin, fmax=fmax,
        adaptive=True, normalization='full', verbose=False
    )
    return psds, freqs

def extract_band_power(psds, freqs, fmin, fmax):
    """Extract average power in frequency band"""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.mean(psds[:, mask], axis=1)

@torch.no_grad()
def generate_sr_eeg(model, lr_eeg, device, diff_params, T=1000, ddim_steps=50):
    """
    Generate SR EEG using DDIM sampling - MUST match training decoder!
    
    ✅ CRITICAL FIX: Uses model.decode_latent_to_sr() instead of manual interpolation
    """
    model.eval()
    lr_eeg = lr_eeg.to(device)
    B = lr_eeg.size(0)
    
    num_patches = model.num_patches
    latent_dim = model.latent_dim
    
    # Start from random noise
    zt = torch.randn(B, num_patches, latent_dim, device=device)
    
    timesteps = torch.linspace(T-1, 0, ddim_steps, dtype=torch.long, device=device)
    lr_pos = get_channel_positions(64, device, B)
    
    # DDIM reverse diffusion
    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        pred_epsilon = model(lr_eeg, zt, t_batch, lr_pos)
        
        alpha_t = diff_params['sqrt_alphas_cumprod'][t] ** 2
        alpha_t_prev = (
            diff_params['sqrt_alphas_cumprod'][timesteps[i+1]] ** 2 
            if i < len(timesteps)-1 
            else torch.tensor(1.0, device=device)
        )
        
        pred_z0 = (zt - torch.sqrt(1 - alpha_t) * pred_epsilon) / torch.sqrt(alpha_t)
        dir_zt = torch.sqrt(1 - alpha_t_prev) * pred_epsilon
        zt = torch.sqrt(alpha_t_prev) * pred_z0 + dir_zt
    
    z0_pred = zt  # Denoised latent
    
    # ✅ FIX: Use SAME decoder as training (model.decode_latent_to_sr)
    sr_eeg = model.decode_latent_to_sr(z0_pred, lr_eeg)  # (B, 256, 2800)
    
    return sr_eeg.cpu().numpy()

def create_multireso_topomap(eeg_dict, sr_eeg, fs=8000, save_path='topomap_multireso.png'):
    """Create multi-resolution topomap comparison with proper brain structure"""
    psds = {}
    freqs_all = {}
    
    for key, eeg_data in eeg_dict.items():
        psd, freqs = compute_psd_multitaper(eeg_data, fs)
        psds[key] = psd
        freqs_all[key] = freqs
    
    psd_sr, freqs_sr = compute_psd_multitaper(sr_eeg, fs)
    psds['256ch_SR'] = psd_sr
    freqs_all['256ch_SR'] = freqs_sr
    
    n_bands = len(FREQ_BANDS)
    n_resolutions = 3
    
    fig = plt.figure(figsize=(5 * n_bands, 4.5 * n_resolutions))
    gs = GridSpec(n_resolutions, n_bands, figure=fig, hspace=0.25, wspace=0.25)
    
    row_configs = [
        ('64ch', '64-ch LR EEG', 64),
        ('128ch', '128-ch HR EEG', 128),
        ('256ch_SR', '256-ch SR EEG (Ours)', 256)
    ]
    
    for row_idx, (key, label, n_channels) in enumerate(row_configs):
        psd = psds[key]
        freqs = freqs_all[key]
        eeg_data = sr_eeg if key == '256ch_SR' else eeg_dict[key]
        
        try:
            if n_channels == 256:
                montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
            elif n_channels == 128:
                montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
            else:
                montage = mne.channels.make_standard_montage('biosemi64')
            
            ch_names = montage.ch_names[:n_channels]
            info = mne.create_info(ch_names, fs, ch_types='eeg')
            info.set_montage(montage)
            
        except Exception as e:
            print(f"Warning: Using synthetic positions for {n_channels} channels")
            ch_names = [f'E{i+1}' for i in range(n_channels)]
            info = mne.create_info(ch_names, fs, ch_types='eeg')
            
            angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
            radius = 0.5
            pos_x = radius * np.cos(angles)
            pos_y = radius * np.sin(angles)
            pos_z = np.zeros(n_channels)
            
            pos_dict = {ch: np.array([x, y, z]) for ch, x, y, z in zip(ch_names, pos_x, pos_y, pos_z)}
            montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')
            info.set_montage(montage)
        
        for col_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
            band_power = extract_band_power(psd, freqs, fmin, fmax)
            
            if len(band_power) != n_channels:
                print(f"Warning: Band power length {len(band_power)} != {n_channels}, adjusting...")
                band_power = band_power[:n_channels]
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            try:
                im, cn = mne.viz.plot_topomap(
                    band_power, info, axes=ax, show=False,
                    cmap='RdBu_r', contours=8, outlines='head',
                    sphere='auto', sensors=True, res=128,
                    extrapolate='head', border='mean', size=4
                )
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, 
                                   format='%.1e', shrink=0.8)
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label('Power (μV²/Hz)', fontsize=8)
                
            except Exception as e:
                print(f"Error plotting topomap for {label}, {band_name}: {e}")
                ax.text(0.5, 0.5, f'{label}\n{band_name}\nVisualization\nFailed', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=9, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
            
            if row_idx == 0:
                ax.set_title(band_name, fontsize=14, fontweight='bold', pad=12)
            
            if col_idx == 0:
                ax.text(-0.15, 0.5, label, fontsize=13, fontweight='bold',
                       rotation=90, va='center', ha='center',
                       transform=ax.transAxes)
    
    fig.suptitle(
        'EEG Topographic Maps: Super-Resolution Comparison\n' + 
        'LR (64-ch) → HR (128-ch) → SR (256-ch)',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    fig.text(0.5, 0.015, 
            'Color intensity indicates power spectral density in each frequency band.\n' +
            'SR model reconstructs fine spatial detail from low-resolution (64-ch) input.',
            ha='center', fontsize=10, style='italic', color='#444')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved brain topomap: {save_path}")
    plt.close()

def visualize_random_channels(eeg_dict, sr_eeg, fs=8000, n_channels=10, 
                              duration_sec=0.5, save_path='timeseries_comparison.png'):
    """
    Visualize time-series of randomly selected channels across resolutions
    """
    # Get actual time dimension from data
    time_steps = eeg_dict['64ch'].shape[1]  # 2800
    
    # ✅ FIX: Clamp requested samples to available data
    n_samples_requested = int(duration_sec * fs)  # 0.5 * 8000 = 4000
    n_samples = min(n_samples_requested, time_steps)  # min(4000, 2800) = 2800
    
    # ✅ FIX: Safe random start index
    if n_samples >= time_steps:
        start_idx = 0
    else:
        max_start = time_steps - n_samples
        start_idx = np.random.randint(0, max(1, max_start))
    
    # ✅ CRITICAL: Create time axis AFTER determining actual n_samples
    actual_duration = n_samples / fs
    time_axis = np.arange(n_samples) / fs * 1000  # Convert to ms
    
    # Randomly select channels
    np.random.seed(42)
    selected_channels_64 = np.random.choice(64, min(n_channels, 64), replace=False)
    selected_channels_128 = np.random.choice(128, min(n_channels, 128), replace=False)
    selected_channels_256 = np.random.choice(256, n_channels, replace=False)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('EEG Time-Series Comparison: LR → HR → SR', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    # Plot 64-channel LR
    ax = axes[0]
    for idx, ch in enumerate(selected_channels_64[:n_channels]):
        signal = eeg_dict['64ch'][ch, start_idx:start_idx+n_samples]
        # ✅ DEBUG: Verify shapes match
        assert len(time_axis) == len(signal), f"Shape mismatch: time_axis={len(time_axis)}, signal={len(signal)}"
        offset = idx * 4
        ax.plot(time_axis, signal + offset, 
               color=colors[idx], linewidth=1.2, alpha=0.8,
               label=f'Ch {ch}')
    ax.set_ylabel('Amplitude (normalized)', fontsize=12, fontweight='bold')
    ax.set_title('64-Channel LR EEG (Input)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', ncol=2, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 128-channel HR
    ax = axes[1]
    for idx, ch in enumerate(selected_channels_128[:n_channels]):
        signal = eeg_dict['128ch'][ch, start_idx:start_idx+n_samples]
        offset = idx * 4
        ax.plot(time_axis, signal + offset,
               color=colors[idx], linewidth=1.2, alpha=0.8,
               label=f'Ch {ch}')
    ax.set_ylabel('Amplitude (normalized)', fontsize=12, fontweight='bold')
    ax.set_title('128-Channel HR EEG (Intermediate)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', ncol=2, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 256-channel SR
    ax = axes[2]
    for idx, ch in enumerate(selected_channels_256):
        signal = sr_eeg[ch, start_idx:start_idx+n_samples]
        offset = idx * 4
        ax.plot(time_axis, signal + offset,
               color=colors[idx], linewidth=1.2, alpha=0.8,
               label=f'Ch {ch}')
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude (normalized)', fontsize=12, fontweight='bold')
    ax.set_title('256-Channel SR EEG (Model Output)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', ncol=2, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    fig.text(0.5, 0.01,
            f'Visualization of {n_channels} randomly selected channels over {actual_duration:.3f}s segment.\n' +
            'Channels offset vertically for clarity. Higher resolution reveals finer temporal dynamics.',
            ha='center', fontsize=9, style='italic', color='#555')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved time-series visualization: {save_path}")
    plt.close()


def visualize_channel_detail_comparison(eeg_dict, sr_eeg, fs=8000, 
                                        save_path='channel_detail_comparison.png'):
    """Detailed comparison of a single channel across all resolutions"""
    # Get time dimension
    time_steps = eeg_dict['64ch'].shape[1]  # 2800
    
    # Use shorter duration for detail view
    duration_sec = 0.2  # 200ms
    n_samples_requested = int(duration_sec * fs)  # 1600
    n_samples = min(n_samples_requested, time_steps)  # min(1600, 2800) = 1600
    
    # ✅ FIX: Safe start index calculation
    if n_samples >= time_steps:
        start_idx = 0
    else:
        max_start = time_steps - n_samples
        start_idx = np.random.randint(0, max(1, max_start))
    
    # Create time axis AFTER determining n_samples
    time_axis = np.arange(n_samples) / fs * 1000
    
    # Pick same spatial location across resolutions
    ch_64 = 32
    ch_128 = 64
    ch_256 = 128
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Single-Channel Detail: Temporal Resolution Enhancement',
                 fontsize=15, fontweight='bold')
    
    # 64-ch
    ax = axes[0]
    signal = eeg_dict['64ch'][ch_64, start_idx:start_idx+n_samples]
    ax.plot(time_axis, signal, color='#E63946', linewidth=2, label='64-ch LR')
    ax.fill_between(time_axis, signal, alpha=0.3, color='#E63946')
    ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(f'64-Channel LR (Ch {ch_64})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # 128-ch
    ax = axes[1]
    signal = eeg_dict['128ch'][ch_128, start_idx:start_idx+n_samples]
    ax.plot(time_axis, signal, color='#457B9D', linewidth=2, label='128-ch HR')
    ax.fill_between(time_axis, signal, alpha=0.3, color='#457B9D')
    ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(f'128-Channel HR (Ch {ch_128})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # 256-ch SR
    ax = axes[2]
    signal = sr_eeg[ch_256, start_idx:start_idx+n_samples]
    ax.plot(time_axis, signal, color='#2A9D8F', linewidth=2, label='256-ch SR (Ours)')
    ax.fill_between(time_axis, signal, alpha=0.3, color='#2A9D8F')
    ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax.set_title(f'256-Channel SR (Ch {ch_256})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved channel detail comparison: {save_path}")
    plt.close()

def compute_metrics(pred, target):
    """
    Compute comprehensive reconstruction metrics
    
    Args:
        pred: (C, T) - Predicted EEG
        target: (C, T) - Target EEG
    
    Returns:
        dict with PCC, NMSE, SNR, MAE
    """
    # Flatten for metrics
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Pearson Correlation Coefficient
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    numerator = (pred_centered * target_centered).sum()
    pred_std = np.sqrt((pred_centered ** 2).sum() + 1e-8)
    target_std = np.sqrt((target_centered ** 2).sum() + 1e-8)
    pcc = numerator / (pred_std * target_std + 1e-8)
    
    # Normalized Mean Squared Error
    mse = np.mean((pred - target) ** 2)
    target_power = np.mean(target ** 2)
    nmse = mse / (target_power + 1e-8)
    
    # Signal-to-Noise Ratio (dB)
    signal_power = target_power
    noise_power = mse
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred - target))
    
    return {
        'pcc': float(pcc),
        'nmse': float(nmse),
        'snr': float(snr),
        'mae': float(mae)
    }

def evaluate_topomap_stad(checkpoint_path, data_path, output_dir='topomap_results', n_samples=5):
    """Main evaluation function"""
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("STAD Topographic Map Evaluation (Multi-Resolution)")
    print("="*60)
    
    print("\n📦 Loading model...")
    model = STAD_LocalizeMI().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
    if unexpected_keys:
        print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)} keys")
    
    model.eval()
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    T = 1000
    betas = get_beta_schedule(T).to(device)
    diff_params = get_diffusion_params(betas)
    diff_params['sqrt_alphas_cumprod'] = torch.clamp(diff_params['sqrt_alphas_cumprod'], min=1e-8)
    diff_params['sqrt_one_minus_alphas_cumprod'] = torch.clamp(diff_params['sqrt_one_minus_alphas_cumprod'], min=1e-8)
    
    print("\n📊 Loading test data (multiple resolutions)...")
    eeg_multireso = load_test_samples_multireso(data_path, n_samples=100)
    
    print(f"  Loaded: {', '.join([f'{k}: {v.shape}' for k, v in eeg_multireso.items()])}")
    
    print(f"\n🎨 Generating multi-resolution topographic maps for {n_samples} samples...")
    
    all_eeg = {k: [] for k in eeg_multireso.keys()}
    all_sr = []
    
    # ✅ NEW: Store metrics comparing SR with ground truth 256ch
    all_metrics_sr_vs_gt = []
    
    for i in range(n_samples):
        print(f"\n  Sample {i+1}/{n_samples}...")
        
        lr_eeg_64 = torch.from_numpy(eeg_multireso['64ch'][i]).unsqueeze(0)
        sr_eeg = generate_sr_eeg(model, lr_eeg_64, device, diff_params, T, ddim_steps=50)[0]
        
        # ✅ Get ground truth 256ch for comparison
        gt_256 = eeg_multireso['256ch_GT'][i]
        
        # ✅ Compute metrics: SR vs Ground Truth
        metrics = compute_metrics(sr_eeg, gt_256)
        all_metrics_sr_vs_gt.append(metrics)
        
        print(f"    📊 SR vs Ground Truth (256-ch):")
        print(f"       PCC:  {metrics['pcc']:.4f}")
        print(f"       NMSE: {metrics['nmse']:.6f}")
        print(f"       SNR:  {metrics['snr']:.2f} dB")
        print(f"       MAE:  {metrics['mae']:.4f}")
        
        # Prepare dict for visualization (exclude GT from plots)
        sample_dict = {k: v[i] for k, v in eeg_multireso.items() if k != '256ch_GT'}
        
        # Topomap
        save_path = os.path.join(output_dir, f'topomap_multireso_sample_{i+1}.png')
        create_multireso_topomap(sample_dict, sr_eeg, fs=8000, save_path=save_path)
        
        # Time-series visualization
        timeseries_path = os.path.join(output_dir, f'timeseries_sample_{i+1}.png')
        visualize_random_channels(sample_dict, sr_eeg, fs=8000, 
                                  n_channels=10, duration_sec=0.5,
                                  save_path=timeseries_path)
        
        # Detailed channel comparison (first sample only)
        if i == 0:
            detail_path = os.path.join(output_dir, 'channel_detail_comparison.png')
            visualize_channel_detail_comparison(sample_dict, sr_eeg, fs=8000,
                                               save_path=detail_path)
        
        for k in eeg_multireso.keys():
            all_eeg[k].append(eeg_multireso[k][i])
        all_sr.append(sr_eeg)
    
    print("\n📊 Creating averaged multi-resolution topographic map...")
    avg_eeg = {k: np.mean(v, axis=0) for k, v in all_eeg.items() if k != '256ch_GT'}
    avg_sr = np.mean(all_sr, axis=0)
    
    save_path = os.path.join(output_dir, 'topomap_multireso_averaged.png')
    create_multireso_topomap(avg_eeg, avg_sr, fs=8000, save_path=save_path)
    
    # Average time-series visualization
    avg_timeseries_path = os.path.join(output_dir, 'timeseries_averaged.png')
    visualize_random_channels(avg_eeg, avg_sr, fs=8000, 
                              n_channels=10, duration_sec=0.5,
                              save_path=avg_timeseries_path)
    
    # ✅ Compute average metrics
    avg_metrics = {
        'pcc': np.mean([m['pcc'] for m in all_metrics_sr_vs_gt]),
        'nmse': np.mean([m['nmse'] for m in all_metrics_sr_vs_gt]),
        'snr': np.mean([m['snr'] for m in all_metrics_sr_vs_gt]),
        'mae': np.mean([m['mae'] for m in all_metrics_sr_vs_gt])
    }
    
    # Save metrics to file
    metrics_summary = {
        'individual_samples': all_metrics_sr_vs_gt,
        'averaged_metrics': avg_metrics,
        'description': 'SR (256-ch) vs Ground Truth (256-ch) metrics'
    }
    np.save(os.path.join(output_dir, 'metrics_summary.npy'), metrics_summary)
    
    # ✅ Save as JSON for readability
    import json
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print(f"   Results saved to: {output_dir}")
    print(f"\n   Generated:")
    print(f"     - {n_samples} topomaps")
    print(f"     - {n_samples} time-series plots")
    print(f"     - 1 channel detail comparison")
    print(f"     - 1 averaged topomap + time-series")
    print(f"\n   📊 Average Reconstruction Metrics:")
    print(f"   ─────────────────────────────────────────────")
    print(f"   256-ch SR vs 256-ch Ground Truth:")
    print(f"     PCC:  {avg_metrics['pcc']:.4f}  (1.0 = perfect correlation)")
    print(f"     NMSE: {avg_metrics['nmse']:.6f}  (0.0 = perfect reconstruction)")
    print(f"     SNR:  {avg_metrics['snr']:.2f} dB  (higher = better)")
    print(f"     MAE:  {avg_metrics['mae']:.4f}  (lower = better)")
    print("="*60)


if __name__ == '__main__':
    os.system('bash fix_cache.sh')
    
    evaluate_topomap_stad(
        checkpoint_path='/home/ab_students/EEG-MTP/New_LocalizeMI/best_stad_localizemi.pt',
        data_path='/home/ab_students/EEG-MTP/DATA/Localize-MI/derivatives/epochs',
        output_dir='topomap_results2',
        n_samples=5
    )
