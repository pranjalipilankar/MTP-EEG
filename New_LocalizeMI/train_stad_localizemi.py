#!/usr/bin/env python3
"""
STAD Training for Localize-MI Dataset (256 channels, 8000Hz HD-EEG)
Channel hierarchy (as per STAD paper):
- LR (Low Res): 64 channels → STC conditioning module
- HR (High Res): 128 channels → MAE encoder → latents
- Target: 256 channels (diffusion super-resolution)

Uses pretrained MAE from trial_mae_Localize-MI/results_128ch/best_checkpoint.pth (128 channels)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from pathlib import Path
from einops.layers.torch import Rearrange

from mae_for_eeg import MAEforEEG
from spatio_temporal_condition import SpatioTemporalConditionModule
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule


def get_beta_schedule(timesteps=1000):
    """Cosine beta schedule for diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
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
    """
    Get channel positions for Localize-MI HD-EEG dataset
    256 channels arranged in high-density grid
    """
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

    return torch.tensor(positions, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1, -1)


def compute_nmse(pred, target):
    """Normalized Mean Squared Error"""
    mse = torch.mean((pred - target) ** 2)
    target_power = torch.mean(target ** 2)
    return (mse / (target_power + 1e-8)).item()


def compute_pcc(pred, target):
    """Pearson Correlation Coefficient (averaged across channels and time)"""
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
    pcc = numerator / (pred_std * target_std + 1e-8)
    return pcc.mean().item()


def compute_snr(pred, target):
    """Signal-to-Noise Ratio in dB"""
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean((pred - target) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.item()


def pcc_loss(pred, target):
    """
    Pearson Correlation Coefficient Loss (1 - PCC to minimize).
    
    Better than MSE for EEG because:
    - Captures temporal/spatial pattern similarity
    - Invariant to linear scaling
    - Encourages physiologically plausible signals
    
    Args:
        pred: (B, C, T) - Predicted EEG
        target: (B, C, T) - Target EEG
    
    Returns:
        loss: Scalar loss value (1 - mean_pcc)
    """
    # Flatten channels and time for correlation
    pred_flat = pred.reshape(pred.size(0), -1)  # (B, C*T)
    target_flat = target.reshape(pred.size(0), -1)
    
    # Center the data
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    # Compute correlation
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
    pcc = numerator / (pred_std * target_std + 1e-8)
    
    # Return 1 - PCC as loss (minimize)
    return (1.0 - pcc).mean()


def psd_loss(pred, target, fs=8000):
    """
    Power Spectral Density Loss - preserves frequency band characteristics.
    
    Critical for EEG because:
    - Preserves oscillatory patterns (delta, theta, alpha, beta, gamma bands)
    - Ensures physiologically realistic frequency content
    - Complements time-domain (MSE) and correlation (PCC) losses
    
    Args:
        pred: (B, C, T) - Predicted EEG
        target: (B, C, T) - Target EEG
        fs: Sampling frequency (default 8000 Hz for LocalizeMI)
    
    Returns:
        loss: Mean squared error between PSDs
    """
    # Cast to float32 for FFT (cuFFT requires float32 for non-power-of-2 sizes)
    pred = pred.float()
    target = target.float()
    
    # Compute FFT along time dimension
    pred_fft = torch.fft.rfft(pred, dim=-1)  # (B, C, T//2+1) complex
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # Compute power spectral density (magnitude squared)
    pred_psd = torch.abs(pred_fft) ** 2  # (B, C, T//2+1)
    target_psd = torch.abs(target_fft) ** 2
    
    # Normalize by number of samples for consistent scaling
    pred_psd = pred_psd / pred.size(-1)
    target_psd = target_psd / target.size(-1)
    
    # MSE between PSDs (log scale for better dynamic range)
    # Log scale: small and large power values contribute more equally
    pred_psd_log = torch.log(pred_psd + 1e-8)
    target_psd_log = torch.log(target_psd + 1e-8)
    
    loss = torch.mean((pred_psd_log - target_psd_log) ** 2)
    
    return loss


@torch.no_grad()
def validate_with_metrics(model, val_loader, device, diff_params, T=1000, num_samples=100):
    """
    Comprehensive validation with NMSE, PCC, SNR metrics.
    Performs DDIM sampling to reconstruct HR latents and decodes them through MAE decoder.
    """
    model.eval()
    all_nmse = []
    all_pcc = []
    all_snr = []
    samples_processed = 0

    for x_lr, y_hr, y_sr in val_loader:
        if samples_processed >= num_samples:
            break

        x_lr = x_lr.to(device)
        y_hr = y_hr.to(device)
        B = x_lr.size(0)

        # Encode ground truth HR to latent
        z0_true = model.encode_hr(y_hr)  # (B, 175, 1024)

        # DDIM sampling with 50 steps
        ddim_steps = 50
        ddim_eta = 0.0
        timesteps = torch.linspace(T-1, 0, ddim_steps, dtype=torch.long, device=device)

        # Start from noise
        zt = torch.randn_like(z0_true)
        lr_pos = get_channel_positions(64, device, B)

        # DDIM reverse process
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            pred_epsilon = model(x_lr, zt, t_batch, lr_pos)

            # DDIM update
            alpha_t = diff_params['sqrt_alphas_cumprod'][t] ** 2
            alpha_t_prev = (
                diff_params['sqrt_alphas_cumprod'][timesteps[i+1]] ** 2
                if i < len(timesteps) - 1
                else torch.tensor(1.0, dtype=torch.float32, device=device)
            )
            sigma_t = ddim_eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            pred_z0 = (zt - torch.sqrt(1 - alpha_t) * pred_epsilon) / torch.sqrt(alpha_t)
            dir_zt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_epsilon
            noise = torch.randn_like(zt) if i < len(timesteps) - 1 else torch.zeros_like(zt)

            zt = torch.sqrt(alpha_t_prev) * pred_z0 + dir_zt + sigma_t * noise

        z0_pred = zt

        # Decode predicted latent to 256ch SR
        pred_sr = model.decode_latent_to_sr(z0_pred, x_lr)  # (B, 256, 2800)
        
        # Compare with 256ch ground truth SR
        y_sr = y_sr.to(device)

        all_nmse.append(compute_nmse(pred_sr, y_sr))
        all_pcc.append(compute_pcc(pred_sr, y_sr))
        all_snr.append(compute_snr(pred_sr, y_sr))
        samples_processed += B

    return {
        'nmse': np.mean(all_nmse),
        'pcc': np.mean(all_pcc),
        'snr': np.mean(all_snr),
    }


class LocalizeMISTADDataset(Dataset):
    """
    Localize-MI dataset for STAD training.
    Channel hierarchy (STAD paper):
    - LR (Low Res): 64 channels → STC conditioning
    - HR (High Res): 128 channels → MAE encoder → latents
    - Super-resolution target: 256 channels
    """

    def __init__(
        self,
        data_path,
        subjects='all',
        indices=None,
        lr_channels=64,
        hr_channels=128,
        sr_channels=256,
        time_len=2800,
        fs=8000,
    ):
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels
        self.time_len = time_len
        self.fs = fs

        if subjects == 'all':
            subject_dirs = sorted(Path(data_path).glob('sub-*/eeg'))
        else:
            subject_dirs = [Path(data_path) / subj / 'eeg' for subj in subjects]

        print(f"Loading Localize-MI HD-EEG data (256 channels)...")
        all_epochs = []
        for subj_dir in subject_dirs:
            if not subj_dir.exists():
                continue
            for epoch_file in sorted(subj_dir.glob('*_epochs.npy')):
                data = np.load(epoch_file)  # (n_epochs, 256, 2081)
                for epoch_idx in range(data.shape[0]):
                    epoch = data[epoch_idx]  # (256, 2081)
                    if epoch.shape[1] >= time_len:
                        epoch = epoch[:, :time_len]
                    else:
                        pad_width = ((0, 0), (0, time_len - epoch.shape[1]))
                        epoch = np.pad(epoch, pad_width, mode='edge')
                    all_epochs.append(epoch)

        all_epochs = np.array(all_epochs)  # (total_epochs, 256, 2800)
        print(f"Loaded {len(all_epochs)} epochs from 256 channels")

        if indices is not None:
            all_epochs = all_epochs[indices]
            print(f"Using {len(all_epochs)} epochs from split")

        self.sr_samples = all_epochs
        self.hr_samples = self.prepare_samples(all_epochs, hr_channels)
        self.lr_samples = self.prepare_samples(all_epochs, lr_channels)

        print(f"Prepared {len(self.hr_samples)} segments")
        print(f"  SR shape (256ch): {self.sr_samples.shape}")
        print(f"  HR shape (128ch): {self.hr_samples.shape}")
        print(f"  LR shape (64ch):  {self.lr_samples.shape}")

    def prepare_samples(self, epochs, target_channels):
        """Prepare samples with channel downsampling and preprocessing"""
        indices = self.get_egi_channel_indices(epochs.shape[1], target_channels)
        epochs_sub = epochs[:, indices, :]

        def apply_filters(data):
            nyquist = 0.5 * self.fs
            low = max(0.1 / nyquist, 0.00001)
            high = min(100.0 / nyquist, 0.99)
            b_band, a_band = butter(4, [low, high], 'band')
            data_filtered = filtfilt(b_band, a_band, data, axis=-1)
            for freq in [50, 100, 150, 200]:
                if freq < nyquist:
                    low_notch = max((freq - 1.0) / nyquist, 0.00001)
                    high_notch = min((freq + 1.0) / nyquist, 0.99999)
                    b_notch, a_notch = butter(2, [low_notch, high_notch], 'bandstop')
                    data_filtered = filtfilt(b_notch, a_notch, data_filtered, axis=-1)
            return data_filtered

        epochs_filtered = np.array([apply_filters(epoch) for epoch in epochs_sub])

        for epoch_idx in range(len(epochs_filtered)):
            for ch in range(epochs_filtered.shape[1]):
                mean = epochs_filtered[epoch_idx, ch].mean()
                std = epochs_filtered[epoch_idx, ch].std() + 1e-6
                epochs_filtered[epoch_idx, ch] = (epochs_filtered[epoch_idx, ch] - mean) / std

        return epochs_filtered.astype(np.float32)

    def get_egi_channel_indices(self, full_channels, target_channels):
        if target_channels == 128:
            indices = np.arange(0, full_channels, 2)
        elif target_channels == 64:
            indices = np.arange(0, full_channels, 4)
        elif target_channels == 32:
            indices = np.arange(0, full_channels, 8)
        elif target_channels == 16:
            indices = np.arange(0, full_channels, 16)
        else:
            indices = np.linspace(0, full_channels - 1, target_channels, dtype=int)
        return indices[:target_channels]

    def __len__(self):
        return len(self.hr_samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.lr_samples[idx], dtype=torch.float32),
            torch.tensor(self.hr_samples[idx], dtype=torch.float32),
            torch.tensor(self.sr_samples[idx], dtype=torch.float32),
        )


class STAD_LocalizeMI(nn.Module):
    """
    STAD model adapted for Localize-MI HD-EEG dataset.
    Channel hierarchy (STAD paper):
    - Input LR: 64 channels → STC conditioning
    - Input HR: 128 channels → MAE encoder → latent
    - Output SR: 256 channels (diffusion upsamples)
    """

    def __init__(
        self,
        lr_channels=64,
        hr_channels=128,
        sr_channels=256,
        seq_len=2800,
        latent_dim=1024,
        n_harmonics=8,
    ):
        super().__init__()

        patch_size = 16
        num_patches = seq_len // patch_size  # 175

        self.mae = MAEforEEG(
            time_len=seq_len,
            patch_size=patch_size,
            embed_dim=1024,
            in_chans=hr_channels,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=1.0,
            norm_layer=nn.LayerNorm,
        )

        self.stc = SpatioTemporalConditionModule(
            lr_channels,
            seq_len,
            embed_dim=latent_dim,
            n_harmonics=n_harmonics,
            patch_size=32,
            n_transformer_layers=6,
            n_heads=16,
        )

        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches,
            latent_dim=latent_dim,
            n_layers=8,
            n_heads=16,
            use_multiscale_conv=True,  # Enable multi-scale 1D convolutions (STAD Eq. 3)
        )

        # Super-resolution upsampling module: 128ch → 256ch (STAD paper: Decoder outputs SR directly)
        self.sr_upsample = nn.Sequential(
            # (B, T, 128) → (B, 128, T)
            Rearrange('b t c -> b c t'),
            
            # Conv1d: uses neighboring channels
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            
            nn.Conv1d(512, 256, kernel_size=1),  # 1x1 conv
            
            # (B, 256, T) → (B, T, 256)
            Rearrange('b c t -> b t c')
        )

        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.sr_channels = sr_channels
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels

    def encode_hr(self, hr_eeg):
        """Encode HR EEG to latent (unnormalized for diffusion)"""
        latent, _, _ = self.mae.forward_encoder(hr_eeg, mask_ratio=0.0)
        latent = latent[:, 1:, :]  # Remove CLS → (B, 175, 1024)
        return latent

    def decode_latent_to_sr(self, latent, lr_eeg=None):
        """
        Decode latent to 256ch SR (STAD paper: Decoder D outputs SR directly).
        LR conditioning only used in MTD during denoising, not in decoder.
        
        Args:
            latent: (B, 175, 1024) - Denoised latent from diffusion
            lr_eeg: (B, 64, T) - Only used to get target temporal length
        """
        B = latent.size(0)
        target_len = lr_eeg.size(-1) if lr_eeg is not None else 2800
        
        # Decode latent to 128ch HR via MAE decoder
        cls_token = self.mae.cls_token.expand(B, -1, -1)
        latent_with_cls = torch.cat([cls_token, latent], dim=1)
        hr_patches = self.mae.forward_decoder(
            latent_with_cls, 
            torch.zeros(B, self.num_patches, dtype=torch.long, device=latent.device)
        )  # (B, 175, 2048) - patches [num_patches, chan*patch_size]
        
        # Unpatchify: convert patches back to full signal
        hr_eeg = self.mae.unpatchify(hr_patches)  # (B, 128, 2800) - full EEG signal
        
        # Resize MAE output to match target length if needed
        if hr_eeg.size(-1) != target_len:
            hr_eeg = torch.nn.functional.interpolate(
                hr_eeg.unsqueeze(1), size=(self.hr_channels, target_len),
                mode='bilinear', align_corners=False
            ).squeeze(1)  # (B, 128, target_len)
        
        # Apply learned upsampling: 128ch → 256ch (per STAD Figure 2: Decoder → SR EEG)
        hr_eeg_t = hr_eeg.transpose(1, 2)  # (B, target_len, 128)
        sr_eeg_t = self.sr_upsample(hr_eeg_t)  # (B, target_len, 256)
        sr_eeg = sr_eeg_t.transpose(1, 2)  # (B, 256, target_len)
        
        return sr_eeg

    def forward(self, lr_eeg, zt, t_steps, lr_chan_pos):
        """Forward pass: predict noise"""
        cond_tokens, cond_pooled = self.stc(lr_eeg, lr_chan_pos, t_steps)
        return self.mtd(zt, t_steps, cond_tokens, cond_pooled)


def train_stad_localizemi(data_path, num_epochs=300, batch_size=8, lr=2e-4, resume_from=None,
                          loss_weights={'diff': 1.0, 'recon_mse': 0.05, 'recon_pcc': 0.4, 'recon_psd': 0.6}):
    """
    Train STAD on Localize-MI dataset.
    
    Args:
        resume_from: Path to checkpoint file to resume from.
        loss_weights: Dictionary with loss component weights (optimized for Motor Imagery):
            - 'diff': Diffusion denoising loss weight (default: 1.0)
              Primary objective - ensures high-quality latent space denoising
              
            - 'recon_mse': MSE reconstruction loss weight (default: 0.05)
              REDUCED for Motor Imagery: Amplitude less critical since MI analysis
              typically normalizes signals. Focus is on spectral/temporal patterns.
              
            - 'recon_pcc': PCC reconstruction loss weight (default: 0.4)
              MODERATE for Motor Imagery: Temporal patterns important for phase
              coherence, but secondary to frequency content in MI tasks.
              
            - 'recon_psd': PSD (frequency) reconstruction loss weight (default: 0.6)
              INCREASED for Motor Imagery: CRITICAL for MI tasks because:
              * Beta band (13-30 Hz) ERD/ERS is primary biomarker
              * Mu rhythm (8-13 Hz) motor cortex activity
              * Gamma band (30+ Hz) for cognitive processing
              * Spectral features used directly in MI classification
              
        Rationale for Motor Imagery-specific weights:
        ────────────────────────────────────────────────────────────
        LocalizeMI captures motor imagery (hand/foot movement imagination).
        Primary MI biomarkers are frequency-domain features:
        - Event-Related Desynchronization (ERD) in beta band during imagery
        - Mu rhythm suppression over motor cortex
        - Spatial patterns (C3, C4, Cz) with frequency-specific power changes
        
        Therefore: PSD >> PCC > MSE for MI reconstruction quality.
        High PSD weight ensures beta/mu band power is accurately reconstructed,
        which is essential for downstream MI classification tasks.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("STAD Training - Localize-MI HD-EEG Dataset (256 channels, 8000Hz)")
    print("Motor Imagery Task: Hand/Foot Movement Imagination")
    print("=" * 60)
    print(f"\n🎯 Loss Weights (Optimized for Motor Imagery):")
    print(f"   Diffusion:      {loss_weights['diff']:.2f} (primary denoising objective)")
    print(f"   MSE:            {loss_weights['recon_mse']:.2f} (amplitude - less critical for MI)")
    print(f"   PCC:            {loss_weights['recon_pcc']:.2f} (temporal patterns - moderate importance)")
    print(f"   PSD:            {loss_weights['recon_psd']:.2f} (frequency content - CRITICAL for MI)")
    print(f"\n💡 Rationale: MI biomarkers = Beta ERD/ERS (13-30 Hz) + Mu rhythm (8-13 Hz)")
    print(f"   → High PSD weight ensures accurate spectral reconstruction\n")

    # ── Dataset splits ──────────────────────────────────────────────────────────
    print("📦 Preparing dataset splits...")
    temp_dataset = LocalizeMISTADDataset(data_path, subjects='all')
    total_size = len(temp_dataset)

    np.random.seed(42)
    indices = np.random.permutation(total_size)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]

    train_dataset = LocalizeMISTADDataset(
        data_path, subjects='all', indices=train_idx,
        lr_channels=64, hr_channels=128, sr_channels=256, time_len=2800,
    )
    val_dataset = LocalizeMISTADDataset(
        data_path, subjects='all', indices=val_idx,
        lr_channels=64, hr_channels=128, sr_channels=256, time_len=2800,
    )

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"\n📊 Dataset (STAD channel hierarchy):")
    print(f"  LR (conditioning): 64 channels")
    print(f"  HR (MAE input):   128 channels")
    print(f"  SR (target):      256 channels")
    print(f"  Train: {len(train_dataset)} epochs")
    print(f"  Val:   {len(val_dataset)} epochs")
    print(f"  Batches/epoch: {len(train_loader)}\n")

    # ── Model ───────────────────────────────────────────────────────────────────
    model = STAD_LocalizeMI(
        lr_channels=64, hr_channels=128, sr_channels=256,
        seq_len=2800, latent_dim=1024, n_harmonics=8,
    ).to(device)

    # Load pretrained MAE
    mae_checkpoint = '/home/ab_students/EEG-MTP/trial_mae_Localize-MI/results_128ch/best_checkpoint.pth'
    if os.path.exists(mae_checkpoint):
        ckpt = torch.load(mae_checkpoint, map_location='cpu', weights_only=False)
        model.mae.load_state_dict(ckpt['model'])
        print(f"✅ Loaded pretrained MAE from {mae_checkpoint}")
        print(f"   MAE correlation: {ckpt['correlation']:.4f}")
        print(f"   MAE epoch: {ckpt['epoch']}\n")
        del ckpt
        torch.cuda.empty_cache()
    else:
        print(f"⚠️  MAE checkpoint not found: {mae_checkpoint}")
        print("   Training without pretrained weights\n")

    # Freeze MAE initially
    for p in model.mae.parameters():
        p.requires_grad = False
    print("🔒 MAE encoder frozen (will unfreeze at epoch 50)\n")

    # ── Diffusion parameters ─────────────────────────────────────────────────────
    T = 1000
    betas = get_beta_schedule(T).to(device)
    diff_params = get_diffusion_params(betas)
    diff_params['sqrt_alphas_cumprod'] = torch.clamp(
        diff_params['sqrt_alphas_cumprod'], min=1e-8
    )
    diff_params['sqrt_one_minus_alphas_cumprod'] = torch.clamp(
        diff_params['sqrt_one_minus_alphas_cumprod'], min=1e-8
    )

    # ── Optimizer / scheduler / scaler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, betas=(0.9, 0.95), weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    metrics_history = []
    start_epoch = 0

    # ── Resume from checkpoint ───────────────────────────────────────────────────
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
        if unexpected_keys:
            print(f"  ⚠️  Ignoring unexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"  ⚠️  Missing keys (random init): {missing_keys}")
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        metrics_history = ckpt.get('metrics_history', [])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_epochs - start_epoch
        )
        print(f"✅ Resumed from epoch {ckpt['epoch']}")
        print(f"   Best val loss: {best_val_loss:.6f}")
        print(f"   Continuing from epoch {start_epoch}\n")
        del ckpt
        torch.cuda.empty_cache()

    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"🔧 Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M\n")

    # ── Training loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):

        # Unfreeze MAE at epoch 50
        if epoch == 50:
            print("\n" + "=" * 60)
            print("🔓 Unfreezing MAE for fine-tuning")
            print("=" * 60)
            torch.cuda.empty_cache()

            new_batch_size = max(1, batch_size // 2)
            print(f"Reducing batch size: {batch_size} → {new_batch_size}")

            train_loader = DataLoader(
                train_dataset, batch_size=new_batch_size, shuffle=True,
                num_workers=4, pin_memory=True, persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=new_batch_size, shuffle=False,
                num_workers=4, pin_memory=True, persistent_workers=True,
            )

            for param in model.mae.parameters():
                param.requires_grad = True

            if hasattr(model.mae, 'gradient_checkpointing_enable'):
                model.mae.gradient_checkpointing_enable()

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-5, weight_decay=0.01,
            )
            torch.cuda.empty_cache()
            print(f"GPU memory after unfreezing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # ── Train one epoch ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        nan_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x_lr, y_hr, y_sr) in enumerate(pbar):
            x_lr  = x_lr.to(device)
            y_hr  = y_hr.to(device)
            y_sr  = y_sr.to(device)
            B = x_lr.size(0)

            # Skip batches with NaN inputs
            if torch.isnan(x_lr).any() or torch.isnan(y_hr).any():
                print(f"\n⚠️  NaN in inputs at batch {batch_idx} - skipping")
                nan_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.float16):
                # Encode HR → latent
                z0 = model.encode_hr(y_hr)  # (B, 175, 1024)

                if torch.isnan(z0).any():
                    print(f"\n⚠️  NaN in latent at batch {batch_idx} - skipping")
                    nan_batches += 1
                    continue

                z0 = torch.clamp(z0, min=-10.0, max=10.0)

                # Forward diffusion
                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)
                sqrt_alpha     = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon

                if torch.isnan(zt).any():
                    print(f"\n⚠️  NaN in noised latent at batch {batch_idx} - skipping")
                    nan_batches += 1
                    continue

                lr_pos = get_channel_positions(64, device, B)
                pred_epsilon = model(x_lr, zt, t, lr_pos)

                if torch.isnan(pred_epsilon).any():
                    print(f"\n⚠️  NaN in prediction at batch {batch_idx} - skipping")
                    nan_batches += 1
                    continue

                # Diffusion denoising loss
                diff_loss = criterion(pred_epsilon, epsilon)

                # Reconstruction loss: decode to 256ch SR and compare with target
                sr_recon = model.decode_latent_to_sr(z0, x_lr)
                recon_mse = criterion(sr_recon, y_sr)
                recon_pcc = pcc_loss(sr_recon, y_sr)
                recon_psd = psd_loss(sr_recon, y_sr, fs=8000)

                # Combined loss (weighted)
                loss = (loss_weights['diff'] * diff_loss + 
                       loss_weights['recon_mse'] * recon_mse + 
                       loss_weights['recon_pcc'] * recon_pcc + 
                       loss_weights['recon_psd'] * recon_psd)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  NaN/Inf loss at batch {batch_idx} - skipping")
                    nan_batches += 1
                    continue

            # ── Backward + gradient check (FIXED) ───────────────────────────────
            scaler.scale(loss).backward()

            # Unscale ONCE — must happen before any gradient inspection or clipping
            scaler.unscale_(optimizer)

            # Check for NaN gradients after the single unscale_ call
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )

            if has_nan_grad:
                print(f"\n⚠️  NaN gradient at batch {batch_idx} - skipping update")
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                # ✅ CRITICAL: reset scaler state even when skipping step
                scaler.update()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'diff': f'{diff_loss.item():.4f}',
                'mse': f'{recon_mse.item():.4f}',
                'pcc': f'{recon_pcc.item():.4f}',
                'psd': f'{recon_psd.item():.4f}'
            })

        # Aggregate train loss
        valid_batches = len(train_loader) - nan_batches
        if nan_batches > 0:
            print(f"\n⚠️  {nan_batches} NaN batches in epoch {epoch+1}")

        if valid_batches > 0:
            train_loss /= valid_batches
        else:
            print(f"\n❌ All batches had NaN at epoch {epoch+1} - stopping")
            break

        if np.isnan(train_loss):
            print(f"\n❌ Training loss is NaN at epoch {epoch+1} - stopping")
            break

        # ── Validation ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_nan_batches = 0

        with torch.no_grad():
            for x_lr, y_hr, y_sr in val_loader:
                x_lr = x_lr.to(device)
                y_hr = y_hr.to(device)
                B = x_lr.size(0)

                if torch.isnan(x_lr).any() or torch.isnan(y_hr).any():
                    val_nan_batches += 1
                    continue

                z0 = model.encode_hr(y_hr)
                if torch.isnan(z0).any():
                    val_nan_batches += 1
                    continue

                z0 = torch.clamp(z0, min=-10.0, max=10.0)

                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)
                sqrt_alpha     = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon

                if torch.isnan(zt).any():
                    val_nan_batches += 1
                    continue

                lr_pos = get_channel_positions(64, device, B)
                pred_epsilon = model(x_lr, zt, t, lr_pos)

                if torch.isnan(pred_epsilon).any():
                    val_nan_batches += 1
                    continue

                loss_val = criterion(pred_epsilon, epsilon)
                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    val_loss += loss_val.item()

        if val_nan_batches > 0:
            print(f"  ⚠️  {val_nan_batches} validation batches had NaN")

        valid_val_batches = len(val_loader) - val_nan_batches
        if valid_val_batches > 0:
            val_loss /= valid_val_batches

        if np.isnan(val_loss):
            print("  ⚠️  Validation loss is NaN - using previous best")
            val_loss = best_val_loss

        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Detailed metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n{'='*60}")
            print(f"  📊 Computing validation metrics (epoch {epoch+1})...")
            print(f"{'='*60}")
            metrics = validate_with_metrics(
                model, val_loader, device, diff_params, T=T, num_samples=100
            )
            print(f"  NMSE: {metrics['nmse']:.6f}")
            print(f"  PCC:  {metrics['pcc']:.4f}")
            print(f"  SNR:  {metrics['snr']:.2f} dB")
            print(f"{'='*60}\n")

            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics,
            })
            np.save('metrics_history_localizemi.npy', metrics_history)

        # Sanity checks
        if epoch > 10:
            if train_loss < 0.01:
                print("  ⚠️  Loss suspiciously low - possible collapse!")
            elif train_loss > 2.0:
                print("  ⚠️  Loss very high - check gradients!")

        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'diff_params': diff_params,
            'metrics_history': metrics_history,
            'config': {
                'lr_channels': 64,
                'hr_channels': 128,
                'sr_channels': 256,
                'seq_len': 2800,
                'latent_dim': 1024,
                'epoch_duration_ms': 350,
                'filters': 'highpass_0.1Hz_notch_50_100_150_200Hz',
                'mae_checkpoint': 'trial_mae_Localize-MI/results_128ch/best_checkpoint.pth',
                'loss_weights': loss_weights,
                'dataset_info': 'Motor Imagery (MI) - hand/foot movement imagination',
                'mi_bands': 'Beta (13-30 Hz) ERD/ERS, Mu (8-13 Hz) motor cortex',
            },
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'best_stad_localizemi.pt')
            print(f"  ✅ Saved best model (val_loss={val_loss:.6f})")

        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, f'checkpoint_localizemi_epoch_{epoch+1}.pt')
            print("  💾 Saved checkpoint")

    # ── Final summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f}")
    if metrics_history:
        last = metrics_history[-1]
        print(f"   Final NMSE: {last['nmse']:.6f}")
        print(f"   Final PCC:  {last['pcc']:.4f}")
        print(f"   Final SNR:  {last['snr']:.2f} dB")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train STAD on Localize-MI dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--diff_weight', type=float, default=1.0,
                        help='Weight for diffusion loss')
    parser.add_argument('--mse_weight', type=float, default=0.05,
                        help='Weight for MSE reconstruction loss (reduced for MI: amplitude less critical)')
    parser.add_argument('--pcc_weight', type=float, default=0.4,
                        help='Weight for PCC reconstruction loss (moderate for MI: temporal patterns important)')
    parser.add_argument('--psd_weight', type=float, default=0.6,
                        help='Weight for PSD reconstruction loss (HIGH for MI: beta/mu bands critical)')
    args = parser.parse_args()

    loss_weights = {
        'diff': args.diff_weight,
        'recon_mse': args.mse_weight,
        'recon_pcc': args.pcc_weight,
        'recon_psd': args.psd_weight
    }

    train_stad_localizemi(
        data_path='/home/ab_students/EEG-MTP/DATA/Localize-MI/derivatives/epochs',
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
        loss_weights=loss_weights
    )