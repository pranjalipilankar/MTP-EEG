#!/usr/bin/env python3
"""
STAD Training for Localize-MI Dataset (256 channels, 8000Hz HD-EEG)
Channel hierarchy (as per STAD paper):
- LR (Low Res): 64 channels → conditioning
- HR (High Res): 128 channels → MAE encoder → latents
- Target: 256 channels (diffusion super-resolution)
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import amp
from tqdm import tqdm
from pathlib import Path
from scipy.signal import butter, filtfilt
try:
    from einops import Rearrange
except ImportError:
    class Rearrange(nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.pattern = pattern.replace(" ", "")

        def forward(self, x):
            if self.pattern == "btc->bct":
                return x.transpose(1, 2)
            if self.pattern == "bct->btc":
                return x.transpose(1, 2)
            raise ValueError(f"Unsupported Rearrange pattern: {self.pattern}")

    print("Warning: einops not installed; using local Rearrange fallback.")

from mae_for_eeg import MAEforEEG
from spatio_temporal_condition import SpatioTemporalConditionModule
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule


def get_localizemi_channel_indices(target_channels):
    """
    Return fixed Localize-MI channel subsets from 256-channel HD-EEG ordering.
    These subsets provide spatial coverage across the electrode grid.
    """
    if target_channels == 256:
        return np.arange(256, dtype=int)
    if target_channels == 128:
        return np.linspace(0, 255, 128, dtype=int)
    if target_channels == 64:
        return np.linspace(0, 255, 64, dtype=int)
    
    return np.linspace(0, 255, target_channels, dtype=int)


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


def psd_loss(pred, target, fs=500):
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

    for batch in val_loader:
        if samples_processed >= num_samples:
            break

        x_lr = batch['lr'].to(device)
        y_hr = batch['hr'].to(device)
        y_sr = batch['sr'].to(device)
        B = x_lr.size(0)

        # Encode ground truth SR to latent (via SR->HR projection)
        z0_true = model.encode_sr(y_sr)  # (B, 175, 1024)

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
        pred_sr = model.decode_latent_to_sr(z0_pred, x_lr)  # (B, 256, 2080)
        
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
        time_len=2080,
        fs=8000,
        preprocessed=False,
    ):
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels
        self.time_len = time_len
        self.fs = fs
        self.preprocessed = preprocessed

        data_path = Path(data_path)

        self._subject_arrays = []
        self._sample_index = []

        if preprocessed:
            self._load_preprocessed_data(data_path, subjects)
        else:
            self._load_raw_data(data_path, subjects)

        if indices is not None:
            self._sample_index = [self._sample_index[i] for i in indices]
            print(f"Using {len(self._sample_index)} segments from split")

    def _resize_epoch(self, epoch):
        if epoch.shape[1] > self.time_len:
            return epoch[:, :self.time_len]
        if epoch.shape[1] < self.time_len:
            pad_width = ((0, 0), (0, self.time_len - epoch.shape[1]))
            return np.pad(epoch, pad_width, mode='edge')
        return epoch

    def _standardize_epoch(self, epoch):
        epoch = epoch.astype(np.float32, copy=False)
        mean = epoch.mean(axis=-1, keepdims=True)
        std = epoch.std(axis=-1, keepdims=True) + 1e-6
        return (epoch - mean) / std

    def _prepare_epoch(self, epoch, target_channels, apply_filters=True):
        epoch = self._resize_epoch(epoch)
        indices = self.get_egi_channel_indices(epoch.shape[0], target_channels)
        epoch_sub = epoch[indices, :]

        if apply_filters:
            nyquist = 0.5 * self.fs
            low = max(0.1 / nyquist, 0.00001)
            high = min(100.0 / nyquist, 0.99)
            b_band, a_band = butter(4, [low, high], 'band')
            epoch_sub = filtfilt(b_band, a_band, epoch_sub, axis=-1)
            for freq in [50, 100, 150, 200]:
                if freq < nyquist:
                    low_notch = max((freq - 1.0) / nyquist, 0.00001)
                    high_notch = min((freq + 1.0) / nyquist, 0.99999)
                    b_notch, a_notch = butter(2, [low_notch, high_notch], 'bandstop')
                    epoch_sub = filtfilt(b_notch, a_notch, epoch_sub, axis=-1)

        return self._standardize_epoch(epoch_sub)

    def _load_raw_data(self, data_path, subjects):
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
                    epoch = data[epoch_idx]
                    all_epochs.append(self._prepare_epoch(epoch, self.sr_channels))

        if len(all_epochs) == 0:
            raise ValueError(f"No epochs found in {data_path}")

        self.sr_samples = np.asarray(all_epochs, dtype=np.float32)
        self.hr_samples = np.asarray([self._prepare_epoch(epoch, self.hr_channels) for epoch in self.sr_samples], dtype=np.float32)
        self.lr_samples = np.asarray([self._prepare_epoch(epoch, self.lr_channels) for epoch in self.sr_samples], dtype=np.float32)
        self._sample_index = list(range(len(self.sr_samples)))

        print(f"Prepared {len(self.hr_samples)} segments")
        print(f"  SR shape (256ch): {self.sr_samples.shape}")
        print(f"  HR shape (128ch): {self.hr_samples.shape}")
        print(f"  LR shape (64ch):  {self.lr_samples.shape}")

        print(f"Prepared {len(self.hr_samples)} segments")
        print(f"  SR shape (256ch): {self.sr_samples.shape}")
        print(f"  HR shape (128ch): {self.hr_samples.shape}")
        print(f"  LR shape (64ch):  {self.lr_samples.shape}")

    def _load_preprocessed_data(self, data_path, subjects):
        """Load preprocessed data from epochs_prc1 format (X_prc1.npy files)"""
        if subjects == 'all':
            subject_dirs = sorted(data_path.glob('sub-*'))
        else:
            subject_dirs = [data_path / subj for subj in subjects]

        print(f"Loading preprocessed Localize-MI data (X_prc1.npy format)...")
        self._subject_arrays = []
        self._sample_index = []

        for subj_dir in subject_dirs:
            if not subj_dir.exists():
                continue

            X_prc1_file = subj_dir / 'X_prc1.npy'
            if X_prc1_file.exists():
                data = np.load(X_prc1_file, mmap_mode='r')  # (n_windows, 256, T)
                subject_idx = len(self._subject_arrays)
                self._subject_arrays.append(data)
                for window_idx in range(data.shape[0]):
                    self._sample_index.append((subject_idx, window_idx))
                print(f"  {subj_dir.name}: loaded {data.shape[0]} windows, shape {data.shape}")

        if len(self._sample_index) == 0:
            raise ValueError(f"No preprocessed data found in {data_path}")

        print(f"Loaded {len(self._sample_index)} preprocessed windows from {len(self._subject_arrays)} subjects")

    def prepare_samples(self, epochs, target_channels):
        """Prepare samples with channel downsampling and preprocessing"""
        return np.asarray([self._prepare_epoch(epoch, target_channels) for epoch in epochs], dtype=np.float32)

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
        if self.preprocessed:
            return len(self._sample_index)
        return len(self.hr_samples)

    def __getitem__(self, idx):
        if self.preprocessed:
            subject_idx, window_idx = self._sample_index[idx]
            epoch = np.asarray(self._subject_arrays[subject_idx][window_idx], dtype=np.float32)
            epoch = self._resize_epoch(epoch)
            sr = self._standardize_epoch(epoch)
            hr = self._prepare_epoch(epoch, self.hr_channels, apply_filters=False)
            lr = self._prepare_epoch(epoch, self.lr_channels, apply_filters=False)
            return {
                'lr': torch.from_numpy(lr).float(),
                'hr': torch.from_numpy(hr).float(),
                'sr': torch.from_numpy(sr).float(),
            }

        return {
            'lr': torch.from_numpy(self.lr_samples[idx]).float(),
            'hr': torch.from_numpy(self.hr_samples[idx]).float(),
            'sr': torch.from_numpy(self.sr_samples[idx]).float(),
        }


class STAD_LocalizeMI(nn.Module):
    """
    STAD model adapted for Localize-MI HD-EEG dataset.
    Channel hierarchy (SEED4-style STAD flow):
    - Input LR: 64 channels → STC conditioning
    - Input SR: 256 channels → project to HR(128) → MAE encoder → latent
    - Output SR: 256 channels (decode HR then upsample head)
    """

    def __init__(
        self,
        lr_channels=64,
        hr_channels=128,
        sr_channels=256,
        seq_len=2080,
        latent_dim=1024,
        n_harmonics=8,
        mae_time_len=256,
        mae_patch_size=16,
        mae_embed_dim=512,
        mae_depth=12,
        mae_num_heads=8,
        mae_decoder_embed_dim=256,
        mae_decoder_depth=8,
        mae_decoder_num_heads=8,
        mae_mlp_ratio=2.0,
    ):
        super().__init__()

        num_patches = mae_time_len // mae_patch_size

        self.mae = MAEforEEG(
            time_len=mae_time_len,
            patch_size=mae_patch_size,
            embed_dim=mae_embed_dim,
            in_chans=hr_channels,
            depth=mae_depth,
            num_heads=mae_num_heads,
            decoder_embed_dim=mae_decoder_embed_dim,
            decoder_depth=mae_decoder_depth,
            decoder_num_heads=mae_decoder_num_heads,
            mlp_ratio=mae_mlp_ratio,
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
            cond_dim=latent_dim,
            n_layers=8,
            n_heads=16,
            use_multiscale_conv=True,  # Enable multi-scale 1D convolutions (STAD Eq. 3)
        )

        # Explicit projection before MAE encoding: SR(256) -> HR(128)
        self.sr_to_hr_projection = nn.Conv1d(
            sr_channels, hr_channels, kernel_size=1, bias=False
        )

        # Super-resolution upsampling module: 128ch → 256ch (STAD paper: Decoder outputs SR directly)
        self.hr_to_sr_upsampler = nn.Sequential(
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
        self.mae_time_len = mae_time_len
        self.sr_channels = sr_channels
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels

    def encode_sr(self, sr_eeg):
        """Encode SR EEG to latent via explicit SR->HR projection."""
        hr_projected = self.sr_to_hr_projection(sr_eeg)
        if hr_projected.size(-1) != self.mae_time_len:
            hr_projected = torch.nn.functional.interpolate(
                hr_projected.unsqueeze(1), size=(self.hr_channels, self.mae_time_len),
                mode='bilinear', align_corners=False
            ).squeeze(1)
        latent, _, _ = self.mae.forward_encoder(hr_projected, mask_ratio=0.0)
        latent = latent[:, 1:, :]
        return latent

    def decode_latent_to_sr(self, latent, lr_eeg=None):
        """
        Decode latent to 256ch SR (STAD paper: Decoder D outputs SR directly).
        LR conditioning only used in MTD during denoising, not in decoder.
        
        Args:
            latent: (B, num_patches, latent_dim) - Denoised latent from diffusion
            lr_eeg: (B, 64, T) - Only used to get target temporal length
        """
        B = latent.size(0)
        target_len = lr_eeg.size(-1) if lr_eeg is not None else 2080
        
        # Decode latent to 128ch HR via MAE decoder
        cls_token = self.mae.cls_token.expand(B, -1, -1)
        latent_with_cls = torch.cat([cls_token, latent], dim=1)
        hr_patches = self.mae.forward_decoder(
            latent_with_cls, 
            torch.zeros(B, self.num_patches, dtype=torch.long, device=latent.device)
        )
        
        # Unpatchify: convert patches back to full signal
        hr_eeg = self.mae.unpatchify(hr_patches)  # (B, 128, 2080) - full EEG signal
        
        # Resize MAE output to match target length if needed
        if hr_eeg.size(-1) != target_len:
            hr_eeg = torch.nn.functional.interpolate(
                hr_eeg.unsqueeze(1), size=(self.hr_channels, target_len),
                mode='bilinear', align_corners=False
            ).squeeze(1)  # (B, 128, target_len)
        
        # Apply learned upsampling: 128ch → 256ch (per STAD Figure 2: Decoder → SR EEG)
        hr_eeg_t = hr_eeg.transpose(1, 2)  # (B, target_len, 128)
        sr_eeg_t = self.hr_to_sr_upsampler(hr_eeg_t)  # (B, target_len, 256)
        sr_eeg = sr_eeg_t.transpose(1, 2)  # (B, 256, target_len)
        
        return sr_eeg

    def forward(self, lr_eeg, zt, t_steps, lr_chan_pos):
        """Forward pass: predict noise"""
        cond_tokens, cond_pooled = self.stc(lr_eeg, lr_chan_pos, t_steps)
        return self.mtd(zt, t_steps, cond_tokens, cond_pooled)


def load_mae_fold_subject_split(mae_results_dir, fold):
    """Load train/val subjects for a MAE fold from fold_splits.json."""
    split_file = Path(mae_results_dir) / 'fold_splits.json'
    if not split_file.exists():
        raise FileNotFoundError(f"fold_splits.json not found: {split_file}")

    with open(split_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    matched = None
    for item in splits:
        if int(item.get('fold', -1)) == int(fold):
            matched = item
            break

    if matched is None:
        available = [int(x.get('fold', -1)) for x in splits]
        raise ValueError(
            f"Fold {fold} not found in {split_file}. Available folds: {available}"
        )

    return matched['train_subjects'], matched['val_subjects']


def load_matching_state_dict(target_module, checkpoint_state_dict):
    """Load only tensors whose names and shapes match the target module."""
    target_state_dict = target_module.state_dict()
    filtered_state_dict = {}
    skipped_keys = []

    for key, value in checkpoint_state_dict.items():
        if key in target_state_dict and target_state_dict[key].shape == value.shape:
            filtered_state_dict[key] = value
        else:
            skipped_keys.append(key)

    missing_keys, unexpected_keys = target_module.load_state_dict(filtered_state_dict, strict=False)
    return missing_keys, unexpected_keys, skipped_keys


def train_stad_localizemi(
    data_path,
    mae_results_dir,
    mae_fold=3,
    num_epochs=300,
    batch_size=8,
    lr=2e-4,
    resume_from=None,
    train_workers=0,
    val_workers=0,
    pin_memory=False,
    persistent_workers=False,
    preprocessed=True,
    loss_weights={'diff': 1.0, 'sr_l1': 0.5},
):
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

    # Keep outputs/checkpoints anchored to this script directory for reliable resume.
    run_dir = Path(__file__).resolve().parent
    best_ckpt_path = run_dir / 'best_stad_localizemi.pt'
    metrics_path = run_dir / 'metrics_history_localizemi.npy'

    diff_weight = float(loss_weights.get('diff', 1.0))
    sr_l1_weight = float(loss_weights.get('sr_l1', loss_weights.get('recon_mse', 0.5)))

    print("=" * 60)
    print("STAD Training - Localize-MI HD-EEG Dataset (256 channels, 8000Hz)")
    print("Motor Imagery Task: Hand/Foot Movement Imagination")
    print("=" * 60)
    print(f"\nLoss Weights (SEED4-style objective):")
    print(f"   Diffusion:      {diff_weight:.2f}")
    print(f"   SR L1:          {sr_l1_weight:.2f}")
    print("   Total:          diff + sr_l1_weight * sr_l1\n")
    print(f"Dataset mode: {'preprocessed (X_prc1.npy)' if preprocessed else 'raw epochs (*_epochs.npy)'}")

    # ── Dataset splits (match MAE fold subjects to prevent leakage) ────────────
    print("📦 Preparing dataset splits (from MAE fold subjects)...")
    train_subjects, val_subjects = load_mae_fold_subject_split(mae_results_dir, mae_fold)
    print(f"  MAE fold: {mae_fold}")
    print(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"  Val subjects ({len(val_subjects)}): {val_subjects}")

    train_dataset = LocalizeMISTADDataset(
        data_path, subjects=train_subjects,
        lr_channels=64, hr_channels=128, sr_channels=256, time_len=2080,
        preprocessed=preprocessed,
    )
    val_dataset = LocalizeMISTADDataset(
        data_path, subjects=val_subjects,
        lr_channels=64, hr_channels=128, sr_channels=256, time_len=2080,
        preprocessed=preprocessed,
    )

    train_persistent_workers = persistent_workers and train_workers > 0
    val_persistent_workers = persistent_workers and val_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True,
        num_workers=train_workers, drop_last=True, pin_memory=pin_memory,
        persistent_workers=train_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=pin_memory,
        persistent_workers=val_persistent_workers,
    )

    print(f"\n📊 Dataset (STAD channel hierarchy):")
    print(f"  LR (conditioning): 64 channels")
    print(f"  HR (MAE input):   128 channels")
    print(f"  SR (target):      256 channels")
    print(f"  Train: {len(train_dataset)} epochs")
    print(f"  Val:   {len(val_dataset)} epochs")
    print(f"  Batches/epoch: {len(train_loader)}\n")

    # ── Resolve MAE checkpoint and build model with matching MAE architecture ──
    mae_checkpoint = str(Path(mae_results_dir) / f'fold_{mae_fold}' / 'best_model.pth')
    mae_payload = None
    mae_state_dict = None
    mae_cfg = {}

    if os.path.exists(mae_checkpoint):
        mae_payload = torch.load(mae_checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in mae_payload:
            mae_state_dict = mae_payload['model_state_dict']
        elif 'model' in mae_payload:
            mae_state_dict = mae_payload['model']
        else:
            mae_state_dict = mae_payload
        if isinstance(mae_payload, dict) and isinstance(mae_payload.get('config', None), dict):
            mae_cfg = mae_payload['config']

    mae_time_len = int(mae_cfg.get('time_len', 256))
    mae_patch_size = int(mae_cfg.get('patch_size', 16))
    mae_embed_dim = int(mae_cfg.get('embed_dim', 512))
    mae_depth = int(mae_cfg.get('depth', 12))
    mae_num_heads = int(mae_cfg.get('num_heads', 8))
    mae_decoder_embed_dim = int(mae_cfg.get('decoder_embed_dim', 256))
    mae_decoder_depth = int(mae_cfg.get('decoder_depth', 8))
    mae_decoder_num_heads = int(mae_cfg.get('decoder_num_heads', 8))
    mae_mlp_ratio = float(mae_cfg.get('mlp_ratio', 2.0))

    model = STAD_LocalizeMI(
        lr_channels=64,
        hr_channels=128,
        sr_channels=256,
        seq_len=2080,
        latent_dim=mae_embed_dim,
        n_harmonics=8,
        mae_time_len=mae_time_len,
        mae_patch_size=mae_patch_size,
        mae_embed_dim=mae_embed_dim,
        mae_depth=mae_depth,
        mae_num_heads=mae_num_heads,
        mae_decoder_embed_dim=mae_decoder_embed_dim,
        mae_decoder_depth=mae_decoder_depth,
        mae_decoder_num_heads=mae_decoder_num_heads,
        mae_mlp_ratio=mae_mlp_ratio,
    ).to(device)

    if mae_state_dict is not None:
        missing_keys, unexpected_keys, skipped_keys = load_matching_state_dict(model.mae, mae_state_dict)
        epoch = mae_payload.get('epoch', 'N/A') if isinstance(mae_payload, dict) else 'N/A'
        val_loss = mae_payload.get('val_loss', 'N/A') if isinstance(mae_payload, dict) else 'N/A'
        val_cor = mae_payload.get('val_cor', mae_payload.get('val_corr', 'N/A')) if isinstance(mae_payload, dict) else 'N/A'
        print(f"✅ Loaded pretrained MAE from k-fold training (fold {mae_fold})")
        print(f"   Checkpoint: {mae_checkpoint}")
        print(
            f"   MAE config: time_len={mae_time_len}, patch={mae_patch_size}, "
            f"embed={mae_embed_dim}, depth={mae_depth}, heads={mae_num_heads}"
        )
        if skipped_keys:
            print(f"   Skipped incompatible MAE tensors: {len(skipped_keys)}")
        if missing_keys:
            print(f"   Missing MAE tensors after load: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   Unexpected MAE tensors after load: {len(unexpected_keys)}")
        if isinstance(epoch, int):
            print(f"   Best epoch: {epoch}")
        if isinstance(val_loss, (int, float)):
            print(f"   Val loss: {val_loss:.6f}")
        if isinstance(val_cor, (int, float)):
            print(f"   Val correlation: {val_cor:.4f}")
        print()
        del mae_payload
        torch.cuda.empty_cache()
    else:
        print(f"⚠️  MAE checkpoint not found: {mae_checkpoint}")
        print("   Training without pretrained MAE weights\n")

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
    scaler = amp.GradScaler('cuda')

    best_val_loss = float('inf')
    metrics_history = []
    start_epoch = 0

    # ── Resume from checkpoint ───────────────────────────────────────────────────
    resume_path = None
    if resume_from:
        candidate = Path(resume_from)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            resume_path = candidate
        else:
            print(f"\n⚠️  Resume checkpoint not found: {candidate}")

    if resume_path is not None:
        print(f"\nResuming from checkpoint: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
        if unexpected_keys:
            print(f"  Ignoring unexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"  Missing keys (random init): {missing_keys}")
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        metrics_history = ckpt.get('metrics_history', [])
        if start_epoch < num_epochs:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, num_epochs - start_epoch
            )
        print(f"Resumed from epoch {ckpt['epoch']}")
        print(f"Best val loss so far: {best_val_loss:.6f}")
        print(f"Continuing from epoch {start_epoch}\n")
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
                num_workers=train_workers, pin_memory=pin_memory,
                persistent_workers=train_persistent_workers,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=new_batch_size, shuffle=False,
                num_workers=val_workers, pin_memory=pin_memory,
                persistent_workers=val_persistent_workers,
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

        use_tqdm = sys.stdout.isatty()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if use_tqdm else train_loader
        for batch_idx, batch in enumerate(pbar):
            x_lr  = batch['lr'].to(device)
            y_hr  = batch['hr'].to(device)
            y_sr  = batch['sr'].to(device)
            B = x_lr.size(0)

            # Skip batches with NaN inputs
            if torch.isnan(x_lr).any() or torch.isnan(y_sr).any():
                print(f"\n⚠️  NaN in inputs at batch {batch_idx} - skipping")
                nan_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda', dtype=torch.float16):
                # Encode SR -> HR projection -> latent
                z0 = model.encode_sr(y_sr)  # (B, 175, 1024)

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

                # Diffusion denoising loss.
                diff_loss = criterion(pred_epsilon, epsilon)

                # SEED4-style SR supervision from denoised latent estimate.
                pred_z0 = (zt - sqrt_one_minus * pred_epsilon) / (sqrt_alpha + 1e-8)
                pred_z0 = torch.clamp(pred_z0, min=-10.0, max=10.0)
                sr_pred = model.decode_latent_to_sr(pred_z0, x_lr)
                sr_l2 = F.mse_loss(sr_pred.float(), y_sr.float())

                loss = diff_weight * diff_loss + sr_l1_weight * sr_l2

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
            if use_tqdm:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'diff': f'{diff_loss.item():.4f}',
                    'sr_l2': f'{sr_l2.item():.4f}'
                })
            elif (batch_idx + 1) % 50 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} diff={diff_loss.item():.4f} sr_l2={sr_l2.item():.4f}"
                )

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
        val_nmse_scores = []
        val_pcc_scores = []
        val_snr_scores = []

        with torch.no_grad():
            for batch in val_loader:
                x_lr = batch['lr'].to(device)
                y_hr = batch['hr'].to(device)
                y_sr = batch['sr'].to(device)
                B = x_lr.size(0)

                if torch.isnan(x_lr).any() or torch.isnan(y_sr).any():
                    val_nan_batches += 1
                    continue

                z0 = model.encode_sr(y_sr)
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

                diff_val = criterion(pred_epsilon, epsilon)
                pred_z0 = (zt - sqrt_one_minus * pred_epsilon) / (sqrt_alpha + 1e-8)
                pred_z0 = torch.clamp(pred_z0, min=-10.0, max=10.0)
                sr_pred_val = model.decode_latent_to_sr(pred_z0, x_lr)
                sr_l2_val = F.mse_loss(sr_pred_val.float(), y_sr.float())
                loss_val = diff_weight * diff_val + sr_l1_weight * sr_l2_val

                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    val_loss += loss_val.item()
                    val_nmse_scores.append(compute_nmse(sr_pred_val.float(), y_sr.float()))
                    val_pcc_scores.append(compute_pcc(sr_pred_val.float(), y_sr.float()))
                    val_snr_scores.append(compute_snr(sr_pred_val.float(), y_sr.float()))

        if val_nan_batches > 0:
            print(f"  ⚠️  {val_nan_batches} validation batches had NaN")

        valid_val_batches = len(val_loader) - val_nan_batches
        if valid_val_batches > 0:
            val_loss /= valid_val_batches

        if np.isnan(val_loss):
            print("  ⚠️  Validation loss is NaN - using previous best")
            val_loss = best_val_loss

        mean_val_nmse = float(np.mean(val_nmse_scores)) if val_nmse_scores else float('inf')
        mean_val_pcc = float(np.mean(val_pcc_scores)) if val_pcc_scores else 0.0
        mean_val_snr = float(np.mean(val_snr_scores)) if val_snr_scores else -float('inf')

        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"PCC: {mean_val_pcc:.4f} | NMSE: {mean_val_nmse:.4f} | "
            f"SNR: {mean_val_snr:.2f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'nmse': mean_val_nmse,
            'pcc': mean_val_pcc,
            'snr': mean_val_snr,
        })
        np.save(str(metrics_path), metrics_history)

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
                'seq_len': 2080,
                'latent_dim': 1024,
                'epoch_duration_ms': 260,
                'filters': 'highpass_0.1Hz_notch_50_100_150_200Hz',
                'mae_checkpoint': mae_checkpoint,
                'mae_fold': mae_fold,
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'loss_weights': {'diff': diff_weight, 'sr_l1': sr_l1_weight},
                'dataset_info': 'Motor Imagery (MI) - hand/foot movement imagination',
                'mi_bands': 'Beta (13-30 Hz) ERD/ERS, Mu (8-13 Hz) motor cortex',
            },
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, str(best_ckpt_path))
            print(f"  Saved best model: {best_ckpt_path.name} (val_loss={val_loss:.6f})")

        if (epoch + 1) % 20 == 0:
            epoch_ckpt_path = run_dir / f'checkpoint_localizemi_epoch_{epoch+1}.pt'
            torch.save(checkpoint, str(epoch_ckpt_path))
            print(f"  Saved checkpoint: {epoch_ckpt_path.name}")
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
    parser.add_argument('--data_path', type=str,
                        default='/home/arnav-a5000/MTP-EEG/DATA/Localize-MI/derivatives/epochs_prc1',
                        help='Path to preprocessed Localize-MI folder with sub-*/X_prc1.npy')
    parser.add_argument('--mae_results_dir', type=str,
                        default='/home/arnav-a5000/MTP-EEG/trial_mae_Localize-MI/results_128ch_kfold_prc1_run_20260422_205948',
                        help='Path to MAE k-fold results containing fold_splits.json and fold_*/best_model.pth')
    parser.add_argument('--mae_fold', type=int, default=3,
                        help='MAE fold to use for checkpoint and subject split (1-5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_workers', type=int, default=0,
                        help='Training DataLoader workers (set 0 to minimize host RAM usage)')
    parser.add_argument('--val_workers', type=int, default=0,
                        help='Validation DataLoader workers (set 0 to minimize host RAM usage)')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Enable pinned CPU memory for faster host->GPU transfer (uses more RAM)')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='Keep DataLoader workers alive between epochs (requires workers > 0)')
    parser.add_argument('--raw_epochs', action='store_true',
                        help='Use raw epochs from derivatives/epochs (sub-*/eeg/*_epochs.npy) instead of preprocessed X_prc1.npy')
    parser.add_argument('--diff_weight', type=float, default=1.0,
                        help='Weight for diffusion loss')
    parser.add_argument('--sr_l1_weight', type=float, default=0.5,
                        help='Weight for SR L1 reconstruction loss (SEED4-style objective)')
    args = parser.parse_args()

    loss_weights = {
        'diff': args.diff_weight,
        'sr_l1': args.sr_l1_weight,
    }

    train_stad_localizemi(
        data_path=args.data_path,
        mae_results_dir=args.mae_results_dir,
        mae_fold=args.mae_fold,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        preprocessed=(not args.raw_epochs),
        loss_weights=loss_weights
    )