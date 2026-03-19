#!/usr/bin/env python3
"""
STAD Training for SEED Dataset (62 channels, 200Hz)
Adapted from train_stad_complete.py with SEED-specific parameters
Uses pretrained MAE from trial_mae_SEED/results/best_checkpoint.pth
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from pathlib import Path



from spatio_temporal_condition import SpatioTemporalConditionModule
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule
from mae_for_eeg import MAEforEEG

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
    """
    Get channel positions for SEED dataset
    SEED uses 62-channel cap with standard positions
    """
    if n_channels == 62:
        # SEED 62-channel positions (approximated from standard 10-20 system)
        # Create positions in a circular pattern to ensure exactly 62 positions
        positions = []
        # Concentric circles with different radii
        radii = [0.0, 0.25, 0.5, 0.75, 1.0]
        channels_per_ring = [1, 8, 12, 18, 23]  # Total = 62
        
        for radius, n_in_ring in zip(radii, channels_per_ring):
            if len(positions) >= 62:
                break
            if n_in_ring == 1:  # Center
                positions.append([0.0, 0.0])
            else:
                for i in range(n_in_ring):
                    if len(positions) >= 62:
                        break
                    angle = 2 * np.pi * i / n_in_ring
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    positions.append([x, y])
        
        positions = np.array(positions[:62], dtype=np.float32)
    elif n_channels == 31:
        # For 31 channels (downsampled from 62)
        full_pos = get_channel_positions(62, 'cpu', 1).squeeze(0).numpy()
        indices = np.linspace(0, 61, 31, dtype=int)
        positions = full_pos[indices]
    elif n_channels == 16:
        # For 16 channels (LR - downsampled from 62)
        full_pos = get_channel_positions(62, 'cpu', 1).squeeze(0).numpy()
        indices = np.linspace(0, 61, 16, dtype=int)
        positions = full_pos[indices]
    else:
        raise ValueError(f"Unsupported channel count: {n_channels}")
    
    return torch.tensor(positions, device=device).unsqueeze(0).expand(batch_size, -1, -1)

def reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50):
    """Reconstruct 62-channel SR EEG from 16-channel LR using diffusion sampling"""
    model.eval()
    with torch.no_grad():
        B = x_lr.shape[0]
        lr_pos = get_channel_positions(16, device, B)  # LR has 16 channels
        
        # Start from noise (B, num_patches, latent_dim)
        # For SEED: segment_length=4000, patch_size=8 → 500 patches
        zt = torch.randn(B, 500, 1024, device=device)  # latent_dim=1024 for SEED
        timesteps = torch.linspace(999, 0, steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            
            # Get conditioning
            cond_tokens, cond_pooled = model.stc(x_lr, lr_pos, t_batch)
            
            # Predict noise
            pred_noise = model.mtd(zt, t_batch, cond_tokens, cond_pooled)
            
            # DDIM step
            alpha_t = diff_params['sqrt_alphas_cumprod'][t] ** 2
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = diff_params['sqrt_alphas_cumprod'][t_prev] ** 2
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            pred_x0 = (zt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            if i < len(timesteps) - 1:
                zt = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                zt = pred_x0
        
        # Decode normalized latent to 62-channel SR using LR conditioning
        sr_eeg = model.decode_latent_to_sr(zt, x_lr)
        
        return sr_eeg

def validate_reconstruction(model, val_loader, diff_params, device):
    """Validate with proper metrics"""
    metrics = {'PCC': [], 'SNR': [], 'NMSE': [], 'MAE': []}
    
    with torch.no_grad():
        for i, (x_lr, y_hr) in enumerate(val_loader):
            if i >= 10:
                break
                
            x_lr, y_hr = x_lr.to(device), y_hr.to(device)
            sr_eeg = reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50)
            
            sr_np = sr_eeg.cpu().numpy()
            hr_np = y_hr.cpu().numpy()
            
            # Align length
            min_len = min(sr_np.shape[2], hr_np.shape[2])
            sr_np = sr_np[:, :, :min_len]
            hr_np = hr_np[:, :, :min_len]
            
            # Channel-wise PCC
            for b in range(sr_np.shape[0]):
                for ch in range(sr_np.shape[1]):
                    sr_sig = sr_np[b, ch]
                    hr_sig = hr_np[b, ch]
                    
                    if np.std(sr_sig) > 1e-6 and np.std(hr_sig) > 1e-6:
                        pcc = np.corrcoef(sr_sig, hr_sig)[0, 1]
                        if not np.isnan(pcc):
                            metrics['PCC'].append(pcc)
            
            # Other metrics
            mse = np.mean((sr_np - hr_np) ** 2)
            nmse = mse / (np.mean(hr_np ** 2) + 1e-10)
            snr = 10 * np.log10((np.mean(hr_np ** 2) + 1e-10) / (mse + 1e-10))
            mae = np.mean(np.abs(sr_np - hr_np))
            
            metrics['NMSE'].append(nmse)
            metrics['SNR'].append(snr)
            metrics['MAE'].append(mae)
    
    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}

class SEEDSTADDataset(Dataset):
    """SEED dataset for STAD training (super-resolution 16→31 channels with MAE, then 31→62)"""
    
    def __init__(self, data_path, split='train', lr_channels=16, hr_channels=31, sr_channels=62,
                 segment_length=4000, fs=200):
        """
        Args:
            data_path: Path to SEED_processed folder
            split: 'train', 'val', or 'test'
            lr_channels: Low-res channel count (16)
            hr_channels: High-res channel count (31) - used for MAE
            sr_channels: Super-res channel count (62) - final output
            segment_length: Segment length in samples (4000 = 20s @ 200Hz)
            fs: Sampling rate (200Hz for SEED)
        """
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels
        self.segment_length = segment_length
        self.fs = fs
        
        # Load data from split
        split_path = Path(data_path) / split
        subject_files = sorted(split_path.glob('*.npy'))
        
        if len(subject_files) == 0:
            raise ValueError(f"No data found in {split_path}")
        
        print(f"Loading {split} data from {len(subject_files)} subjects...")
        
        # Load all data
        all_trials = []
        for subj_file in subject_files:
            data = np.load(subj_file)  # (num_trials, 62, 104000)
            for trial_idx in range(data.shape[0]):
                all_trials.append(data[trial_idx])  # (62, 104000)
        
        all_trials = np.array(all_trials)  # (total_trials, 62, 104000)
        print(f"Loaded {len(all_trials)} trials")
        
        # Prepare SR (62), HR (31), and LR (16) segments
        self.sr_samples = self.prepare_segments(all_trials, sr_channels)  # 62 channels (target)
        self.hr_samples = self.prepare_segments(all_trials, hr_channels)  # 31 channels (for MAE)
        self.lr_samples = self.prepare_segments(all_trials, lr_channels)  # 16 channels (input)
        
        print(f"Created {len(self.sr_samples)} segments")
    
    def prepare_segments(self, trials, target_channels):
        """Prepare segments with channel downsampling and preprocessing"""
        num_trials, num_channels, trial_len = trials.shape
        
        # Channel selection
        indices = np.linspace(0, num_channels-1, target_channels, dtype=int)
        trials_sub = trials[:, indices, :]  # (num_trials, target_channels, trial_len)
        
        # Bandpass filter (1-40 Hz)
        def bandpass_filter(data):
            nyquist = 0.5 * self.fs
            b, a = butter(4, [1.0/nyquist, 40.0/nyquist], 'band')
            return filtfilt(b, a, data, axis=-1)
        
        trials_filtered = np.array([bandpass_filter(trial) for trial in trials_sub])
        
        # Create segments
        segments = []
        for trial in trials_filtered:
            # Non-overlapping segments
            for start in range(0, trial.shape[-1] - self.segment_length + 1, self.segment_length):
                seg = trial[:, start:start + self.segment_length]
                segments.append(seg)
        
        segments = np.stack(segments)  # (num_segments, target_channels, segment_length)
        
        # Normalize per segment per channel
        for seg_idx in range(len(segments)):
            for ch in range(segments.shape[1]):
                mean = segments[seg_idx, ch].mean()
                std = segments[seg_idx, ch].std() + 1e-6
                segments[seg_idx, ch] = (segments[seg_idx, ch] - mean) / std
        
        return segments.astype(np.float32)
    
    def __len__(self):
        return len(self.sr_samples)
    
    def __getitem__(self, idx):
        # Return LR (16ch) and SR (62ch) for training
        # HR (31ch) is used internally by MAE
        return torch.tensor(self.lr_samples[idx]), torch.tensor(self.sr_samples[idx])

class STAD_SEED(nn.Module):
    """STAD model adapted for SEED dataset
    
    Architecture:
    - LR input: 16 channels
    - HR (MAE trained): 31 channels
    - SR output: 62 channels
    - MAE provides latent representations from 31-channel encoding
    - STC conditions on 16-channel LR input
    - MTD denoises to produce 62-channel SR output
    """
    
    def __init__(self, lr_channels=16, hr_channels=31, sr_channels=62, seq_len=4000, latent_dim=1024, n_harmonics=8):
        super().__init__()
        
        # SEED MAE parameters
        patch_size = 8  # From config_seed.py
        num_patches = seq_len // patch_size  # 4000 / 8 = 500
        
        # MAE trained on 31 channels (HR)
        self.mae = MAEforEEG(
            time_len=seq_len,
            patch_size=patch_size,
            embed_dim=1024,  # From SEED config
            in_chans=hr_channels,  # 31 channels
            depth=24,  # From SEED config
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            mlp_ratio=1.0  # CRITICAL: Match pretrained MAE config
        )
        
        self.sr_channels = sr_channels  # 62
        self.hr_channels = hr_channels  # 31
        
        # Spatio-Temporal Conditioning (on 16-channel LR input)
        self.stc = SpatioTemporalConditionModule(
            lr_channels,  # 16 channels
            seq_len,
            embed_dim=latent_dim,  # 1024 to match MAE
            n_harmonics=n_harmonics,
            patch_size=16,
            n_transformer_layers=4,
            n_heads=16
        )
        
        # Multi-scale Transformer Denoising (for 62-channel output)
        # Note: Works in latent space (num_patches, latent_dim)
        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches,  # 500 patches
            latent_dim=latent_dim,  # 1024
            n_layers=8,
            n_heads=16
        )
        
        # ✅ FIX: Proper super-resolution upsampling that uses LR conditioning
        # Instead of naive linear projection, use conditioning-aware upsampling
        self.sr_upsample = nn.ModuleDict({
            # Project 31ch MAE output to intermediate representation
            'hr_proj': nn.Linear(hr_channels, hr_channels * 2),
            # Project 16ch LR conditioning to match intermediate
            'lr_proj': nn.Linear(lr_channels, hr_channels * 2),
            # Fuse and upsample to 62ch
            'fusion': nn.Sequential(
                nn.Linear(hr_channels * 2, hr_channels * 4),
                nn.GELU(),
                nn.Linear(hr_channels * 4, sr_channels)
            )
        })
        
        self.lr_channels = lr_channels
        self.latent_dim = latent_dim
        self.num_patches = num_patches
    
    def decode_latent_to_sr(self, latent_normalized, lr_eeg):
        """Decode normalized latent to 62-channel SR EEG using LR conditioning
        
        Args:
            latent_normalized: (B, 500, 1024) - Denoised latent
            lr_eeg: (B, 16, T) - Low-resolution input for conditioning
        """
        B = latent_normalized.shape[0]
        
        # Decode latent to 31-channel HR
        x = self.mae.decoder_embed(latent_normalized)  # (B, 500, decoder_embed_dim)
        x = x + self.mae.decoder_pos_embed[:, 1:, :]  # Skip CLS position
        for blk in self.mae.decoder_blocks:
            x = blk(x)
        x = self.mae.decoder_norm(x)
        pred_patches = self.mae.decoder_pred(x)  # (B, 500, C*patch_size)
        hr_eeg = self.mae.unpatchify(pred_patches)  # (B, 31, T)
        
        # ✅ FIX: Conditioning-aware upsampling 31→62 using LR
        # Transpose for processing
        hr_eeg_t = hr_eeg.transpose(1, 2)  # (B, T, 31)
        lr_eeg_t = lr_eeg.transpose(1, 2)  # (B, T, 16)
        
        # Project both to intermediate space
        hr_feat = self.sr_upsample['hr_proj'](hr_eeg_t)  # (B, T, 62)
        lr_feat = self.sr_upsample['lr_proj'](lr_eeg_t)  # (B, T, 62)
        
        # Fuse: HR provides structure, LR provides spatial guidance
        fused = hr_feat + lr_feat  # Residual connection
        sr_eeg_t = self.sr_upsample['fusion'](fused)  # (B, T, 62)
        
        sr_eeg = sr_eeg_t.transpose(1, 2)  # (B, 62, T)
        
        return sr_eeg
    
    def encode_sr_to_latent(self, sr_eeg):
        """Encode 62-channel SR EEG to normalized latent via 31-channel MAE
        
        Note: MAE was trained on 31 channels, so we downsample 62→31,
        encode to latent, then upsample back during decoding
        """
        B = sr_eeg.shape[0]
        
        # Downsample 62→31 channels for MAE encoding
        indices = torch.linspace(0, self.sr_channels-1, self.hr_channels, dtype=torch.long, device=sr_eeg.device)
        hr_eeg = sr_eeg[:, indices, :]  # (B, 31, T)
        
        # Encode with MAE (without masking)
        # Patch embed
        x = self.mae.patch_embed(hr_eeg)  # (B, num_patches, embed_dim)
        # Add position embedding
        x = x + self.mae.pos_embed[:, 1:, :]  # Skip CLS position
        # Apply encoder blocks
        for blk in self.mae.blocks:
            x = blk(x)
        latent = self.mae.norm(x)  # (B, 500, 1024)
        
        # Normalize latent to match diffusion noise scale N(0,1)
        # This is critical for stable diffusion training
        latent_mean = latent.mean(dim=(1, 2), keepdim=True)
        latent_std = latent.std(dim=(1, 2), keepdim=True) + 1e-6
        latent_normalized = (latent - latent_mean) / latent_std
        
        return latent_normalized
    
    def forward(self, lr_eeg, zt, t_steps, lr_chan_pos):
        """Forward pass: predict noise"""
        cond_tokens, cond_pooled = self.stc(lr_eeg, lr_chan_pos, t_steps)
        return self.mtd(zt, t_steps, cond_tokens, cond_pooled)

def train_stad_seed(data_path, num_epochs=300, batch_size=16, lr=2e-4):
    """Train STAD on SEED dataset
    
    Architecture:
    - LR input: 16 channels
    - HR (MAE trained): 31 channels
    - SR output: 62 channels
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*60)
    print("STAD Training - SEED Dataset")
    print("LR: 16 channels → HR: 31 channels (MAE) → SR: 62 channels")
    print("="*60)
    
    # Load dataset
    train_dataset = SEEDSTADDataset(data_path, 'train', segment_length=4000)
    val_dataset = SEEDSTADDataset(data_path, 'val', segment_length=4000)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, 
                             num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    print(f"\n📊 Dataset:")
    print(f"  Train: {len(train_dataset)} segments")
    print(f"  Val:   {len(val_dataset)} segments")
    print(f"  Batches/epoch: {len(train_loader)}\n")
    
    # Model
    model = STAD_SEED(
        lr_channels=16,   # Low-resolution input
        hr_channels=31,   # High-resolution (MAE trained on this)
        sr_channels=62,   # Super-resolution output
        seq_len=4000,
        latent_dim=1024,
        n_harmonics=8
    ).to(device)
    
    # Load pretrained MAE
    mae_checkpoint = '/home/ab_students/EEG-MTP/trial_mae_SEED/results/best_checkpoint.pth'
    if os.path.exists(mae_checkpoint):
        print(f"\n📥 Loading MAE checkpoint from: {mae_checkpoint}")
        try:
            checkpoint = torch.load(mae_checkpoint, map_location=device, weights_only=False)
        except ModuleNotFoundError as e:
            print(f"⚠️  ModuleNotFoundError: {e}")
            print("   Attempting to load with pickle compatibility mode...")
            
            # Add current directory to sys.path temporarily
            checkpoint_dir = os.path.dirname(mae_checkpoint)
            if checkpoint_dir and checkpoint_dir not in sys.path:
                sys.path.insert(0, checkpoint_dir)
            
            # Try again
            try:
                checkpoint = torch.load(mae_checkpoint, map_location=device, weights_only=False)
            except ModuleNotFoundError:
                # If still fails, try with weights_only=True (only load state dict)
                print("   Attempting weights_only mode...")
                checkpoint = torch.load(mae_checkpoint, map_location=device, weights_only=True)
        
        model.mae.load_state_dict(checkpoint['model'])
        print(f"✅ Loaded pretrained MAE from {mae_checkpoint}")
        print(f"   MAE correlation: {checkpoint['correlation']:.4f}")
        print(f"   MAE epoch: {checkpoint['epoch']}\n")
        
        # Freeze MAE initially
        for p in model.mae.parameters():
            p.requires_grad = False
        print("🔒 MAE encoder frozen (will unfreeze at epoch 50)\n")
    else:
        print(f"⚠️  MAE checkpoint not found: {mae_checkpoint}")
        print("   Training without pretrained weights\n")
    
    # Diffusion parameters
    T = 1000
    betas = get_beta_schedule(T).to(device)
    diff_params = get_diffusion_params(betas)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    # Training state
    best_val_loss = float('inf')
    best_pcc = 0.0
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"🔧 Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M\n")
    
    # Training loop
    for epoch in range(num_epochs):
        # Unfreeze MAE at epoch 50
        # Find the unfreezing section (around line 540-560) and add memory optimizations:

        # Before unfreezing MAE (around epoch 50):
        if epoch == 50:
            print("\n" + "="*60)
            print("🔓 Unfreezing MAE for fine-tuning")
            print("="*60)
            
            # Clear cache before unfreezing
            torch.cuda.empty_cache()
            
            # Reduce batch size for fine-tuning phase
            if 'batch_size' in locals():
                new_batch_size = max(1, batch_size // 2)  # Halve batch size
                print(f"Reducing batch size from {batch_size} to {new_batch_size}")
                
                # Recreate dataloaders with smaller batch size
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=new_batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=new_batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )
            
            # Unfreeze MAE
            for param in model.mae.parameters():
                param.requires_grad = True
            
            # Enable gradient checkpointing if available
            if hasattr(model.mae, 'gradient_checkpointing_enable'):
                model.mae.gradient_checkpointing_enable()
            
            # Recreate optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,  # Lower LR for fine-tuning
                weight_decay=0.01
            )
            
            # Clear cache again
            torch.cuda.empty_cache()
            print(f"GPU memory after unfreezing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x_lr, y_hr in pbar:
            x_lr, y_hr = x_lr.to(device), y_hr.to(device)
            B = x_lr.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', dtype=torch.float16):
                # Encode 62-channel SR to latent (via 31-channel MAE)
                z0 = model.encode_sr_to_latent(y_hr)  # (B, 500, 1024)
                
                # Debug: Print latent statistics in first epoch
                if epoch == 0 and pbar.n == 0:
                    print(f"\n  Debug - Latent stats: mean={z0.mean().item():.4f}, std={z0.std().item():.4f}, min={z0.min().item():.4f}, max={z0.max().item():.4f}")
                
                # Sample timestep and noise
                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)
                
                # Forward diffusion
                sqrt_alpha = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon
                
                # Get LR channel positions (16 channels)
                lr_pos = get_channel_positions(16, device, B)
                
                # Predict noise
                pred_epsilon = model(x_lr, zt, t, lr_pos)
                
                # Loss
                loss = criterion(pred_epsilon, epsilon)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x_lr, y_hr in val_loader:
                x_lr, y_hr = x_lr.to(device), y_hr.to(device)
                B = x_lr.size(0)
                
                z0 = model.encode_sr_to_latent(y_hr)
                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)
                
                sqrt_alpha = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon
                
                lr_pos = get_channel_positions(16, device, B)
                pred_epsilon = model(x_lr, zt, t, lr_pos)
                
                val_loss += criterion(pred_epsilon, epsilon).item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Sanity checks
        if epoch > 10:
            if train_loss < 0.01:
                print("  ⚠️  Loss suspiciously low - possible collapse!")
            elif train_loss > 2.0:
                print("  ⚠️  Loss very high - check gradients!")
        
        # Reconstruction metrics every 30 epochs (as requested)
        if (epoch + 1) % 30 == 0:
            metrics = validate_reconstruction(model, val_loader, diff_params, device)
            print(f"  📊 Metrics (Epoch {epoch+1}):")
            print(f"     PCC: {metrics['PCC']:.4f}")
            print(f"     RMSE: {np.sqrt(metrics['NMSE']):.4f}")
            print(f"     SNR: {metrics['SNR']:.2f} dB")
            print(f"     MAE: {metrics['MAE']:.4f}")
            
            if metrics['PCC'] > best_pcc:
                best_pcc = metrics['PCC']
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'diff_params': diff_params,
            'config': {
                'lr_channels': 16,
                'hr_channels': 31,
                'sr_channels': 62,
                'seq_len': 4000,
                'latent_dim': 1024
            }
        }
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'best_stad_seed.pt')
            print(f"  ✅ Saved best model (val_loss={val_loss:.6f})")
        
        # Regular checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, f'checkpoint_seed_epoch_{epoch+1}.pt')
            print(f"  💾 Saved checkpoint")
    
    print("\n" + "="*60)
    print("🎉 Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best PCC: {best_pcc:.3f}")
    print("="*60)

if __name__ == '__main__':
    train_stad_seed(
        data_path='/home/ab_students/EEG-MTP/DATA/SEED_processed',
        num_epochs=300,
        batch_size=16,  # Smaller batch for 62 channels
        lr=2e-4
    )
