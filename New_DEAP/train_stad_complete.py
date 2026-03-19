#!/usr/bin/env python3
"""
CRITICAL FIX: Dimension and Statistics Alignment
Main fixes:
1. STC embed_dim matches MAE latent_dim (256, not 128)
2. Proper latent normalization
3. Correct reconstruction with statistics matching
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from pretrain_dreamdiff_mae import MAEforEEG
from spatio_temporal_condition import SpatioTemporalConditionModule  
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule

def get_beta_schedule(timesteps=1000):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_diffusion_params(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
    }

def get_channel_positions(n_channels, device='cpu', batch_size=1):
    if n_channels == 32:
        positions = np.array([
            [-0.35, 0.93], [-0.71, 0.71], [-0.53, 0.84], [-0.88, 0.47],
            [-1.00, 0.00], [-0.88,-0.47], [-0.53,-0.84], [ 0.71,-0.71],
            [ 0.88, 0.47], [ 0.71, 0.71], [ 0.53, 0.84], [ 0.35, 0.93],
            [ 0.71,-0.71], [ 0.00, 0.00], [ 0.35,-0.93], [-0.35,-0.93],
            [ 0.00, 1.00], [-0.25, 0.25], [ 0.25, 0.25], [ 0.00,-1.00],
            [-0.50,-0.25], [ 0.50,-0.25], [-0.25,-0.50], [ 0.25,-0.50],
            [-0.50, 0.25], [ 0.50, 0.25], [-0.25, 0.50], [ 0.25, 0.50],
            [ 0.00, 0.50], [-0.75, 0.00], [ 0.75, 0.00], [ 0.00, 0.00]
        ], dtype=np.float32)
    elif n_channels == 16:
        positions = get_channel_positions(32, 'cpu', 1).squeeze(0).numpy()[::2]
    return torch.tensor(positions, device=device).unsqueeze(0).expand(batch_size, -1, -1)

def reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50):
    """FIXED reconstruction with proper dimensions"""
    model.eval()
    with torch.no_grad():
        B = x_lr.shape[0]
        lr_pos = get_channel_positions(16, device, B)
        
        # Start from noise (B, 100, 256)
        zt = torch.randn(B, 100, 256, device=device)
        
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
        
        # Decode
        cls_token = model.mae.cls_token.expand(B, -1, -1)
        zt_with_cls = torch.cat([cls_token, zt], dim=1)
        pred_patches = model.mae.decode_full(zt_with_cls)
        sr_eeg = model.mae.unpatchify(pred_patches)
        
        return sr_eeg

def validate_reconstruction(model, val_loader, diff_params, device):
    """Validate with proper metrics"""
    metrics = {'PCC': [], 'SNR': [], 'NMSE': [], 'MAE': []}
    
    with torch.no_grad():
        for i, (x_lr, y_hr) in enumerate(val_loader):
            if i >= 10:  # More samples for better stats
                break
                
            x_lr, y_hr = x_lr.to(device), y_hr.to(device)
            sr_eeg = reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50)
            
            sr_np = sr_eeg.cpu().numpy()
            hr_np = y_hr.cpu().numpy()
            
            # Align length
            min_len = min(sr_np.shape[2], hr_np.shape[2])
            sr_np = sr_np[:, :, :min_len]
            hr_np = hr_np[:, :, :min_len]
            
            # Channel-wise PCC (most important metric)
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

class STADDataset(Dataset):
    def __init__(self, npz_path, split='train', lr_channels=16, hr_channels=32, window_size=400, fs=128):
        self.lr_channels, self.hr_channels = lr_channels, hr_channels
        self.window_size, self.fs = window_size, fs
        
        data = np.load(npz_path)
        X = data[f"X_{split}"]
        
        self.hr_samples = self.prepare_segments(X, hr_channels)
        self.lr_samples = self.prepare_segments(X, lr_channels)
    
    def prepare_segments(self, X, target_channels):
        n_trials, n_channels, _ = X.shape
        indices = np.linspace(0, n_channels-1, target_channels, dtype=int)
        X_sub = X[:, indices, :]
        
        def bandpass_filter(data):
            nyquist = 0.5 * self.fs
            b, a = butter(4, [max(1.0/nyquist, 0.01), min(40.0/nyquist, 0.99)], 'band')
            return filtfilt(b, a, data, axis=-1)
        
        X_filtered = np.array([bandpass_filter(trial) for trial in X_sub])
        
        segments = []
        for trial in X_filtered:
            for start in range(0, trial.shape[-1] - self.window_size + 1, self.window_size):
                segments.append(trial[:, start:start + self.window_size])
        
        X_seg = np.stack(segments)
        for ch in range(X_seg.shape[1]):
            mean, std = X_seg[:, ch].mean(axis=1, keepdims=True), X_seg[:, ch].std(axis=1, keepdims=True) + 1e-6
            X_seg[:, ch] = (X_seg[:, ch] - mean) / std
        
        return X_seg.astype(np.float32)
    
    def __len__(self): 
        return len(self.hr_samples)
    
    def __getitem__(self, idx): 
        return torch.tensor(self.lr_samples[idx]), torch.tensor(self.hr_samples[idx])

# ✅ CRITICAL FIX: Proper dimension alignment
class STAD(nn.Module):
    def __init__(self, lr_channels=16, hr_channels=32, seq_len=400, latent_dim=256, n_harmonics=8):
        super().__init__()
        
        patch_size = 4
        num_patches = seq_len // patch_size  # 100
        
        # MAE (embed_dim=256)
        self.mae = MAEforEEG(
            time_len=seq_len,
            patch_size=patch_size,
            embed_dim=256,
            in_chans=hr_channels,
            depth=6,
            num_heads=8,
            decoder_embed_dim=128,
            decoder_depth=4
        )
        
        # ✅ FIX: STC with embed_dim=256 (not 128!)
        self.stc = SpatioTemporalConditionModule(
            lr_channels, 
            seq_len, 
            embed_dim=latent_dim,  # ✅ 256 to match MAE
            n_harmonics=n_harmonics, 
            patch_size=16,
            n_transformer_layers=4,
            n_heads=8
        )
        
        # MTD (latent_dim=256, num_patches=100)
        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches,
            latent_dim=latent_dim,
            n_layers=6,
            n_heads=16
        )
        
        self.latent_dim = latent_dim
        self.num_patches = num_patches
    
    def encode_hr(self, hr_eeg):
        """Encode HR to latent (unnormalized for diffusion)"""
        latent = self.mae.encode_no_mask(hr_eeg)  # (B, 101, 256)
        latent = latent[:, 1:, :]  # Remove CLS → (B, 100, 256)
        
        # Don't normalize - diffusion models work better with unnormalized latents
        # The noise epsilon is sampled from N(0, 1) and should match the latent scale
        return latent
    
    def forward(self, lr_eeg, zt, t_steps, lr_chan_pos):
        cond_tokens, cond_pooled = self.stc(lr_eeg, lr_chan_pos, t_steps)
        return self.mtd(zt, t_steps, cond_tokens, cond_pooled)

def train_stad_fixed(dataset_path, num_epochs=300, batch_size=32, lr=2e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 STAD Training (DIMENSION FIXED)")
    
    train_dataset = STADDataset(dataset_path, 'train', window_size=400)
    val_dataset = STADDataset(dataset_path, 'val', window_size=400)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # ✅ Model with correct dimensions
    model = STAD(
        lr_channels=16,
        hr_channels=32,
        seq_len=400,
        latent_dim=256,  # Matches MAE
        n_harmonics=8
    ).to(device)
    
    # Load pretrained MAE
    mae_checkpoint = 'mae_deap_FIXED.pt'
    if os.path.exists(mae_checkpoint):
        mae_state = torch.load(mae_checkpoint, map_location=device, weights_only=False)
        model.mae.load_state_dict(mae_state)
        print(f"✅ Loaded MAE from {mae_checkpoint}")
        
        # Freeze encoder initially
        for p in model.mae.parameters():
            p.requires_grad = False
        print("🔒 MAE frozen")
    
    # Diffusion
    T = 1000
    betas = get_beta_schedule(T).to(device)
    diff_params = get_diffusion_params(betas)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Unfreeze MAE after 50 epochs
        if epoch == 50:
            print("\n🔓 Unfreezing MAE for fine-tuning")
            for p in model.mae.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr / 10, 
                weight_decay=0.05
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - epoch)
        
        # Train
        model.train()
        train_loss = 0.0
        
        for x_lr, y_hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_lr, y_hr = x_lr.to(device), y_hr.to(device)
            B = x_lr.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', dtype=torch.float16):
                # ✅ Encode with normalization
                z0 = model.encode_hr(y_hr)  # (B, 100, 256), normalized
                
                # Sample timestep and noise
                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)  # N(0,1) noise
                
                # Forward diffusion
                sqrt_alpha = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon
                
                # Conditioning
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
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_lr, y_hr in val_loader:
                x_lr, y_hr = x_lr.to(device), y_hr.to(device)
                B = x_lr.size(0)
                
                z0 = model.encode_hr(y_hr)
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
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Check if loss is in expected range
        if epoch > 10:
            if train_loss < 0.01:
                print("⚠️  WARNING: Loss suspiciously low - possible collapse!")
            elif train_loss > 2.0:
                print("⚠️  WARNING: Loss very high - possible gradient issues!")
        
        # Reconstruction metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = validate_reconstruction(model, val_loader, diff_params, device)
            print(f"📊 RECON: PCC={metrics['PCC']:.3f}, SNR={metrics['SNR']:.1f}dB, NMSE={metrics['NMSE']:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(), 
                'epoch': epoch, 
                'val_loss': val_loss,
                'diff_params': diff_params
            }, 'best_stad_DIMENSION_FIXED2.pt')
            print(f"✅ SAVED (val={val_loss:.6f})")
    
    print(f"\n🎉 Training complete! Best: {best_val_loss:.6f}")

if __name__ == '__main__':
    train_stad_fixed(
        '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
        num_epochs=300,
        batch_size=32,
        lr=2e-4
    )