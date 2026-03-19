#!/usr/bin/env python3
"""
Standalone script to pre-train the Masked Autoencoder (MAE)
Run this BEFORE training the full STAD model

Based on STAD paper Section III.C.1:
"We employ a Masked Autoencoder (MAE) for asymmetric latent space 
representation of HR EEG"
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import butter, filtfilt


# -------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------
def bandpass_filter(data, low=1.0, high=40.0, fs=128.0, order=4):
    """Bandpass filter"""
    nyquist = 0.5 * fs
    low_norm = np.clip(low / nyquist, 0.01, 0.99)
    high_norm = np.clip(high / nyquist, low_norm + 0.01, 0.99)
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data, axis=-1)

class PatchEmbedding(nn.Module):
    """Divide EEG into patches and embed them"""
    def __init__(self, n_channels=32, patch_size=16, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        
        # Project each patch to embedding space
        self.projection = nn.Linear(patch_size, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T) - EEG signal
        Returns:
            patches: (B, C, n_patches, embed_dim)
        """
        B, C, T = x.shape
        n_patches = T // self.patch_size
        
        # Reshape into patches: (B, C, n_patches, patch_size)
        x = x[:, :, :n_patches * self.patch_size]
        x = x.reshape(B, C, n_patches, self.patch_size)
        
        # Embed each patch
        patches = self.projection(x)  # (B, C, n_patches, embed_dim)
        return patches


class MaskedAutoEncoder(nn.Module):
    """
    Masked Autoencoder for EEG pre-training
    Based on STAD paper: "we employ a Masked Autoencoder (MAE) for 
    asymmetric latent space representation of HR EEG"
    """
    def __init__(
        self,
        n_channels=32,
        seq_len=350,  # 350ms as per paper
        patch_size=16,
        embed_dim=128,
        encoder_depth=6,
        decoder_depth=4,
        n_heads=8,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(n_channels, patch_size, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_channels, self.n_patches, embed_dim) * 0.02
        )
        
        # Channel embedding (spatial information)
        self.channel_embed = nn.Parameter(
            torch.randn(1, n_channels, 1, embed_dim) * 0.02
        )
        
        # Encoder (asymmetric - smaller)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # Decoder (larger for reconstruction)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
        
        # Reconstruction head
        self.head = nn.Linear(embed_dim, patch_size)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def random_masking(self, x):
        """
        Random masking per channel per sample
        Args:
            x: (B, C, n_patches, D)
        Returns:
            x_masked: (B, C, n_visible, D)
            mask: (B, C, n_patches) - 0 is keep, 1 is remove
            ids_restore: for unshuffling
        """
        B, C, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, C, N, device=x.device)
        
        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=2)
        ids_restore = torch.argsort(ids_shuffle, dim=2)
        
        # Keep first n_keep patches
        ids_keep = ids_shuffle[:, :, :n_keep]
        
        # Gather visible patches
        x_masked = torch.gather(
            x, dim=2, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, C, N], device=x.device)
        mask[:, :, :n_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        """
        Args:
            x: (B, C, T) - EEG signal
        Returns:
            latent: (B, C, n_visible, D)
            mask: (B, C, n_patches)
            ids_restore: for reconstruction
        """
        # Patchify and embed
        x = self.patch_embed(x)  # (B, C, n_patches, D)
        
        # Add positional and channel embeddings
        x = x + self.pos_embed + self.channel_embed
        
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x)
        
        # Flatten for transformer: (B, C*n_visible, D)
        B, C, n_visible, D = x_masked.shape
        x_flat = x_masked.reshape(B, C * n_visible, D)
        
        # Encode
        latent = self.encoder(x_flat)
        latent = self.norm(latent)
        
        # Reshape back
        latent = latent.reshape(B, C, n_visible, D)
        
        return latent, mask, ids_restore
    
    def forward_decoder(self, latent, ids_restore):
        """
        Args:
            latent: (B, C, n_visible, D)
            ids_restore: restore order
        Returns:
            pred: (B, C, n_patches, patch_size)
        """
        B, C, n_visible, D = latent.shape
        n_patches = self.n_patches
        
        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, C, n_patches - n_visible, D)
        latent_full = torch.cat([latent, mask_tokens], dim=2)  # (B, C, n_patches, D)
        
        # Unshuffle to original order
        latent_full = torch.gather(
            latent_full, dim=2,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        
        # Add positional embeddings
        latent_full = latent_full + self.pos_embed
        
        # Flatten for decoder
        latent_flat = latent_full.reshape(B, C * n_patches, D)
        
        # Decode (cross-attention with memory=latent_flat)
        # For MAE, we use self-attention in decoder
        decoded = latent_flat
        for layer in self.decoder.layers:
            decoded = layer(decoded, latent_flat)
        
        # Reshape and reconstruct
        decoded = decoded.reshape(B, C, n_patches, D)
        pred = self.head(decoded)  # (B, C, n_patches, patch_size)
        
        return pred
    
    def forward(self, x):
        """
        Training forward pass
        Args:
            x: (B, C, T)
        Returns:
            pred: (B, C, T) - reconstructed signal
            mask: (B, C, n_patches) - which patches were masked
        """
        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(x)
        
        # Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        # Reshape predictions to match input
        B, C, n_patches, patch_size = pred.shape
        pred = pred.reshape(B, C, n_patches * patch_size)
        
        return pred, mask
    
    def encode(self, x):
        """
        Inference: encode without masking
        Args:
            x: (B, C, T)
        Returns:
            latent: (B, C, D) - compressed representation
        """
        # Patchify
        x = self.patch_embed(x)  # (B, C, n_patches, D)
        x = x + self.pos_embed + self.channel_embed
        
        # Encode all patches (no masking)
        B, C, n_patches, D = x.shape
        x_flat = x.reshape(B, C * n_patches, D)
        latent = self.encoder(x_flat)
        latent = self.norm(latent)
        
        # Pool across patches to get per-channel latent
        latent = latent.reshape(B, C, n_patches, D)
        latent = latent.mean(dim=2)  # (B, C, D)
        
        return latent
    
    def decode_from_latent(self, z):
        """
        Decode from latent representation (for inference)
        Args:
            z: (B, C, D) - latent vectors
        Returns:
            x_recon: (B, C, T) - reconstructed signal
        """
        B, C, D = z.shape
        
        # Expand to all patches
        z = z.unsqueeze(2).expand(-1, -1, self.n_patches, -1)  # (B, C, n_patches, D)
        z = z + self.pos_embed
        
        # Flatten and decode
        z_flat = z.reshape(B, C * self.n_patches, D)
        decoded = z_flat
        for layer in self.decoder.layers:
            decoded = layer(decoded, z_flat)
        
        # Reconstruct
        decoded = decoded.reshape(B, C, self.n_patches, D)
        pred = self.head(decoded)  # (B, C, n_patches, patch_size)
        
        # Flatten to time series
        pred = pred.reshape(B, C, self.n_patches * self.patch_size)
        
        return pred

# -------------------------------------------------------------
# Dataset for MAE Pre-training (HR EEG only)
# -------------------------------------------------------------
class HREEGDataset(Dataset):
    """
    Dataset containing only HR (high-resolution) EEG.
    MAE learns to reconstruct from masked patches.
    """
    def __init__(self, npz_path, split='train', hr_channels=32, window_size=350, fs=128):
        self.hr_channels = hr_channels
        self.window_size = window_size
        self.fs = fs
        
        # Load data
        data = np.load(npz_path)
        X = data[f"X_{split}"]  # (n_trials, channels, time)
        
        # Prepare HR EEG
        self.samples = self.prepare_segments(X, hr_channels)
        print(f"✅ Loaded {split} HR EEG: {self.samples.shape}")
    
    def prepare_segments(self, X, target_channels):
        """Filter, segment, normalize"""
        n_trials, n_channels, n_points = X.shape
        
        # Downsample channels if needed (to simulate HR target)
        if n_channels > target_channels:
            indices = np.linspace(0, n_channels-1, target_channels, dtype=int)
            X = X[:, indices, :]
        elif n_channels < target_channels:
            # If you have fewer channels than target, you can't pre-train MAE properly
            print(f"⚠️ WARNING: Dataset has {n_channels} channels, need {target_channels}")
            print(f"⚠️ Using all {n_channels} channels instead")
            target_channels = n_channels
        
        # Filter
        X_filtered = np.array([bandpass_filter(trial, fs=self.fs) for trial in X])
        
        # Segment
        segments = []
        for trial in X_filtered:
            for start in range(0, trial.shape[-1] - self.window_size + 1, self.window_size):
                segments.append(trial[:, start:start + self.window_size])
        
        X_seg = np.stack(segments)
        
        # Normalize per channel
        for ch in range(X_seg.shape[1]):
            mean = X_seg[:, ch, :].mean(axis=1, keepdims=True)
            std = X_seg[:, ch, :].std(axis=1, keepdims=True) + 1e-6
            X_seg[:, ch, :] = (X_seg[:, ch, :] - mean) / std
        
        return X_seg.astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# -------------------------------------------------------------
# MAE Pre-training Function
# -------------------------------------------------------------
def pretrain_mae_standalone(
    dataset_path,
    hr_channels=32,
    window_size=350,
    num_epochs=500,
    batch_size=32,
    lr=1e-4,
    save_path='pretrained_mae.pt',
    device='cuda'
):
    """
    Pre-train the Masked Autoencoder on HR EEG data.
    
    This learns robust latent representations by:
    1. Masking 75% of EEG patches
    2. Reconstructing the masked portions
    3. Learning spatio-temporal features
    
    Args:
        dataset_path: Path to .npz file
        hr_channels: Number of HR channels (256 in paper)
        window_size: Temporal window (350ms in paper)
        num_epochs: Training epochs (500 recommended)
        batch_size: Batch size
        lr: Learning rate
        save_path: Where to save pre-trained weights
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # === 1. Load Data ===
    print("📂 Loading HR EEG data...")
    train_dataset = HREEGDataset(
        dataset_path, split='train',
        hr_channels=hr_channels,
        window_size=window_size
    )
    val_dataset = HREEGDataset(
        dataset_path, split='val',
        hr_channels=hr_channels,
        window_size=window_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4
    )
    
    # === 2. Initialize MAE ===
    print("🏗️  Initializing Masked Autoencoder...")
    mae = MaskedAutoEncoder(
        n_channels=hr_channels,
        seq_len=window_size,
        patch_size=16,
        embed_dim=128,
        encoder_depth=6,
        decoder_depth=4,
        n_heads=8,
        mask_ratio=0.75,  # Mask 75% of patches
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    print(f"📊 Model has {n_params:,} trainable parameters")
    
    # === 3. Setup Training ===
    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    # === 4. Training Loop ===
    print(f"🚀 Starting MAE pre-training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # --- Training ---
        mae.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for hr_eeg in pbar:
            hr_eeg = hr_eeg.to(device)  # (B, C, T)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast('cuda', dtype=torch.float16):
                # Forward pass (with masking)
                pred, mask = mae(hr_eeg)
                
                # Compute loss only on MASKED patches
                B, C, n_patches = mask.shape
                patch_size = mae.patch_size
                
                # Reshape to patches
                hr_patches = hr_eeg[:, :, :n_patches * patch_size].reshape(
                    B, C, n_patches, patch_size
                )
                pred_patches = pred.reshape(B, C, n_patches, patch_size)
                
                # Mask for loss computation
                mask_expanded = mask.unsqueeze(-1).expand_as(hr_patches)
                
                # MSE loss on masked regions only
                loss = criterion(
                    pred_patches[mask_expanded == 1],
                    hr_patches[mask_expanded == 1]
                )
            
            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # --- Validation ---
        mae.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for hr_eeg in val_loader:
                hr_eeg = hr_eeg.to(device)
                
                pred, mask = mae(hr_eeg)
                
                B, C, n_patches = mask.shape
                patch_size = mae.patch_size
                
                hr_patches = hr_eeg[:, :, :n_patches * patch_size].reshape(
                    B, C, n_patches, patch_size
                )
                pred_patches = pred.reshape(B, C, n_patches, patch_size)
                mask_expanded = mask.unsqueeze(-1).expand_as(hr_patches)
                
                loss = criterion(
                    pred_patches[mask_expanded == 1],
                    hr_patches[mask_expanded == 1]
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # --- Logging ---
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")
        
        # --- Save Best Model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': mae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'n_channels': hr_channels,
                    'seq_len': window_size,
                    'patch_size': 16,
                    'embed_dim': 128,
                }
            }, save_path)
            print(f"✅ Saved best MAE model (val_loss={val_loss:.6f})")
    
    print("="*60)
    print(f"🎉 MAE pre-training complete!")
    print(f"📁 Best model saved to: {save_path}")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    
    return mae

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == '__main__':
    # Configuration
    DATASET_PATH = '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz'
    HR_CHANNELS = 32  # Target HR channels (adjust based on your data)
    WINDOW_SIZE = 350  # 350ms as per paper
    NUM_EPOCHS = 500   # Paper uses extensive pre-training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    SAVE_PATH = 'pretrained_mae_500.pt'
    DEVICE = 'cuda'
    
    # Run pre-training
    mae = pretrain_mae_standalone(
        dataset_path=DATASET_PATH,
        hr_channels=HR_CHANNELS,
        window_size=WINDOW_SIZE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        save_path=SAVE_PATH,
        device=DEVICE
    )
    
    print("\n✅ You can now use this pre-trained MAE in STAD training!")
    print(f"   Load it with: mae.load_state_dict(torch.load('{SAVE_PATH}')['model_state_dict'])")