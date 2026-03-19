#!/usr/bin/env python3
"""
FIXED DreamDiffusion MAE - DEAP Optimized
Key fixes: unpatchify, inference methods, statistics matching, increased decoder depth
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def bandpass_filter(data, low=1.0, high=40.0, fs=128.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)

class PatchEmbed1D(nn.Module):
    """✅ DreamDiffusion EXACT 1D patch embedding"""
    def __init__(self, time_len=400, patch_size=4, in_chans=32, embed_dim=256):
        super().__init__()
        self.time_len = time_len
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = time_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight.view([embed_dim, -1]))

    def forward(self, x):
        # x: (B, C, T) → (B, L, embed_dim)
        x = self.proj(x).transpose(1, 2)
        return x

class MAEforEEG(nn.Module):
    """✅ Enhanced DreamDiffusion MAE with FIXED unpatchify and inference methods"""
    def __init__(self, time_len=400, patch_size=4, embed_dim=256, in_chans=32,
                 depth=6, num_heads=8, decoder_embed_dim=128, decoder_depth=8):
        super().__init__()
        
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.time_len = time_len
        
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, 
                                      batch_first=True, dropout=0.1)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(decoder_embed_dim, max(1, num_heads//2), decoder_embed_dim*4,
                                      batch_first=True, dropout=0.1)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)

    def random_masking(self, x, mask_ratio=0.75):
        """Enhanced masking strategy"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x_enc, ids_restore):
        x = self.decoder_embed(x_enc)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:]
        return x

    def patchify(self, eeg):
        """Convert (B, C, T) → (B, L, C*P)"""
        p = self.patch_size
        B, C, T = eeg.shape
        assert T % p == 0
        eeg = eeg.transpose(1, 2).reshape(B, T//p, C*p)
        return eeg
    
    # ✅ FIX 1: CORRECT unpatchify
    def unpatchify(self, x):
        """✅ FIXED: (B, L, C*P) → (B, C, T)"""
        p = self.patch_size
        c = self.in_chans
        B, N, _ = x.shape
        eeg = x.reshape(B, N * p, c)  # (B, T, C)
        eeg = eeg.transpose(1, 2)     # (B, C, T)
        return eeg

    def forward_loss(self, eeg, pred, mask):
        """Weighted reconstruction loss on masked regions"""
        target = self.patchify(eeg)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward(self, eeg, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(eeg, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(eeg, pred, mask)
        return loss, pred, mask
    
    # ✅ FIX 2: ADD inference methods
    def encode_no_mask(self, x):
        """Encode without masking"""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:]
        cls_token = self.cls_token + self.pos_embed[:, :1]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
    
    def decode_full(self, x_enc):
        """Decode full sequence (no masking)"""
        x = self.decoder_embed(x_enc)
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:]
        return x
    
    # ✅ FIX 3: Statistics matching
    def reconstruct(self, eeg):
        """Full reconstruction with statistics normalization"""
        latent = self.encode_no_mask(eeg)
        pred_patches = self.decode_full(latent)
        reconstructed = self.unpatchify(pred_patches)
        
        # Match statistics to input
        input_mean = eeg.mean(dim=2, keepdim=True)
        input_std = eeg.std(dim=2, keepdim=True) + 1e-6
        
        recon_mean = reconstructed.mean(dim=2, keepdim=True)
        recon_std = reconstructed.std(dim=2, keepdim=True) + 1e-6
        
        reconstructed = (reconstructed - recon_mean) / recon_std * input_std + input_mean
        
        return reconstructed

class MAEDataset(Dataset):
    def __init__(self, npz_path, split="train", window_size=400, fs=128):
        data = np.load(npz_path)
        X = data[f"X_{split}"]
        print(f"✅ Loaded {split}: {X.shape}")
        
        self.window_size = window_size
        self.fs = fs
        self.samples = self.prepare_segments(X)

    def prepare_segments(self, X):
        eeg_filtered = np.array([bandpass_filter(x, fs=self.fs) for x in X])
        segments = []
        for trial in eeg_filtered:
            for start in range(0, trial.shape[-1] - self.window_size + 1, self.window_size):
                segments.append(trial[:, start:start+self.window_size])
        Xn = np.stack(segments)
        
        # Normalize per-channel
        for ch in range(Xn.shape[1]):
            mean = Xn[:, ch, :].mean(axis=1, keepdims=True)
            std = Xn[:, ch, :].std(axis=1, keepdims=True) + 1e-6
            Xn[:, ch, :] = (Xn[:, ch, :] - mean) / std
        return Xn.astype(np.float32)

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        return torch.FloatTensor(self.samples[idx])

def plot_reconstruction(model, dataset, device, num_samples=3):
    """✅ Visualization of reconstructions"""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 10))
    
    with torch.no_grad():
        for idx in range(num_samples):
            sample = next(iter(loader)).to(device)
            recon = model.reconstruct(sample)
            
            sample_np = sample[0, 0].cpu().numpy()
            recon_np = recon[0, 0].cpu().numpy()
            
            cor = np.corrcoef([sample_np, recon_np])[0, 1]
            
            axes[idx, 0].plot(sample_np)
            axes[idx, 0].set_title('Ground Truth')
            
            axes[idx, 1].plot(recon_np)
            axes[idx, 1].set_title(f'Reconstruction (r={cor:.3f})')
            
            axes[idx, 2].plot(sample_np - recon_np)
            axes[idx, 2].set_title('Error')
    
    plt.tight_layout()
    plt.savefig('mae_reconstruction.png')
    plt.close()

def pretrain_mae(dataset_path, epochs=500, patience=50, min_delta=0.001):
    """✅ Enhanced training loop"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 FIXED DreamDiffusion MAE Pretraining")
    
    model = MAEforEEG(
        time_len=400, 
        patch_size=4, 
        embed_dim=256, 
        in_chans=32,
        decoder_depth=8  # ✅ Increased from 2
    ).to(device)
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)
    
    train_ds = MAEDataset(dataset_path, "train")
    val_ds = MAEDataset(dataset_path, "val")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(batch, mask_ratio=0.75)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss, _, _ = model(batch, mask_ratio=0.75)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f} Val={avg_val:.4f} LR={lr:.2e}")
        
        if avg_val < best_loss - min_delta:
            best_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, "mae_deap_FIXED.pt")
            patience_counter = 0
            print(f"🎉 NEW BEST! Val={best_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"✅ CONVERGED at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            plot_reconstruction(model, val_ds, device)
    
    print("🎯 Pretraining COMPLETE!")

if __name__ == "__main__":
    pretrain_mae("/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz")