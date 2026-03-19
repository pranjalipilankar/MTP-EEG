#!/usr/bin/env python3
"""
FIXED Evaluation Script for DreamDiffusion-Style MAE
Key fix: Correct unpatchify implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ============================================================================
# Fixed MAE Architecture
# ============================================================================

def bandpass_filter(data, low=1.0, high=40.0, fs=128.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)

class PatchEmbed1D(nn.Module):
    def __init__(self, time_len=400, patch_size=4, in_chans=32, embed_dim=256):
        super().__init__()
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = time_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight.view([embed_dim, -1]))

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x

class MAEforEEG(nn.Module):
    def __init__(self, time_len=400, patch_size=4, embed_dim=256, in_chans=32, 
                 depth=6, num_heads=8, decoder_embed_dim=128, decoder_depth=4):
        super().__init__()
        
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, 
                                      batch_first=True, dropout=0.1)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(decoder_embed_dim, num_heads//2, decoder_embed_dim*4, 
                                      batch_first=True, dropout=0.1)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.time_len = time_len

    def random_masking(self, x, mask_ratio):
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
        p = self.patch_size
        B, C, T = eeg.shape
        x = eeg.transpose(1, 2).reshape(B, T//p, C*p)
        return x
    
    def unpatchify(self, x):
        """
        ✅ FIXED: Correct inverse of patchify
        """
        p = self.patch_size
        c = self.in_chans
        B, N, _ = x.shape
        
        # Reshape: (B, N, c*p) → (B, N*p, c)
        eeg = x.reshape(B, N * p, c)
        
        # Transpose: (B, N*p, c) → (B, c, N*p)
        eeg = eeg.transpose(1, 2)
        
        return eeg

    def forward_loss(self, eeg, pred, mask):
        target = self.patchify(eeg)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, eeg, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(eeg, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(eeg, pred, mask)
        return loss, pred, mask
    
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
        """Decode full sequence"""
        x = self.decoder_embed(x_enc)
        x = x + self.decoder_pos_embed
        
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:]  # Remove CLS
        
        return x
    
    def reconstruct(self, eeg):
        """
        ✅ FIXED: Full reconstruction pipeline
        """
        latent = self.encode_no_mask(eeg)
        pred_patches = self.decode_full(latent)
        reconstructed = self.unpatchify(pred_patches)
        return reconstructed

# ============================================================================
# Dataset
# ============================================================================

class MAEDataset(Dataset):
    def __init__(self, npz_path, split="train", window_size=400, fs=128):
        data = np.load(npz_path)
        X = data[f"X_{split}"]
        
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
        
        for ch in range(Xn.shape[1]):
            mean = Xn[:, ch, :].mean(axis=1, keepdims=True)
            std = Xn[:, ch, :].std(axis=1, keepdims=True) + 1e-6
            Xn[:, ch, :] = (Xn[:, ch, :] - mean) / std
        return Xn.astype(np.float32)

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        return torch.FloatTensor(self.samples[idx])

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_mae_reconstruction(checkpoint_path, dataset_path, device='cuda'):
    print("="*80)
    print("🔬 EVALUATING DREAMDIFFUSION MAE (FIXED UNPATCHIFY)")
    print("="*80)
    
    # Load model
    model = MAEforEEG(
        time_len=400,
        patch_size=4,
        embed_dim=256,
        in_chans=32,
        depth=6,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    # Load data
    print("\n📦 Loading validation dataset...")
    val_dataset = MAEDataset(dataset_path, split='val', window_size=400)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Metrics
    all_pcc = []
    all_snr = []
    all_nmse = []
    all_mse = []
    
    print("\n🔄 Computing reconstruction metrics...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 20:
                break
            
            batch = batch.to(device)
            
            # ✅ Use FIXED reconstruct method
            recon = model.reconstruct(batch)
            
            batch_np = batch.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            # Channel-wise PCC
            B, C, T = batch_np.shape
            for b in range(B):
                for ch in range(C):
                    gt = batch_np[b, ch]
                    pred = recon_np[b, ch]
                    
                    if np.std(gt) > 1e-6 and np.std(pred) > 1e-6:
                        pcc, _ = pearsonr(gt, pred)
                        if not np.isnan(pcc):
                            all_pcc.append(pcc)
            
            # Global metrics
            mse = np.mean((recon_np - batch_np) ** 2)
            signal_power = np.mean(batch_np ** 2)
            nmse = mse / (signal_power + 1e-10)
            snr = 10 * np.log10((signal_power + 1e-10) / (mse + 1e-10))
            
            all_nmse.append(nmse)
            all_snr.append(snr)
            all_mse.append(mse)
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/20...")
    
    # Results
    print("\n" + "="*80)
    print("📊 RECONSTRUCTION QUALITY REPORT")
    print("="*80)
    
    mean_pcc = np.mean(all_pcc)
    std_pcc = np.std(all_pcc)
    mean_snr = np.mean(all_snr)
    mean_nmse = np.mean(all_nmse)
    mean_mse = np.mean(all_mse)
    
    print(f"\n{'Metric':<20} {'Value':<20} {'Target':<15} {'Status'}")
    print("-" * 80)
    print(f"{'PCC (mean±std)':<20} {mean_pcc:.4f} ± {std_pcc:.4f}    {'>0.80':<15} {'✅' if mean_pcc > 0.80 else '⚠️' if mean_pcc > 0.60 else '❌'}")
    print(f"{'SNR (dB)':<20} {mean_snr:.2f}              {'>15.0':<15} {'✅' if mean_snr > 15 else '⚠️' if mean_snr > 10 else '❌'}")
    print(f"{'NMSE':<20} {mean_nmse:.4f}            {'<0.05':<15} {'✅' if mean_nmse < 0.05 else '⚠️' if mean_nmse < 0.10 else '❌'}")
    print(f"{'MSE':<20} {mean_mse:.6f}          {'<0.01':<15} {'✅' if mean_mse < 0.01 else '⚠️' if mean_mse < 0.05 else '❌'}")
    
    print(f"\nPCC Distribution:")
    print(f"  Min:  {np.min(all_pcc):.4f}")
    print(f"  25%:  {np.percentile(all_pcc, 25):.4f}")
    print(f"  50%:  {np.percentile(all_pcc, 50):.4f}")
    print(f"  75%:  {np.percentile(all_pcc, 75):.4f}")
    print(f"  Max:  {np.max(all_pcc):.4f}")
    
    # Assessment
    print("\n" + "="*80)
    if mean_pcc > 0.80 and mean_snr > 15:
        print("✅ MAE QUALITY: EXCELLENT - Ready for STAD!")
        recommendation = "proceed"
    elif mean_pcc > 0.60 and mean_snr > 10:
        print("⚠️  MAE QUALITY: ACCEPTABLE - Can proceed with caution")
        recommendation = "monitor"
    else:
        print("❌ MAE QUALITY: NEEDS IMPROVEMENT")
        recommendation = "retrain"
    print("="*80)
    
    # Visualization
    print("\n📈 Generating visualization...")
    
    batch = next(iter(val_loader)).to(device)
    with torch.no_grad():
        recon = model.reconstruct(batch)
    
    batch_np = batch[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(8):
        axes[i].plot(batch_np[i], label='Ground Truth', alpha=0.8, linewidth=1.5, color='blue')
        axes[i].plot(recon_np[i], label='Reconstruction', alpha=0.8, linewidth=1.5, color='red', linestyle='--')
        
        pcc, _ = pearsonr(batch_np[i], recon_np[i])
        mse_ch = np.mean((batch_np[i] - recon_np[i]) ** 2)
        
        axes[i].set_title(f'Ch{i+1} | PCC={pcc:.3f}, MSE={mse_ch:.4f}', fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
    
    plt.suptitle(f'MAE Reconstruction (PCC={mean_pcc:.3f}, SNR={mean_snr:.1f}dB)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mae_eval_FIXED.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: mae_eval_FIXED.png")
    
    return {
        'pcc': mean_pcc,
        'snr': mean_snr,
        'nmse': mean_nmse,
        'recommendation': recommendation
    }

if __name__ == '__main__':
    metrics = evaluate_mae_reconstruction(
        checkpoint_path='mae_deap_pretrain_depth4_800.pt',
        dataset_path='/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*80)
    print("🎯 FINAL VERDICT")
    print("="*80)
    
    if metrics['recommendation'] == 'proceed':
        print("\n✅ Your MAE is READY for STAD training!")
        print("\n📋 Next steps:")
        print("  1. Use this MAE for STAD spatial upsampling")
        print("  2. Expected STAD reconstruction PCC: 0.6-0.8")
        print("  3. Monitor training carefully")
    elif metrics['recommendation'] == 'monitor':
        print("\n⚠️  MAE is acceptable but not optimal")
        print("\n  Option A: Proceed with STAD (expect slower convergence)")
        print("  Option B: Retrain MAE with decoder_depth=4-6")
    else:
        print("\n❌ MAE needs more training")
        print("\n  Action: Train 200-300 more epochs")
    
    print("="*80)