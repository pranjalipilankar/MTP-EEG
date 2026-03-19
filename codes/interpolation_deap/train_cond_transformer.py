#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from encoder import EEGEncoder
from decoder import EEGDecoder
from cond_diffusion_transformer import SpatioTemporalConditionedTransformer
from channel_ops import downsample_channels_keep_odd, upsample_channels_linear_even
from spatio_temporal_condition import SpatioTemporalConditionNet
from graph_harmonic_condition import GraphHarmonicConditionNet
from fourier_positional_condition import FourierPositionConditionNet

def get_32_channel_positions(device="cpu", batch_size=1):
    # Approximate 10-20 system coordinates, normalized to [-1, 1]
    chan_pos_np = np.array([
        [-0.5,  1.0],  # Fp1
        [-0.3,  0.95], # AF3
        [-0.4,  0.8],  # F3
        [-0.7,  0.7],  # F7
        [-0.6,  0.5],  # FC5
        [-0.4,  0.55], # FC1
        [-0.3,  0.3],  # C3
        [-0.6,  0.0],  # T7
        [-0.5, -0.2],  # CP5
        [-0.3,  0.0],  # CP1
        [-0.2, -0.3],  # P3
        [-0.5, -0.5],  # P7
        [-0.3, -0.7],  # PO3
        [-0.1, -0.9],  # O1
        [ 0.0, -1.0],  # Oz
        [ 0.0, -0.5],  # Pz
        [ 0.5,  1.0],  # Fp2
        [ 0.3,  0.95], # AF4
        [ 0.2,  0.8],  # Fz
        [ 0.4,  0.8],  # F4
        [ 0.7,  0.7],  # F8
        [ 0.6,  0.5],  # FC6
        [ 0.4,  0.55], # FC2
        [ 0.0,  0.0],  # Cz
        [ 0.3,  0.3],  # C4
        [ 0.6,  0.0],  # T8
        [ 0.5, -0.2],  # CP6
        [ 0.2,  0.0],  # CP2
        [ 0.2, -0.3],  # P4
        [ 0.5, -0.5],  # P8
        [ 0.3, -0.7],  # PO4
        [ 0.1, -0.9],  # O2
    ], dtype=np.float32)

    # convert to torch tensor and expand to batch
    chan_pos = torch.tensor(chan_pos_np, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    return chan_pos

# ------------------ Utilities ------------------
def bandpass_filter(data, low=1.0, high=40.0, fs=200.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)

def upsample_latents_to_channels(z, target_channels=32):
    if z.dim() != 3:
        raise ValueError("z must be (B, N, D)")
    B, N, D = z.shape
    if N == target_channels:
        return z
    z_t = z.permute(0, 2, 1)
    z_up = F.interpolate(z_t, size=target_channels, mode='linear', align_corners=True)
    return z_up.permute(0, 2, 1)

# -------------------------- Dataset --------------------------
class EEGDataset(Dataset):
    def __init__(self, npz_path, split="train", window_size=400, fs=128, mode="denoise"):
        self.mode = mode
        self.window_size = window_size
        self.fs = fs

        data = np.load(npz_path)
        X = data[f"X_{split}"]  # shape: (n_trials, 32, time)
        self.n_trials, self.n_channels, self.n_points = X.shape
        print(f"✅ Loaded {split} dataset: {X.shape}")

        # Filter + segment + normalize
        self.samples = self.prepare_segments(X)

    def prepare_segments(self, X):
        eeg_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            eeg_filtered[i] = bandpass_filter(X[i], fs=self.fs)

        segments = []
        for trial in eeg_filtered:
            for start in range(0, trial.shape[-1] - self.window_size + 1, self.window_size):
                seg = trial[:, start:start+self.window_size]
                segments.append(seg)

        Xn = np.stack(segments)
        # Normalize per channel
        for ch in range(Xn.shape[1]):
            mean = Xn[:, ch, :].mean(axis=1, keepdims=True)
            std = Xn[:, ch, :].std(axis=1, keepdims=True) + 1e-6
            Xn[:, ch, :] = (Xn[:, ch, :] - mean) / std
        return Xn.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg = torch.tensor(self.samples[idx], dtype=torch.float32)  # (32, T)

        # Downsample → interpolate (simulate noisy input)
        seg_16 = downsample_channels_keep_odd(seg)   # (16, T)
        seg_interp = upsample_channels_linear_even(seg_16)  # (32, T)
        noise = seg_interp - seg                     # noise to learn

        inp = seg_interp
        target = noise  # Model learns to predict noise
        return inp, target

# ------------------ Safe DataLoader ------------------
def safe_dataloader(dataset, batch_size, num_workers=4, drop_last=True):
    while batch_size > 0:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=drop_last)
            return loader, batch_size
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"⚠️ OOM at batch_size={batch_size}, halving it.")
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise e
    raise RuntimeError("GPU memory insufficient even for batch_size=1.")

def train_model(
    dataset_path="/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz",
    mode="denoise",
    num_epochs=10,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-4,
    max_lr=3e-4,
    patience=6,
    save_path="eeg_conditioned_model_diff_denoise.pt"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device} | Mode: {mode}")

    # --- Datasets ---
    train_dataset = EEGDataset(dataset_path, split="train", window_size=400, fs=128)
    val_dataset   = EEGDataset(dataset_path, split="val", window_size=400, fs=128)
    train_loader, batch_size = safe_dataloader(train_dataset, batch_size=batch_size)
    val_loader, _ = safe_dataloader(val_dataset, batch_size=batch_size)
    print(f"✅ Using batch_size={batch_size} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Model ---
    n_time = 400
    latent_dim = 128
    encoder = EEGEncoder(
        in_channels=32, latent_dim=latent_dim, seq_len=n_time,
        n_layers=4, n_heads=8, dropout=0.1, pool_factor=4, checkpoint_segments=4
    ).to(device)

    decoder = EEGDecoder(latent_dim=latent_dim, out_channels=32, seq_len=n_time).to(device)
    transformer = SpatioTemporalConditionedTransformer(
        latent_dim=latent_dim, n_channels=32, n_layers=4, n_heads=8
    ).to(device)

    #condition_net = FourierPositionConditionNet(n_channels=32, model_dim=latent_dim).to(device)
    #condition_net = SpatioTemporalConditionNet(n_channels=32, model_dim=latent_dim, n_conditions=8).to(device)
    condition_net = GraphHarmonicConditionNet(n_channels=32, model_dim=latent_dim, n_conditions=8).to(device)

    # --- Optimizer ---
    params = list(encoder.parameters()) + list(transformer.parameters()) + \
             list(decoder.parameters()) + list(condition_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader),
        epochs=num_epochs, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(enabled=(device != "cpu"))

    best_val_loss, patience_counter = float("inf"), 0

    # --- Training Loop ---
    for epoch in range(num_epochs):
        encoder.train(); transformer.train(); decoder.train(); condition_net.train()
        train_loss = 0.0
        for inp, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
                z, _ = encoder(inp)
                z = upsample_latents_to_channels(z, target_channels=32)
                B = z.size(0)
                t_steps = torch.randint(0, 1000, (B,), device=device)
                chan_pos = get_32_channel_positions(device=device, batch_size=B)
                cond_tokens, cond_pooled = condition_net(inp, chan_pos, t_steps, torch.zeros(B, dtype=torch.long, device=device))
                z_refined = transformer(z, cond_tokens, cond_pooled, t_steps)
                pred_noise = decoder(z_refined)
                loss = criterion(pred_noise, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        print(f"📉 Epoch {epoch+1} | Train Loss: {train_loss:.6f}")

        # --- Validation ---
        encoder.eval(); transformer.eval(); decoder.eval(); condition_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp, target = inp.to(device), target.to(device)
                z, _ = encoder(inp)
                z = upsample_latents_to_channels(z, target_channels=32)
                B = z.size(0)
                t_steps = torch.randint(0, 1000, (B,), device=device)
                chan_pos = get_32_channel_positions(device=device, batch_size=B)
                cond_tokens, cond_pooled = condition_net(inp, chan_pos, t_steps, torch.zeros(B, dtype=torch.long, device=device))
                z_refined = transformer(z, cond_tokens, cond_pooled, t_steps)
                pred_noise = decoder(z_refined)
                loss = criterion(pred_noise, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"🔍 Validation Loss: {val_loss:.6f}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "transformer": transformer.state_dict(),
                "decoder": decoder.state_dict(),
                "condition_net": condition_net.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss
            }, save_path)
            print(f"✅ Saved model at epoch {epoch+1} → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    print(f"🎉 Training complete. Best loss = {best_val_loss:.6f}")

# ------------------ Main ------------------
if __name__ == "__main__":
    train_model(
        dataset_path="/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz",
        mode="denoise",
        num_epochs=20,
        batch_size=4,
        lr=1e-4,
        weight_decay=1e-4,
        max_lr=3e-4,
        patience=6,
        save_path="harmonic_model_denoise_pcc.pt"
    )
