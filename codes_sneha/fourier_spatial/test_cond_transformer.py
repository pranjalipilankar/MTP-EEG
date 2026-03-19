#!/usr/bin/env python3
"""
test_cond_transformer.py
Evaluate trained Encoder + SpatioTemporalConditionedTransformer + Decoder
with real spatial & temporal conditioning.

- denoise: test with 31→62 interpolated input (inp is 62-ch interpolated)
- completion: test with 31→62 direct completion (inp is 31-ch)
Saves metrics, arrays, per-channel CSVs and a 10-channel grid.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import r2_score

from encoder import EEGEncoder
from decoder import EEGDecoder
from cond_diffusion_transformer import SpatioTemporalConditionedTransformer
from spatio_temporal_condition import SpatioTemporalConditionNet
from channel_ops import downsample_channels_average, upsample_channels_linear


# ---------------------------
# Dataset (must match training segmentation/normalization)
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, data_root, window_size=400, fs=200, mode="denoise"):
        self.fs = fs
        self.mode = mode
        self.window_size = window_size
        self.samples, self.subject_ids = self.load_all_subjects(data_root, window_size, fs)

    def segment_eeg_blocks(self, eeg_data, window_size=400):
        n_blocks, n_channels, n_points = eeg_data.shape
        windows = []
        for b in range(n_blocks):
            for start in range(0, n_points - window_size + 1, window_size):
                seg = eeg_data[b, :, start:start + window_size]
                windows.append(seg)
        return np.stack(windows)

    def normalize_per_channel(self, X):
        n_segments, n_channels, n_time = X.shape
        Xn = np.zeros_like(X, dtype=np.float32)
        for ch in range(n_channels):
            scaler = StandardScaler()
            Xn[:, ch, :] = scaler.fit_transform(X[:, ch, :])
        return Xn

    def load_all_subjects(self, data_root, window_size, fs):
        all_segments = []
        all_subject_ids = []
        subj_id = 0
        for fname in sorted(os.listdir(data_root)):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(data_root, fname)
            eeg = np.load(path)
            print(f"📂 Loading {fname}  shape={eeg.shape}")

            eeg_segments = self.segment_eeg_blocks(eeg, window_size)
            eeg_segments = self.normalize_per_channel(eeg_segments)
            all_segments.append(eeg_segments)
            all_subject_ids.extend([subj_id] * len(eeg_segments))
            subj_id += 1

        all_segments = np.concatenate(all_segments, axis=0)
        all_subject_ids = np.array(all_subject_ids, dtype=np.int64)
        print(f"✅ Loaded {all_segments.shape[0]} segments from all subjects.")
        return all_segments.astype(np.float32), all_subject_ids

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        seg = self.samples[idx]
        seg_t = torch.tensor(seg, dtype=torch.float32)
        seg_31 = downsample_channels_average(seg_t)
        seg_interp = upsample_channels_linear(seg_31)

        if self.mode == "denoise":
            inp = seg_interp
            target = seg_t
        else:
            inp = seg_31
            target = seg_t

        subj_id = int(self.subject_ids[idx])
        return inp, target, subj_id


# ---------------------------
# Helper: load electrode positions
# ---------------------------
def load_channel_positions(locs_file):
    coords = []
    with open(locs_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].lstrip("-").replace('.', '', 1).isdigit():
                try:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append([x, y])
                except:
                    continue
    return torch.tensor(coords, dtype=torch.float32)


# ---------------------------
# Helper: upsample latents
# ---------------------------
def upsample_latents_to_channels(z, target_channels=62):
    if z.dim() != 3:
        raise ValueError("z must be (B, N, D)")
    B, N, D = z.shape
    if N == target_channels:
        return z
    z_t = z.permute(0, 2, 1)
    z_up = F.interpolate(z_t, size=target_channels, mode='linear', align_corners=False)
    z_up = z_up.permute(0, 2, 1)
    return z_up


# ---------------------------
# Metrics helper
# ---------------------------
def compute_metrics(y_true, y_pred):
    """
    Compute MSE, RMSE, MAE, mean Pearson correlation, R², and SNR (dB)
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    mse = np.mean((y_true_np - y_pred_np) ** 2)
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    rmse = np.sqrt(mse)

    # Mean Pearson correlation per channel
    corr_per_channel = [np.corrcoef(y_true_np[i], y_pred_np[i])[0, 1] for i in range(y_true_np.shape[0])]
    mean_corr = np.nanmean(corr_per_channel)

    # Mean R² per channel
    r2_per_channel = [r2_score(y_true_np[i], y_pred_np[i]) for i in range(y_true_np.shape[0])]
    mean_r2 = np.mean(r2_per_channel)

    # Signal-to-noise ratio (dB)
    signal_power = np.mean(y_true_np ** 2)
    noise_power = np.mean((y_true_np - y_pred_np) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

    return mse, rmse, mae, mean_corr, mean_r2, snr


# ---------------------------
# Test pipeline
# ---------------------------
def test_model(data_root="/home/ab_students/EEG-MTP/DATA/SEED",
               checkpoint="eeg_conditioned_model_diff_denoise.pt",
               mode="denoise",
               batch_size=16,
               output_dir="results_test",
               locs_path="/home/ab_students/EEG-MTP/DATA/SEED/channel_62_pos.locs"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | Mode: {mode}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = EEGDataset(data_root, mode=mode)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder_in_channels = 62
    decoder_out_channels = 62
    cond_channels = 62
    n_time = 400
    latent_dim = 128
    out_channels = 62

    encoder = EEGEncoder(in_channels=encoder_in_channels, latent_dim=latent_dim,
                         seq_len=n_time, n_layers=4, n_heads=8, dropout=0.1,
                         pool_factor=4, checkpoint_segments=4).to(device)

    condition_net = SpatioTemporalConditionNet(n_channels=cond_channels, model_dim=latent_dim).to(device)
    transformer = SpatioTemporalConditionedTransformer(latent_dim=latent_dim,
                                                       n_channels=decoder_out_channels,
                                                       n_layers=4, n_heads=8).to(device)
    decoder = EEGDecoder(latent_dim=latent_dim, out_channels=decoder_out_channels,
                         seq_len=n_time).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    condition_net.load_state_dict(ckpt["condition_net"])
    transformer.load_state_dict(ckpt["transformer"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"✅ Loaded checkpoint '{checkpoint}' keys = {ckpt.keys()}")

    encoder.eval(); condition_net.eval(); transformer.eval(); decoder.eval()

    # Metrics accumulators
    total_loss = total_rmse = total_mae = total_corr = total_r2 = total_snr = 0.0
    all_recons, all_targets, all_inps = [], [], []

    # Diffusion config (as training)
    num_diffusion_steps = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    chan_pos = load_channel_positions(locs_path).to(device)

    with torch.no_grad():
        for inp, target, subj_id in tqdm(test_loader, desc="Testing"):
            inp = inp.to(device)
            target = target.to(device)
            subj_id = subj_id.to(device)
            B = inp.shape[0]

            t_steps = torch.randint(0, num_diffusion_steps, (B,), device=device)
            cond_c = subj_id

            alpha_t = alphas_cumprod[t_steps].view(B, 1, 1)
            noise = torch.randn_like(target)
            noisy_target = torch.sqrt(alpha_t) * target + torch.sqrt(1 - alpha_t) * noise

            if mode == "completion":
                lr_eeg = inp
                lr_indices = torch.arange(31, device=device)
                chan_pos_lr = chan_pos[lr_indices]
                cond_tokens, cond_pooled = condition_net(lr_eeg, chan_pos_lr, t_steps, cond_c)
            else:
                lr_eeg = inp
                cond_tokens, cond_pooled = condition_net(lr_eeg, chan_pos, t_steps, cond_c)

            z, _ = encoder(inp)
            if z.shape[1] != out_channels:
                z = upsample_latents_to_channels(z, target_channels=out_channels)

            z_refined = transformer(z, cond_tokens, cond_pooled, t_steps)
            pred_clean = decoder(z_refined)
            recon = pred_clean

            mse, rmse, mae, mean_corr, mean_r2, snr = compute_metrics(target, recon)
            total_loss += mse
            total_rmse += rmse
            total_mae += mae
            total_corr += mean_corr
            total_r2 += mean_r2
            total_snr += snr

            all_recons.append(recon.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_inps.append(inp.cpu().numpy())

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches
    avg_rmse = total_rmse / n_batches
    avg_mae = total_mae / n_batches
    avg_corr = total_corr / n_batches
    avg_r2 = total_r2 / n_batches
    avg_snr = total_snr / n_batches

    print(f"📊 Test Metrics → MSE={avg_loss:.6f} | RMSE={avg_rmse:.6f} | MAE={avg_mae:.6f} | "
          f"Corr={avg_corr:.4f} | R2={avg_r2:.4f} | SNR={avg_snr:.2f} dB")

    metrics_path = os.path.join(output_dir, f"metrics_{mode}.csv")
    pd.DataFrame([{
        "Mode": mode,
        "MSE": avg_loss,
        "RMSE": avg_rmse,
        "MAE": avg_mae,
        "MeanCorr": avg_corr,
        "R2": avg_r2,
        "SNR_dB": avg_snr
    }]).to_csv(metrics_path, index=False)
    print(f"✅ Saved metrics → {metrics_path}")

    # Save reconstructed arrays
    np.savez(os.path.join(output_dir, f"reconstructions_{mode}.npz"),
             recons=np.concatenate(all_recons, axis=0),
             targets=np.concatenate(all_targets, axis=0),
             inputs=np.concatenate(all_inps, axis=0))
    print(f"✅ Saved reconstructed EEG segments → {output_dir}/reconstructions_{mode}.npz")

    # Per-channel CSVs + multi-channel grid (first sample)
    if len(all_recons) == 0:
        print("No reconstructions collected. Exiting.")
        return

    recon_ex = all_recons[0][0]
    target_ex = all_targets[0][0]
    inp_ex = all_inps[0][0]

    save_dir = os.path.join(output_dir, f"channel_samples_{mode}")
    os.makedirs(save_dir, exist_ok=True)

    n_save_channels = min(10, recon_ex.shape[0])
    ch_indices = list(range(n_save_channels))
    for ch in ch_indices:
        df = pd.DataFrame({
            "Input": inp_ex[ch],
            "Reconstructed": recon_ex[ch],
            "Target": target_ex[ch]
        })
        csv_path = os.path.join(save_dir, f"channel_{ch+1:02d}.csv")
        df.to_csv(csv_path, index=False)
    print(f"✅ Saved detailed per-channel CSVs → {save_dir}")

    fig, axes = plt.subplots(2, int(np.ceil(n_save_channels / 2)), figsize=(18, 6))
    axes = axes.flatten()
    for i, ch in enumerate(ch_indices):
        ax = axes[i]
        ax.plot(target_ex[ch], label="Target", linewidth=1)
        ax.plot(inp_ex[ch], label="Input", linestyle="--", linewidth=0.8)
        ax.plot(recon_ex[ch], label="Reconstructed", linestyle=":", linewidth=1.2)
        ax.set_title(f"Channel {ch+1}")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    grid_fig_path = os.path.join(output_dir, f"multi_channel_reconstruction_{mode}.png")
    plt.savefig(grid_fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved {n_save_channels}-channel grid visualization → {grid_fig_path}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EEG conditioned transformer")
    parser.add_argument("--data_root", type=str, default="/home/ab_students/EEG-MTP/DATA/SEED")
    parser.add_argument("--checkpoint", type=str, default="eeg_conditioned_model_diff_denoise.pt")
    parser.add_argument("--mode", type=str, choices=["denoise", "completion"], default="denoise")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--locs_path", type=str, default="/home/ab_students/EEG-MTP/DATA/SEED/channel_62_pos.locs")
    args = parser.parse_args()

    test_model(data_root=args.data_root,
               checkpoint=args.checkpoint,
               mode=args.mode,
               batch_size=args.batch_size,
               locs_path=args.locs_path)
