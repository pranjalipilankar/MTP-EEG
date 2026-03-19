#!/usr/bin/env python3
"""
test_cond_transformer.py - TRUE SUBJECT-WISE TESTING VERSION
Tests ONE SUBJECT AT A TIME with separate DataLoaders per subject.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from encoder import EEGEncoder
from decoder import EEGDecoder
from cond_diffusion_transformer import SpatioTemporalConditionedTransformer
from spatio_temporal_condition import SpatioTemporalConditionNet
from channel_ops import downsample_channels_average, upsample_channels_linear
from scipy.signal import butter, filtfilt


# ---------------------------
# Bandpass filter helper
# ---------------------------
def bandpass_filter(data, low=1.0, high=40.0, fs=200.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)


# ---------------------------
# Dataset (UNCHANGED)
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
                windows.append(eeg_data[b, :, start:start + window_size])
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
        print(f"✅ Loaded {all_segments.shape[0]} segments from {subj_id} subjects.")
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
# Channel positions (UNCHANGED)
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
# Upsample latent tokens (UNCHANGED)
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
# FIXED Metrics computation
# ---------------------------
def compute_metrics(y_true, y_pred):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    B, C, T = y_true.shape
    
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    rmse = np.sqrt(mse)
    
    corr_per_channel = [np.corrcoef(y_true_np[:, ch, :].flatten(), 
                                   y_pred_np[:, ch, :].flatten())[0,1] 
                       for ch in range(C)]
    mean_corr = np.nanmean(corr_per_channel)
    
    r2_per_channel = [r2_score(y_true_np[:, ch, :].flatten(), 
                              y_pred_np[:, ch, :].flatten()) for ch in range(C)]
    mean_r2 = np.mean(r2_per_channel)
    
    signal_power = np.mean(y_true_np ** 2)
    noise_power = np.mean((y_true_np - y_pred_np) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    return mse, rmse, mae, mean_corr, mean_r2, snr


# ---------------------------
# **TRUE SUBJECT-WISE TESTING** - NEW!
# ---------------------------
def create_subject_loaders(dataset):
    """Create separate DataLoader for EACH subject"""
    subj_data = {}
    for idx in range(len(dataset)):
        subj_id = dataset.subject_ids[idx]
        if subj_id not in subj_data:
            subj_data[subj_id] = []
        subj_data[subj_id].append(idx)
    
    loaders = {}
    for subj_id, indices in subj_data.items():
        subj_dataset = torch.utils.data.Subset(dataset, indices)
        loaders[subj_id] = DataLoader(subj_dataset, batch_size=16, 
                                    shuffle=False, num_workers=4)
    return loaders, len(subj_data)


def test_model(data_root,
               checkpoint,
               mode="denoise",
               batch_size=16,
               output_dir="results_test",
               locs_path=None):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device} | Mode: {mode}")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = EEGDataset(data_root, mode=mode)
    
    # **CREATE SUBJECT-WISE LOADERS**
    subject_loaders, n_subjects = create_subject_loaders(dataset)
    print(f"✅ Created {n_subjects} subject-wise DataLoaders")

    # Model setup
    encoder_in_channels = 62
    decoder_out_channels = 62
    cond_channels = 62
    n_time = 400
    latent_dim = 128
    n_conditions = 8
    out_channels = 62

    encoder = EEGEncoder(in_channels=encoder_in_channels, latent_dim=latent_dim,
                         seq_len=n_time, n_layers=4, n_heads=8, dropout=0.1,
                         pool_factor=4, checkpoint_segments=4).to(device)

    condition_net = SpatioTemporalConditionNet(n_channels=cond_channels,
                                              model_dim=latent_dim,
                                              n_conditions=n_conditions).to(device)

    transformer = SpatioTemporalConditionedTransformer(latent_dim=latent_dim,
                                                      n_channels=decoder_out_channels,
                                                      n_layers=4,
                                                      n_heads=8).to(device)

    decoder = EEGDecoder(latent_dim=latent_dim, out_channels=decoder_out_channels,
                         seq_len=n_time).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    state = ckpt["condition_net"]
    model_keys = set(condition_net.state_dict().keys())
    for k in list(state.keys()):
        if k not in model_keys:
            del state[k]
    condition_net.load_state_dict(state)
    transformer.load_state_dict(ckpt["transformer"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"✅ Loaded checkpoint '{checkpoint}'")

    encoder.eval(); condition_net.eval(); transformer.eval(); decoder.eval()

    # Diffusion setup
    num_diffusion_steps = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    chan_pos = load_channel_positions(locs_path).to(device)

    # **SUBJECT-WISE TESTING**
    all_subj_metrics = []
    
    for subj_id, subj_loader in tqdm(subject_loaders.items(), desc="Subjects"):
        print(f"\n🧠 Testing Subject {subj_id:03d} ({len(subj_loader.dataset)} segments)")
        
        subj_recons, subj_targets, subj_inputs = [], [], []
        subj_total_mse = subj_total_rmse = subj_total_mae = 0
        subj_total_corr = subj_total_r2 = subj_total_snr = 0
        n_subj_batches = 0
        
        with torch.no_grad():
            for inp, target, _ in tqdm(subj_loader, desc=f"Subj{subj_id}", leave=False):
                inp = inp.to(device)
                target = target.to(device)
                
                B = inp.shape[0]
                t_steps = torch.randint(0, num_diffusion_steps, (B,), device=device)
                cond_c = torch.full((B,), subj_id, dtype=torch.long, device=device)  # SAME subject
                
                alpha_t = alphas_cumprod[t_steps].view(B, 1, 1)
                noise = torch.randn_like(target)
                noisy_target = torch.sqrt(alpha_t) * target + torch.sqrt(1 - alpha_t) * noise
                lr_eeg = inp
                cond_tokens, cond_pooled = condition_net(lr_eeg, chan_pos, t_steps, cond_c)

                z, _ = encoder(inp)
                if z.shape[1] != out_channels:
                    z = upsample_latents_to_channels(z, target_channels=out_channels)

                z_refined = transformer(z, cond_tokens, cond_pooled, t_steps)
                pred_clean = decoder(z_refined)
                recon = pred_clean

                # Store subject data
                subj_recons.append(recon.cpu().numpy())
                subj_targets.append(target.cpu().numpy())
                subj_inputs.append(inp.cpu().numpy())
                
                # Accumulate metrics
                mse, rmse, mae, mean_corr, mean_r2, snr = compute_metrics(target, recon)
                subj_total_mse += mse
                subj_total_rmse += rmse
                subj_total_mae += mae
                subj_total_corr += mean_corr
                subj_total_r2 += mean_r2
                subj_total_snr += snr
                n_subj_batches += 1
        
        # **SUBJECT METRICS**
        subj_avg_mse = subj_total_mse / n_subj_batches
        subj_avg_rmse = subj_total_rmse / n_subj_batches
        subj_avg_mae = subj_total_mae / n_subj_batches
        subj_avg_corr = subj_total_corr / n_subj_batches
        subj_avg_r2 = subj_total_r2 / n_subj_batches
        subj_avg_snr = subj_total_snr / n_subj_batches
        
        # **SAVE SUBJECT FILES**
        subj_dir = os.path.join(output_dir, f"subject_{subj_id:03d}")
        os.makedirs(subj_dir, exist_ok=True)
        
        reconstructed = np.concatenate(subj_recons, axis=0)      # (N_subj, 62, 400)
        ground_truth = np.concatenate(subj_targets, axis=0)     # (N_subj, 62, 400)
        model_inputs = np.concatenate(subj_inputs, axis=0)      # (N_subj, 62, 400)
        
        np.save(os.path.join(subj_dir, "reconstructed.npy"), reconstructed)
        np.save(os.path.join(subj_dir, "ground_truth.npy"), ground_truth)
        np.save(os.path.join(subj_dir, "inputs.npy"), model_inputs)

        subj_metrics = {
            "Subject_ID": subj_id,
            "N_Segments": reconstructed.shape[0],
            "MSE": subj_avg_mse, "RMSE": subj_avg_rmse, "MAE": subj_avg_mae,
            "MeanCorr": subj_avg_corr, "R2": subj_avg_r2, "SNR_dB": subj_avg_snr
        }
        all_subj_metrics.append(subj_metrics)
        
        print(f"✅ Subject {subj_id:03d}: RMSE={subj_avg_rmse:.6f} | Corr={subj_avg_corr:.4f} | "
              f"{reconstructed.shape[0]} segments → {subj_dir}/")

    # **OVERALL SUMMARY**
    metrics_df = pd.DataFrame(all_subj_metrics)
    overall_path = os.path.join(output_dir, f"subject_wise_metrics_{mode}.csv")
    metrics_df.to_csv(overall_path, index=False)
    
    summary = {
        "Mode": mode,
        "N_Subjects": n_subjects,
        "Mean_RMSE": metrics_df["RMSE"].mean(),
        "Std_RMSE": metrics_df["RMSE"].std(),
        "Mean_Corr": metrics_df["MeanCorr"].mean(),
        "Best_Subject_RMSE": metrics_df["RMSE"].min()
    }
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, f"summary_{mode}.csv"), index=False)
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"Mean RMSE: {summary['Mean_RMSE']:.6f} ± {summary['Std_RMSE']:.6f}")
    print(f"Best Subject RMSE: {summary['Best_Subject_RMSE']:.6f}")
    print(f"📁 Check {output_dir}/subject_* for individual results")
    print(f"📊 Subject-wise metrics: {overall_path}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subject-wise EEG evaluation")
    parser.add_argument("--data_root", type=str, default="/home/ab_students/EEG-MTP/DATA/SEED_processed/val")
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
