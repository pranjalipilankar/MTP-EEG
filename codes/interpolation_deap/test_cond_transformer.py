#!/usr/bin/env python3
"""
test_cond_transformer_with_stats.py
Evaluate trained Encoder + Transformer + Decoder for denoising/completion.
Includes:
- Full reconstruction evaluation
- Batch-level metrics (MSE, RMSE, MAE, mean Corr, mean R², mean SNR)
- Epoch/Channel metrics
- Statistical comparisons: paired t-tests, Wilcoxon, FDR, Cohen’s d
- Covariance/correlation heatmaps
- Multi-channel plots
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from encoder import EEGEncoder
from decoder import EEGDecoder
from cond_diffusion_transformer import SpatioTemporalConditionedTransformer
from channel_ops import downsample_channels_keep_odd, upsample_channels_linear_even
from spatio_temporal_condition import SpatioTemporalConditionNet
from graph_harmonic_condition import GraphHarmonicConditionNet
from fourier_positional_condition import FourierPositionConditionNet

import seaborn as sns
from scipy import stats
import statsmodels.stats.multitest as smm
from sklearn.metrics import r2_score   # ★ NEW

#######################################################################
# ---------------------- HELPER FUNCTIONS -----------------------------
#######################################################################

def channel_cov_matrix(x):
    B, C, T = x.shape
    xm = x - x.mean(dim=2, keepdim=True)
    cov = torch.matmul(xm, xm.transpose(1, 2)) / (T - 1)
    return cov


def channel_corr_matrix(x):
    B, C, T = x.shape
    xm = x - x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-6
    xn = xm / std
    corr = torch.matmul(xn, xn.transpose(1, 2)) / (T - 1)
    return corr


def plot_cov_heatmaps(cov_target, cov_recon, save_dir, prefix="cov"):
    os.makedirs(save_dir, exist_ok=True)
    diff = cov_recon - cov_target
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, mat, title in zip(
        axes,
        [cov_target, cov_recon, diff],
        ["Target Covariance", "Reconstructed", "Δ Covariance (Recon - Target)"]
    ):
        sns.heatmap(mat, cmap="viridis", square=True, ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_heatmaps.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved covariance heatmaps → {path}")


def plot_corr_heatmaps(corr_target, corr_recon, save_dir, prefix="corr"):
    os.makedirs(save_dir, exist_ok=True)
    diff = corr_recon - corr_target
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, mat, title in zip(
        axes,
        [corr_target, corr_recon, diff],
        ["Target Correlation", "Reconstructed", "Δ Correlation (Recon - Target)"]
    ):
        sns.heatmap(mat, cmap="coolwarm", square=True, vmin=-1, vmax=1, ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_heatmaps.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmaps → {path}")


#######################################################################
# ---------------------- DATASET -------------------------------------
#######################################################################

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split="train", window_size=400, fs=200, mode="denoise"):
        self.fs = fs
        self.window_size = window_size
        self.mode = mode

        data = np.load(data_root)
        key = {"train": "X_train", "val": "X_val", "test": "X_test"}[split]
        X = data[key]  # (N, C, T)
        print(f"Loaded {key}, shape={X.shape}")

        self.samples = self.segment_eeg_blocks(X, window_size)
        self.samples = self.normalize_per_channel(self.samples)

        self.subject_ids = np.zeros(len(self.samples), dtype=np.int64)
        print(f"Dataset '{split}': {len(self.samples)} segments")

    def segment_eeg_blocks(self, X, window=400):
        N, C, T = X.shape
        blocks = []
        for i in range(N):
            for s in range(0, T - window + 1, window):
                blocks.append(X[i, :, s:s+window])
        return np.stack(blocks)

    def normalize_per_channel(self, X):
        Z = np.zeros_like(X, dtype=np.float32)
        for ch in range(X.shape[1]):
            mu = X[:, ch, :].mean(axis=1, keepdims=True)
            sd = X[:, ch, :].std(axis=1, keepdims=True) + 1e-6
            Z[:, ch, :] = (X[:, ch, :] - mu) / sd
        return Z

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        seg = torch.tensor(self.samples[idx], dtype=torch.float32)
        seg16 = downsample_channels_keep_odd(seg)
        seg_interp = upsample_channels_linear_even(seg16)
        return seg_interp, seg, int(self.subject_ids[idx])


#######################################################################
# ---------------------- POSITION + LATENT OPS ------------------------
#######################################################################

def generate_channel_positions(n_channels, device="cpu"):
    theta = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
    coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return torch.tensor(coords, dtype=torch.float32, device=device)


def upsample_latents_to_channels(z, target_channels):
    B, N, D = z.shape
    if N == target_channels:
        return z
    z_t = z.permute(0, 2, 1)
    z_up = F.interpolate(z_t, size=target_channels, mode="linear", align_corners=False)
    return z_up.permute(0, 2, 1)


#######################################################################
# ---------------------- EPOCH-CHANNEL METRICS ------------------------
#######################################################################

def compute_epoch_channel_metrics(targets, preds):
    N, C, T = targets.shape
    mse = np.mean((targets - preds) ** 2, axis=2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(targets - preds), axis=2)

    r = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        for ch in range(C):
            a, b = targets[i, ch], preds[i, ch]
            if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                r[i, ch] = 0
            else:
                r[i, ch] = stats.pearsonr(a, b)[0]

    var_sig = np.var(targets, axis=2)
    var_err = np.var(targets - preds, axis=2)
    snr = 10 * np.log10((var_sig + 1e-12) / (var_err + 1e-12))

    return dict(mse=mse, rmse=rmse, mae=mae, r=r, snr_db=snr)


def paired_stat_tests(a, b, test_type="paired_t"):
    N, C = a.shape
    stat = np.zeros(C)
    p = np.ones(C)
    for ch in range(C):
        x, y = a[:, ch], b[:, ch]
        if test_type == "paired_t":
            stat[ch], p[ch] = stats.ttest_rel(x, y, nan_policy="omit")
        else:
            try:
                stat[ch], p[ch] = stats.wilcoxon(x, y)
            except:
                stat[ch], p[ch] = np.nan, 1.0
    return stat, p


def cohens_d_paired(a, b):
    diff = a - b
    md = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1) + 1e-12
    return md / sd


#######################################################################
# ---------------------- BATCH-LEVEL METRICS --------------------------
#######################################################################

def compute_metrics(y_true, y_pred):
    """
    Computes:
    - MSE, RMSE, MAE
    - mean Pearson corr
    - mean R²
    - mean SNR (dB)
    """
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()

    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(mse))

    C = yt.shape[1]
    cors, r2s, snrs = [], [], []

    for ch in range(C):
        a = yt[0, ch]
        b = yp[0, ch]

        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            cors.append(0.0)
        else:
            cors.append(np.corrcoef(a, b)[0, 1])

        try:
            r2s.append(float(r2_score(a, b)))
        except:
            r2s.append(0.0)

        ps = np.mean(a ** 2)
        pn = np.mean((a - b) ** 2)
        snrs.append(10 * np.log10((ps + 1e-12) / (pn + 1e-12)))

    return (
        mse,
        rmse,
        mae,
        float(np.mean(cors)),
        float(np.mean(r2s)),
        float(np.mean(snrs)),
    )


#######################################################################
# -------------------------- MAIN TEST FUNCTION -----------------------
#######################################################################

def test_model(data_root, checkpoint, split="test", mode="denoise",
               batch_size=16, output_dir="results_test"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Split={split} | Mode={mode}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = EEGDataset(data_root, split=split, mode=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder = EEGEncoder(32, 128, 400, n_layers=4, n_heads=8, dropout=0.1,
                         pool_factor=4, checkpoint_segments=4).to(device)
    transformer = SpatioTemporalConditionedTransformer(128, 32, 4, 8).to(device)
    decoder = EEGDecoder(128, 32, 400).to(device)
    
    encoder_in_channels = 32 
    decoder_out_channels = 32 
    cond_channels = 32
    
    n_time = 400 
    latent_dim = 128 
    n_conditions = 8 
    out_channels = decoder_out_channels
    condition_net = FourierPositionConditionNet(n_channels=32, model_dim=latent_dim).to(device) 
    #condition_net = SpatioTemporalConditionNet(n_channels=32, model_dim=latent_dim, n_conditions=8).to(device) 
    #condition_net = GraphHarmonicConditionNet(n_channels=32, model_dim=latent_dim, n_conditions=8).to(device)
    
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    transformer.load_state_dict(ckpt["transformer"])
    decoder.load_state_dict(ckpt["decoder"])
    condition_net.load_state_dict(ckpt["condition_net"], strict=False)

    print(f"Loaded checkpoint: {checkpoint}")
    print("Epoch:", ckpt.get("epoch", "N/A"))

    encoder.eval(); transformer.eval(); decoder.eval(); condition_net.eval()

    # ★ NEW: totals including corr, R2, SNR
    tot_mse = tot_rmse = tot_mae = 0.0
    tot_corr = tot_r2 = tot_snr = 0.0

    all_recons = []
    all_targets = []
    all_inputs = []

    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    with torch.no_grad():
        for inp, target, subj in tqdm(loader):
            inp = inp.to(device)
            target = target.to(device)
            subj = subj.to(device)

            B = inp.size(0)
            t = torch.randint(0, 1000, (B,), device=device)

            chan_pos = generate_channel_positions(32, device=device)
            cond_tokens, cond_pool = condition_net(inp, chan_pos, t, subj)

            z, _ = encoder(inp)
            z = upsample_latents_to_channels(z, 32)

            z_ref = transformer(z, cond_tokens, cond_pool, t)
            pred_noise = decoder(z_ref)

            recon = inp - pred_noise

            # ★ accumulate metrics
            mse, rmse, mae, corr, r2, snr = compute_metrics(target, recon)
            tot_mse += mse
            tot_rmse += rmse
            tot_mae += mae
            tot_corr += corr
            tot_r2 += r2
            tot_snr += snr

            # ★ accumulate for stats
            all_recons.append(recon.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_inputs.append(inp.cpu().numpy())

    # convert lists → arrays (N, C, T)
    all_recons = np.concatenate(all_recons, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    N_batches = len(loader)
    avg_mse = tot_mse / N_batches
    avg_rmse = tot_rmse / N_batches
    avg_mae = tot_mae / N_batches
    avg_corr = tot_corr / N_batches
    avg_r2 = tot_r2 / N_batches
    avg_snr = tot_snr / N_batches

    print("\n================= FINAL BATCH-LEVEL METRICS =================")
    print(f"MSE : {avg_mse:.6f}")
    print(f"RMSE: {avg_rmse:.6f}")
    print(f"MAE : {avg_mae:.6f}")
    print(f"Corr: {avg_corr:.4f}")
    print(f"R²  : {avg_r2:.4f}")
    print(f"SNR : {avg_snr:.2f} dB")
    print("==============================================================\n")

    ###################################################################
    # SAVE RECONS
    ###################################################################
    np.savez(os.path.join(output_dir, f"reconstructions_{split}_{mode}.npz"),
             recons=all_recons, targets=all_targets, inputs=all_inputs)
    print(f"Saved recon arrays → {output_dir}")

    ###################################################################
    # FIRST-SAMPLE PLOTS
    ###################################################################
    sample_recon = all_recons[0]
    sample_target = all_targets[0]
    sample_input = all_inputs[0]

    plot_dir = os.path.join(output_dir, f"plots_{split}_{mode}")
    os.makedirs(plot_dir, exist_ok=True)

    C = sample_recon.shape[0]
    n_show = min(10, C)

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    axes = axes.flatten()

    for i in range(n_show):
        ax = axes[i]
        ax.plot(sample_target[i], label="Target")
        ax.plot(sample_input[i], "--", label="Interp")
        ax.plot(sample_recon[i], ":", label="Recon")
        ax.set_title(f"Ch {i+1}")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "reconstruction_grid.png"), dpi=300)
    plt.close()

    ###################################################################
    # Covariance & Correlation heatmaps
    ###################################################################
    t_tensor = torch.tensor(sample_target).unsqueeze(0)
    r_tensor = torch.tensor(sample_recon).unsqueeze(0)

    cov_t = channel_cov_matrix(t_tensor)[0].numpy()
    cov_r = channel_cov_matrix(r_tensor)[0].numpy()
    plot_cov_heatmaps(cov_t, cov_r, plot_dir, prefix="cov")

    corr_t = channel_corr_matrix(t_tensor)[0].numpy()
    corr_r = channel_corr_matrix(r_tensor)[0].numpy()
    plot_corr_heatmaps(corr_t, corr_r, plot_dir, prefix="corr")

    ###################################################################
    # STATISTICAL TESTS
    ###################################################################
    print("\nRunning statistical tests...")

    m_interp = compute_epoch_channel_metrics(all_targets, all_inputs)
    m_model  = compute_epoch_channel_metrics(all_targets, all_recons)

    C = all_targets.shape[1]

    # per-channel MEANS
    mean_rmse_interp = m_interp["rmse"].mean(0)
    mean_rmse_model  = m_model["rmse"].mean(0)

    mean_mae_interp = m_interp["mae"].mean(0)
    mean_mae_model  = m_model["mae"].mean(0)

    mean_r_interp = m_interp["r"].mean(0)
    mean_r_model  = m_model["r"].mean(0)

    mean_snr_interp = m_interp["snr_db"].mean(0)
    mean_snr_model  = m_model["snr_db"].mean(0)

    # paired tests
    stat_rmse, p_rmse = paired_stat_tests(m_model["rmse"], m_interp["rmse"])
    stat_mae,  p_mae  = paired_stat_tests(m_model["mae"],  m_interp["mae"])
    stat_r,    p_r    = paired_stat_tests(m_model["r"],    m_interp["r"])
    stat_snr,  p_snr  = paired_stat_tests(m_model["snr_db"], m_interp["snr_db"])

    # FDR
    rej_rmse, p_rmse_fdr = smm.multipletests(p_rmse, alpha=0.05, method="fdr_bh")[:2]
    rej_mae,  p_mae_fdr  = smm.multipletests(p_mae,  alpha=0.05, method="fdr_bh")[:2]
    rej_r,    p_r_fdr    = smm.multipletests(p_r,    alpha=0.05, method="fdr_bh")[:2]
    rej_snr,  p_snr_fdr  = smm.multipletests(p_snr,  alpha=0.05, method="fdr_bh")[:2]

    # Effect sizes
    d_rmse = cohens_d_paired(m_model["rmse"], m_interp["rmse"])
    d_mae  = cohens_d_paired(m_model["mae"], m_interp["mae"])
    d_r    = cohens_d_paired(m_model["r"], m_interp["r"])
    d_snr  = cohens_d_paired(m_model["snr_db"], m_interp["snr_db"])

    df = pd.DataFrame({
        "chan": np.arange(1, C+1),

        "mean_rmse_interp": mean_rmse_interp,
        "mean_rmse_model":  mean_rmse_model,
        "rmse_tstat": stat_rmse,
        "rmse_pval": p_rmse,
        "rmse_pval_fdr": p_rmse_fdr,
        "rmse_sig_fdr": rej_rmse,
        "rmse_d": d_rmse,

        "mean_mae_interp": mean_mae_interp,
        "mean_mae_model":  mean_mae_model,
        "mae_tstat": stat_mae,
        "mae_pval": p_mae,
        "mae_pval_fdr": p_mae_fdr,
        "mae_sig_fdr": rej_mae,
        "mae_d": d_mae,

        "mean_r_interp": mean_r_interp,
        "mean_r_model":  mean_r_model,
        "r_tstat": stat_r,
        "r_pval": p_r,
        "r_pval_fdr": p_r_fdr,
        "r_sig_fdr": rej_r,
        "r_d": d_r,

        "mean_snr_interp": mean_snr_interp,
        "mean_snr_model":  mean_snr_model,
        "snr_tstat": stat_snr,
        "snr_pval": p_snr,
        "snr_pval_fdr": p_snr_fdr,
        "snr_sig_fdr": rej_snr,
        "snr_d": d_snr,
    })

    csv_path = os.path.join(output_dir, f"stats_{split}_{mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved stats CSV → {csv_path}")

    print("\nSignificant (FDR p<0.05):")
    print("RMSE ↓ :", np.where(rej_rmse)[0]+1)
    print("MAE  ↓ :", np.where(rej_mae)[0]+1)
    print("Corr ↑ :", np.where(rej_r)[0]+1)
    print("SNR  ↑ :", np.where(rej_snr)[0]+1)

    print("\n✔ DONE")


#######################################################################
# ------------------------------ CLI ---------------------------------
#######################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz")
    p.add_argument("--checkpoint", type=str, default="fourier_model_denoise_2.pt")
    p.add_argument("--split", choices=["train","val","test"], default="test")
    p.add_argument("--mode", choices=["denoise","completion"], default="denoise")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="results_test_fourier_2")
    a = p.parse_args()

    test_model(
        data_root=a.data_root,
        checkpoint=a.checkpoint,
        split=a.split,
        mode=a.mode,
        batch_size=a.batch_size,
        output_dir=a.output_dir,
    )
