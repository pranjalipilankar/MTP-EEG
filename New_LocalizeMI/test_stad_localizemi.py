#!/usr/bin/env python3
"""
Localize-MI STAD testing and visualization.

This script loads the trained STAD checkpoint, runs a small evaluation pass,
and saves:
- reconstruction metrics
- training curve plots
- metric summary graphs
- frequency-band graphs
- topographic maps
- time-series comparisons
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from mne.time_frequency import psd_array_multitaper
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_stad_localizemi import (  # noqa: E402
    LocalizeMISTADDataset,
    STAD_LocalizeMI,
    compute_nmse,
    compute_pcc,
    compute_snr,
    get_beta_schedule,
    get_channel_positions,
    get_diffusion_params,
)


FREQ_BANDS = {
    "delta (0.5-4 Hz)": (0.5, 4),
    "theta (4-8 Hz)": (4, 8),
    "alpha (8-13 Hz)": (8, 13),
    "beta (13-30 Hz)": (13, 30),
    "gamma (30-45 Hz)": (30, 45),
}


def load_fold_subject_split(mae_results_dir, fold):
    split_file = Path(mae_results_dir) / "fold_splits.json"
    if not split_file.exists():
        raise FileNotFoundError(f"fold_splits.json not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    matched = None
    for item in splits:
        if int(item.get("fold", -1)) == int(fold):
            matched = item
            break

    if matched is None:
        available = [int(x.get("fold", -1)) for x in splits]
        raise ValueError(
            f"Fold {fold} not found in {split_file}. Available folds: {available}"
        )

    train_subjects = matched.get("train_subjects", [])
    val_subjects = matched.get("val_subjects", [])
    if not train_subjects or not val_subjects:
        raise ValueError(f"Invalid fold split format in {split_file} for fold {fold}")

    return train_subjects, val_subjects


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return checkpoint, missing_keys, unexpected_keys


def get_samples(dataset, n_samples, seed=42):
    n_samples = min(int(n_samples), len(dataset))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if len(indices) > n_samples:
        indices = rng.choice(indices, size=n_samples, replace=False)
        indices.sort()

    samples = []
    for idx in indices:
        samples.append(dataset[int(idx)])
    return samples, indices.tolist()


@torch.no_grad()
def ddim_generate_sr(model, lr_eeg, device, diff_params, timesteps=1000, ddim_steps=50):
    model.eval()
    lr_eeg = lr_eeg.to(device)
    batch_size = lr_eeg.size(0)

    latent = torch.randn(batch_size, model.num_patches, model.latent_dim, device=device)
    lr_pos = get_channel_positions(model.lr_channels, device, batch_size)
    schedule = torch.linspace(timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)

    for step_idx, timestep in enumerate(schedule):
        t_batch = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
        pred_epsilon = model(lr_eeg, latent, t_batch, lr_pos)

        alpha_t = diff_params["sqrt_alphas_cumprod"][timestep] ** 2
        if step_idx < len(schedule) - 1:
            alpha_t_prev = diff_params["sqrt_alphas_cumprod"][schedule[step_idx + 1]] ** 2
        else:
            alpha_t_prev = torch.tensor(1.0, device=device)

        pred_z0 = (latent - torch.sqrt(1 - alpha_t) * pred_epsilon) / torch.sqrt(alpha_t)
        direction = torch.sqrt(1 - alpha_t_prev) * pred_epsilon
        latent = torch.sqrt(alpha_t_prev) * pred_z0 + direction

    return model.decode_latent_to_sr(latent, lr_eeg)


@torch.no_grad()
def reconstruct_like_training_validation(model, lr_eeg, sr_gt, device, diff_params, timesteps=1000):
    """
    Mirror the training validation path so test metrics are directly comparable.

    This uses the same teacher-forced denoising setup as train_stad_localizemi.py:
    - encode the ground-truth SR sample to latent
    - sample a random diffusion timestep
    - corrupt with Gaussian noise
    - predict epsilon and reconstruct sr_pred from the denoised latent
    """
    model.eval()
    lr_eeg = lr_eeg.to(device)
    sr_gt = sr_gt.to(device)
    batch_size = lr_eeg.size(0)

    z0 = model.encode_sr(sr_gt)
    z0 = torch.clamp(z0, min=-10.0, max=10.0)

    t = torch.randint(0, timesteps, (batch_size,), device=device)
    epsilon = torch.randn_like(z0)
    sqrt_alpha = diff_params["sqrt_alphas_cumprod"][t].view(batch_size, 1, 1)
    sqrt_one_minus = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(batch_size, 1, 1)
    zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon

    lr_pos = get_channel_positions(model.lr_channels, device, batch_size)
    pred_epsilon = model(lr_eeg, zt, t, lr_pos)
    pred_z0 = (zt - sqrt_one_minus * pred_epsilon) / (sqrt_alpha + 1e-8)
    pred_z0 = torch.clamp(pred_z0, min=-10.0, max=10.0)
    return model.decode_latent_to_sr(pred_z0, lr_eeg)


def make_montage(n_channels):
    positions = get_channel_positions(n_channels, device="cpu", batch_size=1).squeeze(0).cpu().numpy()
    ch_names = [f"E{i+1}" for i in range(n_channels)]
    ch_pos = {
        name: np.array([x, y, 0.0], dtype=float)
        for name, (x, y) in zip(ch_names, positions)
    }
    info = mne.create_info(ch_names=ch_names, sfreq=8000, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage)
    return info


def band_power_from_eeg(eeg, fs):
    psd, freqs = psd_array_multitaper(
        eeg,
        sfreq=fs,
        fmin=0.5,
        fmax=45,
        adaptive=True,
        normalization="full",
        verbose=False,
    )

    band_powers = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band_name] = np.mean(psd[:, mask], axis=1)
    return band_powers


def plot_training_history(metrics_history, save_path):
    if not metrics_history:
        return

    epochs = [item["epoch"] for item in metrics_history]
    train_loss = [item["train_loss"] for item in metrics_history]
    val_loss = [item["val_loss"] for item in metrics_history]
    nmse = [item["nmse"] for item in metrics_history]
    pcc = [item["pcc"] for item in metrics_history]
    snr = [item["snr"] for item in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(epochs, train_loss, label="Train", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, nmse, color="#457B9D", linewidth=2)
    axes[1].set_title("NMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NMSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, pcc, color="#2A9D8F", linewidth=2)
    axes[2].set_title("PCC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PCC")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, snr, color="#E76F51", linewidth=2)
    axes[3].set_title("SNR")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("dB")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Localize-MI STAD Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_summary(avg_metrics, save_path):
    keys = ["pcc", "nmse", "snr", "mae"]
    labels = ["PCC", "NMSE", "SNR", "MAE"]
    values = [avg_metrics[k] for k in keys]
    colors = ["#2A9D8F", "#457B9D", "#E76F51", "#F4A261"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title("Average Reconstruction Metrics")
    ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_band_power_summary(eeg_dict, sr_eeg, fs, save_path):
    band_names = list(FREQ_BANDS.keys())
    rows = ["64ch", "128ch", "256ch_SR"]
    eeg_map = {
        "64ch": eeg_dict["64ch"],
        "128ch": eeg_dict["128ch"],
        "256ch_SR": sr_eeg,
    }

    band_values = {row: [] for row in rows}
    for row in rows:
        band_power = band_power_from_eeg(eeg_map[row], fs)
        for band_name in band_names:
            band_values[row].append(float(np.mean(band_power[band_name])))

    x = np.arange(len(band_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, band_values["64ch"], width, label="64-ch LR", color="#E63946")
    ax.bar(x, band_values["128ch"], width, label="128-ch HR", color="#457B9D")
    ax.bar(x + width, band_values["256ch_SR"], width, label="256-ch SR", color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=20, ha="right")
    ax.set_ylabel("Mean band power")
    ax.set_title("Frequency-Band Power Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_multireso_topomap(eeg_dict, sr_eeg, fs, save_path):
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, len(FREQ_BANDS), figure=fig, hspace=0.25, wspace=0.18)

    rows = [
        ("64ch", "64-ch LR", eeg_dict["64ch"], 64),
        ("128ch", "128-ch HR", eeg_dict["128ch"], 128),
        ("256ch_SR", "256-ch SR", sr_eeg, 256),
    ]

    for row_idx, (_, row_label, eeg, n_channels) in enumerate(rows):
        info = make_montage(n_channels)
        band_power = band_power_from_eeg(eeg, fs)

        for col_idx, (band_name, _) in enumerate(FREQ_BANDS.items()):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            values = band_power[band_name]
            if len(values) != n_channels:
                values = values[:n_channels]

            try:
                im, _ = mne.viz.plot_topomap(
                    values,
                    info,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    contours=8,
                    outlines="head",
                    sphere="auto",
                    sensors=True,
                    res=128,
                    extrapolate="head",
                    border="mean",
                )
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Topomap failed\n{exc}", ha="center", va="center", transform=ax.transAxes)

            if row_idx == 0:
                ax.set_title(band_name, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.text(-0.18, 0.5, row_label, rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("EEG Topographic Maps: LR -> HR -> SR", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_timeseries(eeg_dict, sr_eeg, fs, save_path, n_channels=10, duration_sec=0.5):
    time_steps = eeg_dict["64ch"].shape[1]
    requested = int(duration_sec * fs)
    n_samples = min(requested, time_steps)
    start_idx = 0 if n_samples >= time_steps else np.random.randint(0, max(1, time_steps - n_samples))
    time_axis = np.arange(n_samples) / fs * 1000

    rng = np.random.default_rng(42)
    selected_64 = rng.choice(64, min(n_channels, 64), replace=False)
    selected_128 = rng.choice(128, min(n_channels, 128), replace=False)
    selected_256 = rng.choice(256, min(n_channels, 256), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_channels, 10)))

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("EEG Time-Series Comparison: LR -> HR -> SR", fontsize=16, fontweight="bold")

    series = [
        (axes[0], eeg_dict["64ch"], selected_64, "64-Channel LR EEG (Input)", "#E63946"),
        (axes[1], eeg_dict["128ch"], selected_128, "128-Channel HR EEG (Intermediate)", "#457B9D"),
        (axes[2], sr_eeg, selected_256, "256-Channel SR EEG (Model Output)", "#2A9D8F"),
    ]

    for ax, eeg, channels, title, fallback_color in series:
        for idx, ch in enumerate(channels[:n_channels]):
            signal = eeg[ch, start_idx:start_idx + n_samples]
            offset = idx * 4
            color = colors[idx % len(colors)] if len(colors) else fallback_color
            ax.plot(time_axis, signal + offset, color=color, linewidth=1.2, alpha=0.85)
        ax.set_ylabel("Amplitude\n(normalized)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_metrics(pred, target):
    pred_np = np.asarray(pred, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)

    pred_flat = pred_np.reshape(-1)
    target_flat = target_np.reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()

    numerator = float(np.sum(pred_centered * target_centered))
    pred_std = float(np.sqrt(np.sum(pred_centered ** 2) + 1e-8))
    target_std = float(np.sqrt(np.sum(target_centered ** 2) + 1e-8))
    pcc = numerator / (pred_std * target_std + 1e-8)

    mse = float(np.mean((pred_np - target_np) ** 2))
    target_power = float(np.mean(target_np ** 2))
    nmse = mse / (target_power + 1e-8)
    snr = 10.0 * np.log10(target_power / (mse + 1e-8))
    mae = float(np.mean(np.abs(pred_np - target_np)))

    return {"pcc": float(pcc), "nmse": float(nmse), "snr": float(snr), "mae": float(mae)}


def evaluate_stad(
    checkpoint_path,
    data_path,
    output_dir,
    n_samples,
    preprocessed,
    metrics_history_path,
    subjects="all",
    eval_mode="validation_like",
):
    output_dir = ensure_dir(Path(output_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Localize-MI STAD Test / Visualization")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data path: {data_path}")
    print(f"Eval mode: {eval_mode}")
    if subjects == "all":
        print("Subjects: all")
    else:
        print(f"Subjects ({len(subjects)}): {subjects}")

    dataset = LocalizeMISTADDataset(
        data_path,
        subjects=subjects,
        lr_channels=64,
        hr_channels=128,
        sr_channels=256,
        time_len=2080,
        preprocessed=preprocessed,
    )

    samples, sample_indices = get_samples(dataset, n_samples)
    print(f"Loaded {len(dataset)} available samples; evaluating {len(samples)} of them.")

    # Load checkpoint first to extract model config
    print("Loading checkpoint to read model config...")
    checkpoint_raw = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = checkpoint_raw.get("config", {})
    state_dict = checkpoint_raw.get("model", checkpoint_raw)
    
    # Infer latent_dim from actual checkpoint weight shape (most reliable)
    # Check stc.pos_embed shape: [1, 64, 65, latent_dim]
    if "stc.pos_embed" in state_dict:
        latent_dim = state_dict["stc.pos_embed"].shape[-1]
    else:
        latent_dim = cfg.get("latent_dim", 1024)
    
    # Extract model config; fall back to defaults if missing
    mae_embed_dim = latent_dim  # mae_embed_dim should match inferred latent_dim
    mae_time_len = cfg.get("mae_time_len", 256)
    mae_patch_size = cfg.get("mae_patch_size", 16)
    mae_depth = cfg.get("mae_depth", 12)
    mae_num_heads = cfg.get("mae_num_heads", 8)
    mae_decoder_embed_dim = cfg.get("mae_decoder_embed_dim", 256)
    mae_decoder_depth = cfg.get("mae_decoder_depth", 8)
    mae_decoder_num_heads = cfg.get("mae_decoder_num_heads", 8)
    mae_mlp_ratio = cfg.get("mae_mlp_ratio", 2.0)
    
    print(f"Model config: latent_dim={latent_dim} (inferred from weights), mae_embed_dim={mae_embed_dim}")

    model = STAD_LocalizeMI(
        lr_channels=64,
        hr_channels=128,
        sr_channels=256,
        seq_len=2080,
        latent_dim=latent_dim,
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

    checkpoint, missing_keys, unexpected_keys = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    if missing_keys:
        print(f"Missing checkpoint keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected checkpoint keys: {len(unexpected_keys)}")

    T = 1000
    betas = get_beta_schedule(T).to(device)
    diff_params = get_diffusion_params(betas)
    diff_params["sqrt_alphas_cumprod"] = torch.clamp(diff_params["sqrt_alphas_cumprod"], min=1e-8)
    diff_params["sqrt_one_minus_alphas_cumprod"] = torch.clamp(diff_params["sqrt_one_minus_alphas_cumprod"], min=1e-8)

    all_metrics = []
    all_lr = []
    all_hr = []
    all_sr = []

    for item_idx, sample in enumerate(samples, start=1):
        print(f"Evaluating sample {item_idx}/{len(samples)}")
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).cpu().numpy()[0]

        sr_gt_tensor = sample["sr"].unsqueeze(0).to(device)
        sr_gt = sr_gt_tensor.cpu().numpy()[0]

        if eval_mode == "validation_like":
            sr_pred = reconstruct_like_training_validation(
                model,
                lr,
                sr_gt_tensor,
                device,
                diff_params,
                timesteps=T,
            )
        elif eval_mode == "ddim":
            sr_pred = ddim_generate_sr(model, lr, device, diff_params, timesteps=T, ddim_steps=50)
        else:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

        sr_pred_np = sr_pred.detach().cpu().numpy()[0]

        metrics = compute_metrics(sr_pred_np, sr_gt)
        all_metrics.append(metrics)
        all_lr.append(sample["lr"].cpu().numpy())
        all_hr.append(hr)
        all_sr.append(sr_pred_np)

        print(
            "  PCC={pcc:.4f} NMSE={nmse:.6f} SNR={snr:.2f} dB MAE={mae:.4f}".format(
                **metrics
            )
        )

        eeg_dict = {
            "64ch": sample["lr"].cpu().numpy(),
            "128ch": sample["hr"].cpu().numpy(),
        }
        create_multireso_topomap(
            eeg_dict,
            sr_pred_np,
            fs=8000,
            save_path=output_dir / f"topomap_sample_{item_idx}.png",
        )
        visualize_timeseries(
            eeg_dict,
            sr_pred_np,
            fs=8000,
            save_path=output_dir / f"timeseries_sample_{item_idx}.png",
        )
        plot_band_power_summary(
            eeg_dict,
            sr_pred_np,
            fs=8000,
            save_path=output_dir / f"band_power_sample_{item_idx}.png",
        )

    avg_metrics = {
        key: float(np.mean([metric[key] for metric in all_metrics])) for key in all_metrics[0].keys()
    }
    std_metrics = {
        key: float(np.std([metric[key] for metric in all_metrics])) for key in all_metrics[0].keys()
    }

    summary = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "subjects": subjects,
        "sample_indices": sample_indices,
        "individual_samples": all_metrics,
        "average_metrics": avg_metrics,
        "std_metrics": std_metrics,
    }

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.save(output_dir / "metrics_summary.npy", summary)

    if metrics_history_path and Path(metrics_history_path).exists():
        metrics_history = np.load(metrics_history_path, allow_pickle=True).tolist()
        plot_training_history(metrics_history, output_dir / "training_history.png")

    plot_metric_summary(avg_metrics, output_dir / "metric_summary.png")

    if all_sr:
        avg_eeg_dict = {
            "64ch": np.mean(np.stack(all_lr, axis=0), axis=0),
            "128ch": np.mean(np.stack(all_hr, axis=0), axis=0),
        }
        avg_sr = np.mean(np.stack(all_sr, axis=0), axis=0)
        create_multireso_topomap(avg_eeg_dict, avg_sr, fs=8000, save_path=output_dir / "topomap_average.png")
        visualize_timeseries(avg_eeg_dict, avg_sr, fs=8000, save_path=output_dir / "timeseries_average.png")
        plot_band_power_summary(avg_eeg_dict, avg_sr, fs=8000, save_path=output_dir / "band_power_average.png")

    print("=" * 60)
    print("Evaluation complete")
    print(f"Results saved to: {output_dir}")
    print(
        "Average metrics: PCC={pcc:.4f}, NMSE={nmse:.6f}, SNR={snr:.2f} dB, MAE={mae:.4f}".format(
            **avg_metrics
        )
    )
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Test STAD on Localize-MI and generate graphs/topomaps")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SCRIPT_DIR / "best_stad_localizemi.pt"),
        help="Path to the trained STAD checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(REPO_DIR / "DATA/Localize-MI/derivatives/epochs_prc1"),
        help="Path to Localize-MI data folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "test_results_localizemi"),
        help="Directory to store graphs and topomaps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--raw_epochs",
        action="store_true",
        help="Use raw *_epochs.npy data instead of preprocessed X_prc1.npy",
    )
    parser.add_argument(
        "--metrics_history",
        type=str,
        default=str(SCRIPT_DIR / "metrics_history_localizemi.npy"),
        help="Optional path to training metrics_history.npy for plotting curves",
    )
    parser.add_argument(
        "--mae_results_dir",
        type=str,
        default=None,
        help="Optional MAE k-fold directory containing fold_splits.json",
    )
    parser.add_argument(
        "--mae_fold",
        type=int,
        default=None,
        help="Fold index from fold_splits.json to select subjects from",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["val", "train"],
        default="val",
        help="Which fold split subjects to evaluate when --mae_fold is set",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["validation_like", "ddim"],
        default="validation_like",
        help="validation_like matches training validation; ddim runs free generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocessed = not args.raw_epochs

    subjects = "all"
    if args.mae_fold is not None:
        if not args.mae_results_dir:
            raise ValueError("--mae_results_dir is required when --mae_fold is set")
        train_subjects, val_subjects = load_fold_subject_split(args.mae_results_dir, args.mae_fold)
        subjects = val_subjects if args.eval_split == "val" else train_subjects

    evaluate_stad(
        checkpoint_path=Path(args.checkpoint),
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples,
        preprocessed=preprocessed,
        metrics_history_path=Path(args.metrics_history),
        subjects=subjects,
        eval_mode=args.eval_mode,
    )


if __name__ == "__main__":
    main()