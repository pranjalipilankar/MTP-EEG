#!/usr/bin/env python3
"""Evaluate STAD checkpoint on SEED-IV preprocessed data with enhanced visualizations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from scipy.fft import fft

try:
    import mne
    from mne.time_frequency import psd_array_multitaper
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("Warning: MNE not available. Topomap visualizations disabled.")

from mae_for_eeg import MAEforEEG
from stad_model_CORRECT import STADModel


def get_seed4_channel_indices(target_channels):
    """Fixed channel subsets used in training."""
    if target_channels == 62:
        return np.arange(62, dtype=int)
    if target_channels == 31:
        return np.array([
            0, 2, 4, 5, 7, 9, 11, 13,
            15, 17, 19, 21, 23, 25, 27, 29,
            31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 58, 60,
        ], dtype=int)
    if target_channels == 16:
        return np.array([
            0, 2, 5, 7, 9, 13, 17, 21,
            23, 27, 31, 35, 39, 45, 53, 60,
        ], dtype=int)
    return np.linspace(0, 61, target_channels, dtype=int)


def compute_sr_metrics(pred_sr, target_sr, eps=1e-8):
    """Compute PCC, NMSE, and SNR in dB."""
    pred = pred_sr.reshape(pred_sr.shape[0], -1)
    target = target_sr.reshape(target_sr.shape[0], -1)

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered.pow(2).sum(dim=1) + eps) *
        (target_centered.pow(2).sum(dim=1) + eps)
    )
    pcc = (numerator / denominator).mean().item()

    mse = (pred - target).pow(2).mean(dim=1)
    signal_power = target.pow(2).mean(dim=1)
    nmse = (mse / (signal_power + eps)).mean().item()
    snr = (10.0 * torch.log10((signal_power + eps) / (mse + eps))).mean().item()

    return {
        'pcc': pcc,
        'nmse': nmse,
        'snr': snr,
    }


# ============================================================================
# ENHANCED: Dataset Statistics & Validation
# ============================================================================

def compute_dataset_statistics(dataset, sample_size=None):
    """Compute comprehensive dataset statistics for validation."""
    if sample_size is None:
        sample_size = min(100, len(dataset))
    
    sr_samples = []
    lr_samples = []
    hr_samples = []
    signal_ranges = []
    
    for i in range(sample_size):
        batch = dataset[i]
        sr_samples.append(batch['sr'].numpy())
        lr_samples.append(batch['lr'].numpy())
        hr_samples.append(batch['hr'].numpy())
        signal_ranges.append(batch['sr'].numpy().max() - batch['sr'].numpy().min())
    
    sr_data = np.concatenate(sr_samples, axis=0)
    lr_data = np.concatenate(lr_samples, axis=0)
    hr_data = np.concatenate(hr_samples, axis=0)
    
    stats = {
        'sr': {
            'shape': sr_data.shape,
            'mean': sr_data.mean(),
            'std': sr_data.std(),
            'min': sr_data.min(),
            'max': sr_data.max(),
            'channels': sr_data.shape[0],
        },
        'lr': {
            'shape': lr_data.shape,
            'mean': lr_data.mean(),
            'std': lr_data.std(),
            'min': lr_data.min(),
            'max': lr_data.max(),
            'channels': lr_data.shape[0],
        },
        'hr': {
            'shape': hr_data.shape,
            'mean': hr_data.mean(),
            'std': hr_data.std(),
            'min': hr_data.min(),
            'max': hr_data.max(),
            'channels': hr_data.shape[0],
        },
        'total_samples': len(dataset),
        'sampled_count': sample_size,
        'signal_range_mean': np.mean(signal_ranges),
        'signal_range_std': np.std(signal_ranges),
    }
    
    return stats


def print_dataset_report(stats):
    """Print formatted dataset statistics report."""
    print('\n' + '='*80)
    print('DATASET STATISTICS REPORT')
    print('='*80)
    print(f"Total Samples: {stats['total_samples']} (analyzed: {stats['sampled_count']})")
    print(f"Signal Range: {stats['signal_range_mean']:.4f} ± {stats['signal_range_std']:.4f}")
    
    for key in ['lr', 'hr', 'sr']:
        d = stats[key]
        print(f"\n{key.upper()} EEG:")
        print(f"  Shape: {d['shape']}")
        print(f"  Channels: {d['channels']}")
        print(f"  Mean: {d['mean']:.6f}, Std: {d['std']:.6f}")
        print(f"  Range: [{d['min']:.6f}, {d['max']:.6f}]")


# ============================================================================
# ENHANCED: Visualization Functions
# ============================================================================

def compute_psd_batch(eeg_data, fs=128, fmin=0.5, fmax=45, nperseg=256):
    """Compute power spectral density for batch of signals."""
    from scipy import signal
    
    psds = []
    for ch_data in eeg_data:
        f, psd = signal.welch(ch_data, fs=fs, nperseg=min(nperseg, len(ch_data)),
                              scaling='density')
        mask = (f >= fmin) & (f <= fmax)
        psds.append(psd[mask])
    
    return np.array(psds), f[mask]


def extract_band_power(eeg_data, fs=128, freq_ranges=None):
    """Extract power in frequency bands."""
    if freq_ranges is None:
        freq_ranges = {
            'δ (0.5-4 Hz)': (0.5, 4),
            'θ (4-8 Hz)': (4, 8),
            'α (8-13 Hz)': (8, 13),
            'β (13-30 Hz)': (13, 30),
            'γ (30-45 Hz)': (30, 45),
        }
    
    psd, freqs = compute_psd_batch(eeg_data, fs=fs)
    
    band_powers = {}
    for band_name, (fmin, fmax) in freq_ranges.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band_name] = np.mean(psd[:, mask], axis=1)
    
    return band_powers


def save_metric_distributions_plot(pcc_scores, nmse_scores, snr_scores, out_path):
    """Save histograms of evaluation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # PCC distribution
    axes[0].hist(pcc_scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(pcc_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pcc_scores):.4f}')
    axes[0].set_xlabel('PCC Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PCC Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # NMSE distribution
    axes[1].hist(nmse_scores, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(nmse_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(nmse_scores):.4f}')
    axes[1].set_xlabel('NMSE Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('NMSE Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # SNR distribution
    axes[2].hist(snr_scores, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(snr_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(snr_scores):.2f} dB')
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('SNR Distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    fig.suptitle('Evaluation Metrics Distribution', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved metrics distribution: {out_path}")


def save_loss_curves_plot(diff_losses, sr_losses, out_path):
    """Save loss curves over batches."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Filter out NaN values for diff_loss
    diff_losses_filtered = [l for l in diff_losses if not np.isnan(l)]
    
    # Diffusion loss
    if diff_losses_filtered:
        axes[0].plot(diff_losses_filtered, linewidth=1.5, color='steelblue', label='Diff Loss')
        axes[0].fill_between(range(len(diff_losses_filtered)), diff_losses_filtered, alpha=0.3, color='steelblue')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Batch')
        axes[0].set_title('Diffusion Loss Over Batches')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No diffusion loss (sampling mode)', ha='center', va='center',
                    transform=axes[0].transAxes, fontsize=12, color='gray')
    
    # SR L1 loss
    axes[1].plot(sr_losses, linewidth=1.5, color='coral', label='SR L1 Loss')
    axes[1].fill_between(range(len(sr_losses)), sr_losses, alpha=0.3, color='coral')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Batch')
    axes[1].set_title('SR L1 Loss Over Batches')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    fig.suptitle('Training/Evaluation Curves', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved loss curves: {out_path}")


def save_comparison_plot(pred_sr, target_sr, channels_to_plot, out_path):
    """Save side-by-side comparison of predictions vs targets."""
    n_samples = min(3, pred_sr.shape[0])
    n_channels = len(channels_to_plot)
    
    fig, axes = plt.subplots(n_samples, n_channels, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(n_samples):
        for ch_idx, ch in enumerate(channels_to_plot):
            ax = axes[sample_idx, ch_idx]
            
            pred_signal = pred_sr[sample_idx, ch, :]
            target_signal = target_sr[sample_idx, ch, :]
            
            ax.plot(target_signal, label='Target', linewidth=1.5, color='black', alpha=0.7)
            ax.plot(pred_signal, label='Predicted', linewidth=1.5, color='steelblue', alpha=0.7)
            
            ax.set_title(f'Sample {sample_idx}, Ch {ch}')
            ax.set_ylabel('Amplitude (μV)')
            ax.grid(alpha=0.3)
            if sample_idx == 0 and ch_idx == 0:
                ax.legend(loc='upper right')
    
    axes[-1, 0].set_xlabel('Time Samples')
    fig.suptitle('Predicted vs Target EEG Signals', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison plot: {out_path}")


# Topomap-specific functions
def save_topomap_visualization(pred_sr, target_sr, out_path, n_channels=62):
    """Generate and save topomap visualizations if MNE available."""
    if not HAS_MNE:
        print("Skipping topomap visualization (MNE not available)")
        return
    
    try:
        # Use first sample
        pred_sample = pred_sr[0] if pred_sr.ndim > 2 else pred_sr
        target_sample = target_sr[0] if target_sr.ndim > 2 else target_sr
        
        if pred_sample.ndim == 2:
            pred_sample = pred_sample.mean(axis=1)
        if target_sample.ndim == 2:
            target_sample = target_sample.mean(axis=1)
        
        # Create mock info
        ch_names = [f'E{i+1}' for i in range(n_channels)]
        fs = 128
        info = mne.create_info(ch_names, fs, ch_types='eeg')
        
        # Generate synthetic electrode positions (circle)
        angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
        radius = 0.5
        pos_x = radius * np.cos(angles)
        pos_y = radius * np.sin(angles)
        pos_z = np.zeros(n_channels)
        
        pos_dict = {ch: np.array([x, y, z]) 
                   for ch, x, y, z in zip(ch_names, pos_x, pos_y, pos_z)}
        montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')
        info.set_montage(montage)
        
        # Create comparison figure
        fig = plt.figure(figsize=(12, 5))
        
        # Target topomap
        ax1 = plt.subplot(1, 2, 1)
        im1, _ = mne.viz.plot_topomap(target_sample, info, axes=ax1, show=False,
                                      cmap='RdBu_r', contours=6, outlines='head',
                                      sensors=True, vmin=target_sample.min(), 
                                      vmax=target_sample.max())
        ax1.set_title('Target SR (Ground Truth)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Amplitude (μV)')
        
        # Predicted topomap
        ax2 = plt.subplot(1, 2, 2)
        im2, _ = mne.viz.plot_topomap(pred_sample, info, axes=ax2, show=False,
                                      cmap='RdBu_r', contours=6, outlines='head',
                                      sensors=True, vmin=target_sample.min(),
                                      vmax=target_sample.max())
        ax2.set_title('Predicted SR (Model Output)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Amplitude (μV)')
        
        fig.suptitle('EEG Topographic Map Comparison', fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved topomap visualization: {out_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to generate topomap: {e}")



def create_split(n_folds=5, test_fold=0):
    """Reproduce split logic from training script."""
    all_subjects = [str(i) for i in range(1, 16)]

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2024)

    splits = list(kf.split(all_subjects))
    train_val_idx, test_idx = splits[test_fold]

    val_size = len(train_val_idx) // 5
    val_idx = train_val_idx[:val_size]
    train_idx = train_val_idx[val_size:]

    return {
        'train': [all_subjects[i] for i in train_idx],
        'val': [all_subjects[i] for i in val_idx],
        'test': [all_subjects[i] for i in test_idx],
    }


class SEED4PreprocessedDataset(Dataset):
    """Loader for preprocessed SEED-IV folder structure."""

    def __init__(self, data_path, subjects=None, lr_channels=16, hr_channels=31):
        self.lr_indices = get_seed4_channel_indices(lr_channels)
        self.hr_indices = get_seed4_channel_indices(hr_channels)

        data_path = Path(data_path)

        if data_path.is_file() and data_path.suffix == '.npz':
            self._load_from_npz(data_path, subjects=subjects)
            return

        all_windows = []
        all_subject_ids = []
        all_session_ids = []
        all_trial_ids = []
        all_norm_stats = []
        prc1_meta = None

        for subject_id in subjects:
            for session in ['1', '2', '3']:
                session_path = data_path / session
                if not session_path.exists():
                    continue

                subject_folders = list(session_path.glob(f'{subject_id}_*'))
                for folder in subject_folders:
                    x_file = folder / 'X_prc1.npy'
                    stats_file = folder / 'X_prc1_norm_stats.npy'
                    meta_file = folder / 'prc1_meta.json'
                    trial_labels_file = folder / 'trial_labels.json'
                    if x_file.exists():
                        x_data = np.load(x_file)
                        trial_ids = None
                        if trial_labels_file.exists():
                            with open(trial_labels_file, 'r', encoding='utf-8') as f:
                                trial_meta = json.load(f)
                            windows_per_trial = trial_meta.get('windows_per_trial', None)
                            if windows_per_trial is not None:
                                trial_seq = []
                                for tid, n_w in enumerate(windows_per_trial):
                                    trial_seq.extend([tid] * int(n_w))
                                if len(trial_seq) == len(x_data):
                                    trial_ids = np.array(trial_seq, dtype=int)

                        if trial_ids is None:
                            # Fallback: keep IDs in valid SEED-IV range.
                            trial_ids = np.arange(len(x_data), dtype=int) % 24
                        all_windows.append(x_data)
                        all_subject_ids.extend([str(subject_id)] * len(x_data))
                        all_session_ids.extend([str(session)] * len(x_data))
                        all_trial_ids.extend(trial_ids.tolist())
                        if stats_file.exists():
                            s_data = np.load(stats_file)
                            if len(s_data) == len(x_data):
                                all_norm_stats.append(s_data)
                        if prc1_meta is None and meta_file.exists():
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                prc1_meta = json.load(f)

        if not all_windows:
            raise ValueError(f"No preprocessed data found for subjects {subjects} at {data_path}")

        all_windows = np.concatenate(all_windows, axis=0).astype(np.float32)
        self.sr_samples = all_windows
        self.hr_samples = all_windows[:, self.hr_indices, :]
        self.lr_samples = all_windows[:, self.lr_indices, :]
        self.subject_ids = np.array(all_subject_ids)
        self.session_ids = np.array(all_session_ids)
        self.trial_ids = np.array(all_trial_ids, dtype=int)
        self.prc1_meta = prc1_meta
        if all_norm_stats and sum(len(s) for s in all_norm_stats) == len(self.sr_samples):
            self.norm_stats = np.concatenate(all_norm_stats, axis=0).astype(np.float32)
        else:
            self.norm_stats = None

        print(f"Loaded {len(self.sr_samples)} test windows from subjects {subjects}")

    def _load_from_npz(self, npz_path, subjects=None):
        """Load test data from npz payload (supports multiple key layouts)."""
        payload = np.load(npz_path, allow_pickle=True)

        if 'SR' in payload:
            sr_all = payload['SR'].astype(np.float32)
            if 'test_indices' in payload:
                base_idx = payload['test_indices'].astype(int)
            else:
                base_idx = np.arange(len(sr_all), dtype=int)
            sr = sr_all[base_idx]
        elif 'X_test' in payload:
            sr = payload['X_test'].astype(np.float32)
            base_idx = np.arange(len(sr), dtype=int)
        else:
            raise KeyError(
                f"Unsupported npz format: {npz_path}. Expected SR (+optional test_indices) or X_test key."
            )

        n = len(sr)
        if 'subject_ids' in payload:
            subject_all = np.asarray(payload['subject_ids']).astype(str)
            if len(subject_all) == len(base_idx):
                subject_ids = subject_all
            elif len(subject_all) > np.max(base_idx):
                subject_ids = subject_all[base_idx]
            elif len(subject_all) >= n:
                subject_ids = subject_all[:n]
            else:
                subject_ids = np.array([f'unknown_{i:05d}' for i in range(n)])
        else:
            subject_ids = np.array([f'unknown_{i:05d}' for i in range(n)])

        if 'session_ids' in payload:
            session_all = np.asarray(payload['session_ids']).astype(str)
            if len(session_all) == len(base_idx):
                session_ids = session_all
            elif len(session_all) > np.max(base_idx):
                session_ids = session_all[base_idx]
            elif len(session_all) >= n:
                session_ids = session_all[:n]
            else:
                session_ids = np.array(['1'] * n)
        else:
            session_ids = np.array(['1'] * n)

        if 'trial_ids' in payload:
            trial_all = np.asarray(payload['trial_ids'])
            if len(trial_all) == len(base_idx):
                trial_ids = trial_all.astype(int)
            elif len(trial_all) > np.max(base_idx):
                trial_ids = trial_all[base_idx].astype(int)
            elif len(trial_all) >= n:
                trial_ids = trial_all[:n].astype(int)
            else:
                trial_ids = np.arange(n, dtype=int)
        else:
            if 'labels' in payload:
                labels_all = np.asarray(payload['labels'])
                if len(labels_all) == len(base_idx):
                    labels = labels_all
                elif len(labels_all) > np.max(base_idx):
                    labels = labels_all[base_idx]
                elif len(labels_all) >= n:
                    labels = labels_all[:n]
                else:
                    labels = None

                if labels is not None and len(labels) == n:
                    trial_ids = np.zeros(n, dtype=int)
                    running_trial = {}
                    last_label = {}
                    for i in range(n):
                        subj = str(subject_ids[i])
                        lab = int(labels[i])
                        if subj not in running_trial:
                            running_trial[subj] = 0
                            last_label[subj] = lab
                            trial_ids[i] = 0
                            continue
                        if lab != last_label[subj]:
                            running_trial[subj] += 1
                            last_label[subj] = lab
                        trial_ids[i] = running_trial[subj]
                    print(
                        "Info: trial_ids missing in npz; inferred trial boundaries from label transitions "
                        "within each subject."
                    )
                else:
                    trial_ids = np.arange(n, dtype=int)
            else:
                trial_ids = np.arange(n, dtype=int)

        if subjects is not None:
            wanted = np.array([str(s) for s in subjects])
            mask = np.isin(subject_ids, wanted)
            if np.any(mask):
                sr = sr[mask]
                subject_ids = subject_ids[mask]
                session_ids = session_ids[mask]
                trial_ids = trial_ids[mask]
                print(
                    f"Loaded {len(sr)} test windows from npz after subject filtering "
                    f"({len(wanted)} subjects): {npz_path}"
                )
            else:
                available = sorted(np.unique(subject_ids).tolist())[:20]
                raise ValueError(
                    f"No npz samples matched requested test subjects {subjects}. "
                    f"Available subject_ids examples: {available}"
                )
        else:
            print(f"Loaded {len(sr)} test windows from npz: {npz_path}")

        self.sr_samples = sr
        self.hr_samples = sr[:, self.hr_indices, :]
        self.lr_samples = sr[:, self.lr_indices, :]
        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.trial_ids = trial_ids.astype(int)
        self.norm_stats = None
        self.prc1_meta = None

    def __len__(self):
        return len(self.sr_samples)

    def __getitem__(self, idx):
        return {
            'lr': torch.from_numpy(self.lr_samples[idx]).float(),
            'hr': torch.from_numpy(self.hr_samples[idx]).float(),
            'sr': torch.from_numpy(self.sr_samples[idx]).float(),
            'subject_id': self.subject_ids[idx],
            'session_id': self.session_ids[idx],
            'trial_id': self.trial_ids[idx],
        }


def build_mae_encoder(mae_checkpoint_path, device):
    """Load 31-channel MAE model used by STAD."""
    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu', weights_only=False)

    mae_model = MAEforEEG(
        time_len=1000,
        patch_size=8,
        embed_dim=768,
        in_chans=31,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
    )

    if 'model_state_dict' in checkpoint:
        mae_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        mae_model.load_state_dict(checkpoint['model'])
    else:
        mae_model.load_state_dict(checkpoint)

    mae_model.to(device)
    mae_model.eval()
    for p in mae_model.parameters():
        p.requires_grad = False

    return mae_model


def save_eeg_signal_figure(pred_sr, target_sr, lr_eeg, out_path, sample_idx, channels_to_plot):
    """Save overlay plots of predicted vs target EEG signals for selected channels."""
    pred_np = pred_sr[sample_idx].detach().cpu().numpy()
    target_np = target_sr[sample_idx].detach().cpu().numpy()
    lr_np = lr_eeg[sample_idx].detach().cpu().numpy()

    n_rows = len(channels_to_plot)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for row, ch_idx in enumerate(channels_to_plot):
        ax = axes[row]
        ax.plot(target_np[ch_idx], color='black', linewidth=1.0, label='Target SR (62ch)')
        ax.plot(pred_np[ch_idx], color='tab:blue', linewidth=0.9, alpha=0.85, label='Predicted SR')

        if ch_idx < lr_np.shape[0]:
            ax.plot(lr_np[ch_idx], color='tab:orange', linewidth=0.8, alpha=0.65, label='LR input (if mapped)')

        ax.set_ylabel(f"Ch {ch_idx}")
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=8)

    axes[-1].set_xlabel('Time samples')
    fig.suptitle(f'EEG Signals: sample {sample_idx}', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _npz_has_built_in_test_split(npz_path):
    """Return True if NPZ already defines test subset or explicit test tensor."""
    try:
        payload = np.load(npz_path, allow_pickle=True)
        return ('test_indices' in payload) or ('X_test' in payload)
    except Exception:
        return False


def save_grouped_outputs(pred_sr, target_sr, subject_ids, session_ids, trial_ids, output_root):
    """Save outputs grouped by session/subject/trial in a raw-data-like hierarchy."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    groups = {}
    for idx, (subj, sess, trial) in enumerate(zip(subject_ids, session_ids, trial_ids)):
        key = (str(sess), str(subj), int(trial))
        groups.setdefault(key, []).append(idx)

    for (sess, subj, trial), indices in groups.items():
        trial_dir = output_root / str(sess) / f'subject_{subj}' / f'trial_{int(trial):02d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        idx_arr = np.asarray(indices, dtype=int)
        np.save(trial_dir / 'pred_sr.npy', pred_sr[idx_arr])
        np.save(trial_dir / 'target_sr.npy', target_sr[idx_arr])
        np.savez(
            trial_dir / 'meta.npz',
            indices=idx_arr,
            subject_id=np.array([subj]),
            session_id=np.array([sess]),
            trial_id=np.array([int(trial)], dtype=int),
        )

    print(
        f"Saved grouped outputs for {len(groups)} session/subject/trial groups -> {output_root}"
    )


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = args.data_path
    # Common naming mismatch: use preprocessed_data.npz if eeg_processed_data.npz was provided.
    if data_path.endswith('eeg_processed_data.npz') and not Path(data_path).exists():
        fallback = data_path.replace('eeg_processed_data.npz', 'preprocessed_data.npz')
        if Path(fallback).exists():
            print(f"Info: {data_path} not found, using {fallback}")
            data_path = fallback

    test_subjects = None
    use_fold_split = True
    data_path_obj = Path(data_path)
    if data_path_obj.is_file() and data_path_obj.suffix == '.npz':
        has_npz_test_split = _npz_has_built_in_test_split(data_path_obj)
        if has_npz_test_split and not args.force_fold_subject_filter:
            use_fold_split = False
            print('Using NPZ-provided test subset directly (no fold subject filtering).')
        elif not has_npz_test_split:
            if args.allow_potential_leakage:
                use_fold_split = False
                print('Warning: NPZ has no explicit test split; evaluating full NPZ (potential leakage).')
            else:
                use_fold_split = True
                print('NPZ has no explicit test split; using fold subject filter to avoid leakage.')

    if use_fold_split:
        splits = create_split(n_folds=5, test_fold=args.test_fold)
        test_subjects = splits['test']
        print(f"Test subjects: {test_subjects}")
    else:
        print('Test subjects: inferred from NPZ test payload')

    dataset = SEED4PreprocessedDataset(
        data_path=data_path,
        subjects=test_subjects,
        lr_channels=16,
        hr_channels=31,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ✅ ENHANCED: Print dataset statistics
    print("\n" + "="*80)
    print("Computing Dataset Statistics...")
    print("="*80)
    dataset_stats = compute_dataset_statistics(dataset, sample_size=min(50, len(dataset)))
    print_dataset_report(dataset_stats)


    mae_encoder = build_mae_encoder(args.mae_checkpoint, device)

    model = STADModel(
        mae_encoder=mae_encoder,
        lr_channels=16,
        hr_channels=31,
        sr_channels=62,
        latent_dim=768,
        num_patches=125,
        diffusion_schedule=args.diffusion_schedule,
        lr_channel_indices=dataset.lr_indices,
        device=device,
    ).to(device)

    ckpt = torch.load(args.stad_checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading STAD checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading STAD checkpoint: {len(unexpected)}")

    model.eval()

    fig_dir = None
    saved_figures = 0
    if args.save_fig_dir:
        fig_dir = Path(args.save_fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

    channels_to_plot = [0, 7, 15, 23, 31, 45, 61]

    diff_losses = []
    sr_losses = []
    pcc_scores = []
    nmse_scores = []
    snr_scores = []
    saved_pred_sr = []
    saved_target_sr = []
    saved_subject_ids = []
    saved_session_ids = []
    saved_trial_ids = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Testing')):
            lr_eeg = batch['lr'].to(device)
            hr_eeg = batch['hr'].to(device)
            sr_eeg = batch['sr'].to(device)
            subject_ids = list(batch['subject_id'])
            session_ids = list(batch['session_id'])
            trial_ids = batch['trial_id'].cpu().numpy().astype(int).tolist()

            if args.use_sampling:
                pred_sr = model.sample_sr(lr_eeg, num_inference_steps=args.num_inference_steps)
                diff_loss = torch.tensor(float('nan'), device=device)
            else:
                diff_loss, pred_sr = model(lr_eeg, hr_eeg, sr_eeg)

            sr_loss = F.l1_loss(pred_sr.float(), sr_eeg.float())
            metrics = compute_sr_metrics(pred_sr.float(), sr_eeg.float())

            saved_pred_sr.append(pred_sr.detach().cpu().numpy())
            saved_target_sr.append(sr_eeg.detach().cpu().numpy())
            saved_subject_ids.extend(subject_ids)
            saved_session_ids.extend(session_ids)
            saved_trial_ids.extend(trial_ids)

            diff_losses.append(float(diff_loss.item()))
            sr_losses.append(float(sr_loss.item()))
            pcc_scores.append(metrics['pcc'])
            nmse_scores.append(metrics['nmse'])
            snr_scores.append(metrics['snr'])

            if fig_dir is not None and saved_figures < args.num_fig_samples:
                batch_size = pred_sr.shape[0]
                for b in range(batch_size):
                    if saved_figures >= args.num_fig_samples:
                        break
                    fig_path = fig_dir / f"eeg_signal_sample_{saved_figures:03d}.png"
                    save_eeg_signal_figure(
                        pred_sr=pred_sr,
                        target_sr=sr_eeg,
                        lr_eeg=lr_eeg,
                        out_path=fig_path,
                        sample_idx=b,
                        channels_to_plot=channels_to_plot,
                    )
                    saved_figures += 1

            if args.max_batches > 0 and (i + 1) >= args.max_batches:
                break

    mean_diff = np.nanmean(diff_losses) if diff_losses else float('nan')
    mean_sr = np.mean(sr_losses) if sr_losses else float('nan')
    mean_pcc = np.mean(pcc_scores) if pcc_scores else float('nan')
    mean_nmse = np.mean(nmse_scores) if nmse_scores else float('nan')
    mean_snr = np.mean(snr_scores) if snr_scores else float('nan')

    print('\n' + '=' * 80)
    print('Preprocessed Data Test Results')
    print('=' * 80)
    print(f"Samples tested: {len(sr_losses) * args.batch_size} (approx)")
    print(f"Diff Loss: {mean_diff:.6f}")
    print(f"SR L1 Loss: {mean_sr:.6f}")
    print(f"PCC: {mean_pcc:.4f}")
    print(f"NMSE: {mean_nmse:.4f}")
    print(f"SNR: {mean_snr:.2f} dB")
    if fig_dir is not None:
        print(f"Saved EEG figures: {saved_figures} -> {fig_dir}")

    # ✅ ENHANCED: Generate comprehensive visualizations
    print('\n' + '=' * 80)
    print('Generating Comprehensive Visualizations...')
    print('=' * 80)
    
    viz_dir = Path(args.visualization_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metric distributions
    if pcc_scores and nmse_scores and snr_scores:
        metric_dist_path = viz_dir / 'metrics_distribution.png'
        save_metric_distributions_plot(pcc_scores, nmse_scores, snr_scores, str(metric_dist_path))
    
    # Save loss curves
    loss_curves_path = viz_dir / 'loss_curves.png'
    save_loss_curves_plot(diff_losses, sr_losses, str(loss_curves_path))
    
    # Save comparison plots
    if saved_pred_sr and saved_target_sr:
        channels_to_plot_cmp = [0, 15, 31, 45, 61]
        comp_path = viz_dir / 'eeg_comparison.png'
        pred_all = np.concatenate(saved_pred_sr, axis=0)
        target_all = np.concatenate(saved_target_sr, axis=0)
        save_comparison_plot(pred_all, target_all, channels_to_plot_cmp, str(comp_path))
        
        # Save topomap
        topomap_path = viz_dir / 'topomap_comparison.png'
        save_topomap_visualization(pred_all, target_all, str(topomap_path), n_channels=62)
    
    print(f"✅ All visualizations saved to: {viz_dir}")


    if args.save_sr_output_path:
        sr_out = np.concatenate(saved_pred_sr, axis=0)
        save_path = Path(args.save_sr_output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, sr_out)
        print(f"Saved predicted SR EEG: {sr_out.shape} -> {save_path}")

        if args.save_target_output_path:
            target_out = np.concatenate(saved_target_sr, axis=0)
            target_path = Path(args.save_target_output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, target_out)
            print(f"Saved target SR EEG: {target_out.shape} -> {target_path}")

        if args.save_test_metadata_path:
            meta_path = Path(args.save_test_metadata_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                meta_path,
                subject_ids=np.array(saved_subject_ids),
                session_ids=np.array(saved_session_ids),
                trial_ids=np.array(saved_trial_ids, dtype=int),
            )
            print(f"Saved test metadata: {meta_path}")

        if args.save_norm_stats_path:
            if getattr(dataset, 'norm_stats', None) is None:
                print("Warning: norm stats unavailable for this data source; not saved.")
            else:
                stats_path = Path(args.save_norm_stats_path)
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(stats_path, dataset.norm_stats)
                print(f"Saved norm stats: {dataset.norm_stats.shape} -> {stats_path}")

        if args.save_prc1_meta_path:
            if getattr(dataset, 'prc1_meta', None) is None:
                print("Warning: PrC-1 meta unavailable for this data source; not saved.")
            else:
                meta_json_path = Path(args.save_prc1_meta_path)
                meta_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(meta_json_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset.prc1_meta, f, indent=2)
                print(f"Saved PrC-1 meta: {meta_json_path}")

    if args.save_grouped_output_dir:
        if not saved_pred_sr or not saved_target_sr:
            print('Warning: no predictions collected, grouped output not saved.')
        else:
            pred_group = np.concatenate(saved_pred_sr, axis=0)
            target_group = np.concatenate(saved_target_sr, axis=0)
            save_grouped_outputs(
                pred_sr=pred_group,
                target_sr=target_group,
                subject_ids=np.array(saved_subject_ids),
                session_ids=np.array(saved_session_ids),
                trial_ids=np.array(saved_trial_ids, dtype=int),
                output_root=args.save_grouped_output_dir,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test STAD on SEED-IV preprocessed data')
    parser.add_argument('--data_path', type=str,
                        default='/home/ab_students/EEG-MTP/DATA/seed4/eeg_processed_data',
                        help='Path to preprocessed SEED-IV folder OR npz file (SR/X_test)')
    parser.add_argument('--mae_checkpoint', type=str, required=True,
                        help='Path to pretrained MAE checkpoint')
    parser.add_argument('--stad_checkpoint', type=str,
                        default='/home/ab_students/EEG-MTP/New_SEED4/results_stad/best_stad_model.pth',
                        help='Path to trained STAD checkpoint')
    parser.add_argument('--test_fold', type=int, default=0,
                        help='Fold index used for test split (same as training)')
    parser.add_argument('--force_fold_subject_filter', action='store_true',
                        help='Force fold-based subject filtering even for npz files with test_indices/X_test')
    parser.add_argument('--allow_potential_leakage', action='store_true',
                        help='Allow evaluating full npz when no explicit test split exists (not recommended)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--diffusion_schedule', type=str, default='cosine', choices=['linear', 'cosine'],
                        help='Diffusion beta schedule used in STAD model')
    parser.add_argument('--use_sampling', action='store_true',
                        help='Use iterative sampling (slow, more realistic inference)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Sampling steps when --use_sampling is enabled')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='If >0, stop early after this many batches (debug)')
    parser.add_argument('--save_fig_dir', type=str, default='test_figures_preprocessed',
                        help='Directory to save EEG signal figures (empty to disable)')
    parser.add_argument('--num_fig_samples', type=int, default=3,
                        help='How many samples to plot as EEG figures')
    parser.add_argument('--visualization_dir', type=str, default='test_visualizations',
                        help='Directory to save comprehensive visualizations (metrics, loss curves, topomaps)')
    parser.add_argument('--save_sr_output_path', type=str, default='',
                        help='Optional path to save predicted SR EEG windows (.npy)')
    parser.add_argument('--save_target_output_path', type=str, default='',
                        help='Optional path to save target SR EEG windows (.npy)')
    parser.add_argument('--save_test_metadata_path', type=str, default='',
                        help='Optional path to save test metadata (.npz with subject_ids/session_ids)')
    parser.add_argument('--save_norm_stats_path', type=str, default='',
                        help='Optional path to save PrC-1 norm stats (.npy) for reconstruction reversal')
    parser.add_argument('--save_prc1_meta_path', type=str, default='',
                        help='Optional path to save PrC-1 meta (.json) for reconstruction reversal')
    parser.add_argument('--save_grouped_output_dir', type=str, default='',
                        help='Optional root dir to save outputs grouped by session/subject/trial')

    evaluate(parser.parse_args())
