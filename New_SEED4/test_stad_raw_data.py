#!/usr/bin/env python3
"""
Enhanced STAD evaluation on SEED-IV raw_data.npz with:
- Multi-resolution topomaps (LR/HR/SR)
- 10-sample visualization plots
- Subject-wise output saving
- Data leakage prevention via fold-based subject filtering
"""

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


class RawDataNPZDataset(Dataset):
    """Loader for raw_data.npz format with LR/HR/SR keys."""

    def __init__(self, npz_path, test_subjects=None, lr_channels=16, hr_channels=31):
        self.lr_indices = get_seed4_channel_indices(lr_channels)
        self.hr_indices = get_seed4_channel_indices(hr_channels)
        
        payload = np.load(npz_path, allow_pickle=True)
        
        # Extract data arrays
        self.lr_all = payload['LR'].astype(np.float32)  # (N, 16, 1000)
        self.hr_all = payload['HR'].astype(np.float32)  # (N, 31, 1000)
        self.sr_all = payload['SR'].astype(np.float32)  # (N, 62, 1000)
        self.subject_ids_all = np.asarray(payload['subject_ids']).astype(str)
        
        # Filter by test subjects if provided (prevents data leakage)
        if test_subjects is not None:
            wanted = np.array([str(s) for s in test_subjects])
            mask = np.isin(self.subject_ids_all, wanted)
            
            if np.any(mask):
                self.lr_all = self.lr_all[mask]
                self.hr_all = self.hr_all[mask]
                self.sr_all = self.sr_all[mask]
                self.subject_ids_all = self.subject_ids_all[mask]
                print(f"✅ Filtered {np.sum(mask)} samples for test subjects {wanted}")
            else:
                available = sorted(np.unique(self.subject_ids_all).tolist())
                raise ValueError(f"No samples matched test subjects {test_subjects}. Available: {available}")
        
        # Generate trial indices (trial per N samples approximately)
        self.trial_ids = np.arange(len(self.sr_all), dtype=int)
        
        print(f"Loaded {len(self.sr_all)} samples from NPZ (test subjects only)")

    def __len__(self):
        return len(self.sr_all)

    def __getitem__(self, idx):
        return {
            'lr': torch.from_numpy(self.lr_all[idx]).float(),
            'hr': torch.from_numpy(self.hr_all[idx]).float(),
            'sr': torch.from_numpy(self.sr_all[idx]).float(),
            'subject_id': self.subject_ids_all[idx],
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


def save_multiresolution_topomap_visualization(lr_eeg, hr_eeg, sr_eeg, pred_sr, out_path, n_samples=10):
    """Generate multi-resolution topomaps for multiple samples."""
    if not HAS_MNE:
        print("Skipping multi-resolution topomap (MNE not available)")
        return
    
    try:
        # Create separate figure for each sample
        for sample_idx in range(min(n_samples, len(pred_sr))):
            # Extract sample
            lr_sample = lr_eeg[sample_idx].mean(axis=1)  # (16,)
            hr_sample = hr_eeg[sample_idx].mean(axis=1)  # (31,)
            sr_sample = sr_eeg[sample_idx].mean(axis=1)  # (62,)
            pred_sample = pred_sr[sample_idx].mean(axis=1)  # (62,)
            
            fig = plt.figure(figsize=(16, 8))
            
            # Helper to create MNE info
            def get_info_and_montage(n_ch):
                ch_names = [f'E{i+1}' for i in range(n_ch)]
                fs = 128
                info = mne.create_info(ch_names, fs, ch_types='eeg')
                
                angles = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
                radius = 0.5
                pos_x = radius * np.cos(angles)
                pos_y = radius * np.sin(angles)
                pos_z = np.zeros(n_ch)
                
                pos_dict = {ch: np.array([x, y, z]) 
                           for ch, x, y, z in zip(ch_names, pos_x, pos_y, pos_z)}
                montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')
                info.set_montage(montage)
                return info
            
            vmin = sr_sample.min()
            vmax = sr_sample.max()
            
            # LR (16 channels)
            info_lr = get_info_and_montage(16)
            ax1 = plt.subplot(2, 2, 1)
            im1, _ = mne.viz.plot_topomap(lr_sample, info_lr, axes=ax1, show=False,
                                          cmap='RdBu_r', contours=6, outlines='head',
                                          sensors=True, vmin=vmin, vmax=vmax)
                vmin_lr = lr_sample.min()
                vmax_lr = lr_sample.max()
                im1.set_clim(vmin_lr, vmax_lr) if hasattr(im1, 'set_clim') else None
                ax1.set_title('LR (16-ch Input)', fontsize=12, fontweight='bold')
                cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                cbar1.set_label('Amplitude (μV)', fontsize=9)
            
            # HR (31 channels)
            info_hr = get_info_and_montage(31)
            ax2 = plt.subplot(2, 2, 2)
            im2, _ = mne.viz.plot_topomap(hr_sample, info_hr, axes=ax2, show=False,
                                          cmap='RdBu_r', contours=6, outlines='head',
                                          sensors=True, vmin=vmin, vmax=vmax)
                vmin_hr = hr_sample.min()
                vmax_hr = hr_sample.max()
                im2.set_clim(vmin_hr, vmax_hr) if hasattr(im2, 'set_clim') else None
                ax2.set_title('HR (31-ch Intermediate)', fontsize=12, fontweight='bold')
                cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                cbar2.set_label('Amplitude (μV)', fontsize=9)
            
            # SR Target (62 channels)
            info_sr = get_info_and_montage(62)
            ax3 = plt.subplot(2, 2, 3)
            im3, _ = mne.viz.plot_topomap(sr_sample, info_sr, axes=ax3, show=False,
                                          cmap='RdBu_r', contours=6, outlines='head',
                                          sensors=True, vmin=vmin, vmax=vmax)
                ax3.set_title('SR Target (Ground Truth)', fontsize=12, fontweight='bold')
                cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                cbar3.set_label('Amplitude (μV)', fontsize=9)
            
            # SR Predicted (62 channels)
            ax4 = plt.subplot(2, 2, 4)
            im4, _ = mne.viz.plot_topomap(pred_sample, info_sr, axes=ax4, show=False,
                                          cmap='RdBu_r', contours=6, outlines='head',
                                          sensors=True, vmin=vmin, vmax=vmax)
                ax4.set_title('SR Predicted (Model Output)', fontsize=12, fontweight='bold')
                cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
                cbar4.set_label('Amplitude (μV)', fontsize=9)
            
            fig.suptitle(f'Multi-Resolution Topomaps: Sample {sample_idx}', 
                        fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            sample_out_path = str(out_path).replace('.png', f'_sample_{sample_idx:02d}.png')
            fig.savefig(sample_out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        print(f"✅ Saved {min(n_samples, len(pred_sr))} multi-resolution topomaps")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to generate topomaps: {e}")


def save_ten_samples_comparison(lr_eeg, hr_eeg, pred_sr, target_sr, out_path, n_samples=10):
    """Save 10 sample waveform comparisons."""
    n_samples = min(n_samples, len(pred_sr))
    channels_to_plot = [0, 15, 31, 45, 61]  # Representative channels
    
    fig, axes = plt.subplots(n_samples, len(channels_to_plot), 
                             figsize=(18, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for s_idx in range(n_samples):
        for ch_idx, ch in enumerate(channels_to_plot):
            ax = axes[s_idx, ch_idx]
            
            pred_signal = pred_sr[s_idx, ch, :]
            target_signal = target_sr[s_idx, ch, :]
            
            ax.plot(target_signal, label='Target', linewidth=1.0, color='black', alpha=0.8)
            ax.plot(pred_signal, label='Predicted', linewidth=1.0, color='steelblue', alpha=0.7)
            
            ax.set_title(f'Sample {s_idx}, Ch {ch}', fontsize=10)
            ax.set_ylabel('Amplitude (μV)', fontsize=9)
            ax.grid(alpha=0.3)
            
            if s_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    axes[-1, 0].set_xlabel('Time Samples', fontsize=10)
    fig.suptitle(f'10-Sample EEG Waveform Comparison (LR→HR→SR)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved 10-sample comparison: {out_path}")


def save_subject_wise_outputs(pred_sr, target_sr, subject_ids, trial_ids, output_root):
    """Save outputs organized by subject like eeg_raw_data structure."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    groups = {}
    for idx, (subj, trial) in enumerate(zip(subject_ids, trial_ids)):
        key = (str(subj), int(trial))
        groups.setdefault(key, []).append(idx)
    
    print(f"\nSaving {len(groups)} subject/trial groups to {output_root}")
    
    for (subj, trial), indices in groups.items():
        subject_dir = output_root / f'subject_{subj}'
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        idx_arr = np.asarray(indices, dtype=int)
        
        # Save predicted and target SR
        np.save(subject_dir / f'trial_{int(trial):02d}_pred_sr.npy', pred_sr[idx_arr])
        np.save(subject_dir / f'trial_{int(trial):02d}_target_sr.npy', target_sr[idx_arr])
        
        # Save metadata
        with open(subject_dir / f'trial_{int(trial):02d}_meta.json', 'w') as f:
            json.dump({
                'subject_id': str(subj),
                'trial_id': int(trial),
                'n_windows': len(idx_arr),
                'indices': idx_arr.tolist(),
            }, f, indent=2)
    
    print(f"✅ Saved subject-wise outputs to: {output_root}")


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test split for data leakage prevention
    splits = create_split(n_folds=5, test_fold=args.test_fold)
    test_subjects = splits['test']
    print(f"\n{'='*80}")
    print(f"Test Fold: {args.test_fold}")
    print(f"Test Subjects: {test_subjects}")
    print(f"{'='*80}\n")
    
    # Load dataset with test subject filtering
    dataset = RawDataNPZDataset(
        args.data_path,
        test_subjects=test_subjects,
        lr_channels=16,
        hr_channels=31,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Load models
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
        print(f"Warning: {len(missing)} missing keys in checkpoint")
    
    model.eval()
    
    # Evaluation loop
    all_pred_sr = []
    all_target_sr = []
    all_lr_eeg = []
    all_hr_eeg = []
    all_subject_ids = []
    all_trial_ids = []
    
    pcc_scores = []
    nmse_scores = []
    snr_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Evaluating')):
            lr_eeg = batch['lr'].to(device)
            hr_eeg = batch['hr'].to(device)
            sr_eeg = batch['sr'].to(device)
            subject_ids = list(batch['subject_id'])
            trial_ids = batch['trial_id'].cpu().numpy().astype(int).tolist()
            
            # Model inference
            diff_loss, pred_sr = model(lr_eeg, hr_eeg, sr_eeg)
            
            # Compute metrics
            metrics = compute_sr_metrics(pred_sr.float(), sr_eeg.float())
            
            # Collect outputs
            all_pred_sr.append(pred_sr.detach().cpu().numpy())
            all_target_sr.append(sr_eeg.detach().cpu().numpy())
            all_lr_eeg.append(lr_eeg.detach().cpu().numpy())
            all_hr_eeg.append(hr_eeg.detach().cpu().numpy())
            all_subject_ids.extend(subject_ids)
            all_trial_ids.extend(trial_ids)
            
            pcc_scores.append(metrics['pcc'])
            nmse_scores.append(metrics['nmse'])
            snr_scores.append(metrics['snr'])
            
            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                break
    
    # Aggregate results
    mean_pcc = np.mean(pcc_scores) if pcc_scores else float('nan')
    mean_nmse = np.mean(nmse_scores) if nmse_scores else float('nan')
    mean_snr = np.mean(snr_scores) if snr_scores else float('nan')
    
    print('\n' + '='*80)
    print('EVALUATION RESULTS (Raw Data - Test Subjects Only)')
    print('='*80)
    print(f"Total Samples: {len(all_subject_ids)}")
    print(f"PCC:  {mean_pcc:.4f}")
    print(f"NMSE: {mean_nmse:.4f}")
    print(f"SNR:  {mean_snr:.2f} dB")
    
    # Save all predictions and targets
    pred_all = np.concatenate(all_pred_sr, axis=0)
    target_all = np.concatenate(all_target_sr, axis=0)
    lr_all = np.concatenate(all_lr_eeg, axis=0)
    hr_all = np.concatenate(all_hr_eeg, axis=0)
    
    # Generate visualizations
    print('\n' + '='*80)
    print('Generating Visualizations...')
    print('='*80)
    
    viz_dir = Path(args.output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 10-sample comparison
    ten_sample_path = viz_dir / 'ten_samples_comparison.png'
    save_ten_samples_comparison(lr_all[:10], hr_all[:10], pred_all[:10], target_all[:10], 
                               str(ten_sample_path), n_samples=10)
    
    # Multi-resolution topomaps
    topomap_path = viz_dir / 'topomap_multiresolution.png'
    save_multiresolution_topomap_visualization(lr_all[:10], hr_all[:10], target_all[:10], 
                                             pred_all[:10], str(topomap_path), n_samples=10)
    
    # Save subject-wise outputs
    if args.save_subject_wise:
        save_subject_wise_outputs(pred_all, target_all, all_subject_ids, all_trial_ids, 
                                args.save_subject_wise)
    
    # Save summary NPZ
    summary_path = viz_dir / 'results_summary.npz'
    np.savez(
        summary_path,
        pred_sr=pred_all,
        target_sr=target_all,
        subject_ids=np.array(all_subject_ids),
        trial_ids=np.array(all_trial_ids),
        pcc_scores=np.array(pcc_scores),
        nmse_scores=np.array(nmse_scores),
        snr_scores=np.array(snr_scores),
    )
    print(f"✅ Saved summary results: {summary_path}")
    
    print(f"\n✅ All outputs saved to: {viz_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('STAD Evaluation on SEED-IV raw_data.npz')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to raw_data.npz')
    parser.add_argument('--mae_checkpoint', type=str, required=True,
                       help='Path to MAE checkpoint')
    parser.add_argument('--stad_checkpoint', type=str, required=True,
                       help='Path to STAD checkpoint')
    parser.add_argument('--test_fold', type=int, default=0,
                       help='Fold for test split (0-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='cuda or cpu')
    parser.add_argument('--diffusion_schedule', type=str, default='cosine',
                       help='Diffusion schedule')
    parser.add_argument('--max_batches', type=int, default=0,
                       help='Max batches to process (0=all)')
    parser.add_argument('--output_dir', type=str, default='stad_raw_output',
                       help='Directory for visualizations')
    parser.add_argument('--save_subject_wise', type=str, default='',
                       help='Directory to save subject-wise outputs')

    evaluate(parser.parse_args())
