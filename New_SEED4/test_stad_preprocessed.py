#!/usr/bin/env python3
"""Evaluate STAD checkpoint on SEED-IV preprocessed data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

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


class SEED4PreprocessedDataset(Dataset):
    """Loader for preprocessed SEED-IV folder structure."""

    def __init__(self, data_path, subjects, lr_channels=16, hr_channels=31):
        self.lr_indices = get_seed4_channel_indices(lr_channels)
        self.hr_indices = get_seed4_channel_indices(hr_channels)

        data_path = Path(data_path)

        if data_path.is_file() and data_path.suffix == '.npz':
            self._load_from_npz(data_path)
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

    def _load_from_npz(self, npz_path):
        """Load test data from npz payload (supports multiple key layouts)."""
        payload = np.load(npz_path, allow_pickle=True)

        if 'SR' in payload:
            sr_all = payload['SR'].astype(np.float32)
            if 'test_indices' in payload:
                idx = payload['test_indices'].astype(int)
                sr = sr_all[idx]
            else:
                sr = sr_all
        elif 'X_test' in payload:
            sr = payload['X_test'].astype(np.float32)
        else:
            raise KeyError(
                f"Unsupported npz format: {npz_path}. Expected SR (+optional test_indices) or X_test key."
            )

        self.sr_samples = sr
        self.hr_samples = sr[:, self.hr_indices, :]
        self.lr_samples = sr[:, self.lr_indices, :]
        self.norm_stats = None
        self.prc1_meta = None

        n = len(sr)
        if 'subject_ids' in payload and len(payload['subject_ids']) >= n:
            if 'test_indices' in payload:
                self.subject_ids = np.asarray(payload['subject_ids'])[payload['test_indices']].astype(str)
            else:
                self.subject_ids = np.asarray(payload['subject_ids'][:n]).astype(str)
        else:
            self.subject_ids = np.array([f'unknown_{i:05d}' for i in range(n)])

        if 'session_ids' in payload and len(payload['session_ids']) >= n:
            if 'test_indices' in payload:
                self.session_ids = np.asarray(payload['session_ids'])[payload['test_indices']].astype(str)
            else:
                self.session_ids = np.asarray(payload['session_ids'][:n]).astype(str)
        else:
            self.session_ids = np.array(['1'] * n)

        if 'trial_ids' in payload and len(payload['trial_ids']) >= n:
            if 'test_indices' in payload:
                self.trial_ids = np.asarray(payload['trial_ids'])[payload['test_indices']].astype(int)
            else:
                self.trial_ids = np.asarray(payload['trial_ids'][:n]).astype(int)
        else:
            self.trial_ids = np.arange(n, dtype=int)

        print(f"Loaded {n} test windows from npz: {npz_path}")

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

    splits = create_split(n_folds=5, test_fold=args.test_fold)
    test_subjects = splits['test']
    print(f"Test subjects: {test_subjects}")

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

    evaluate(parser.parse_args())
