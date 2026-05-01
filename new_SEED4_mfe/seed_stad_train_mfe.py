#!/usr/bin/env python3
"""
STAD Training for SEED-IV Dataset (62 channels, 250Hz)
Using MFE (Multiscale Fuzzy Entropy) Profile Loss for complexity preservation.

Features:
- Differentiable Multiscale Fuzzy Entropy loss
- Z-score normalization for robust entropy calculation (Handled internally)
- Combined loss: diffusion + L1 (Baseline) + MFE
- Default weights: SR=0.1, MFE=0.1 (tunable)
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# Import MFE loss
try:
    from mfe_profile_loss import MFEProfileLoss
    print("✅ MFE Profile Loss imported successfully.")
except ImportError as e:
    raise ImportError(f"Could not import MFEProfileLoss: {e}")

from config_seed4 import Config_MAE_SEED4
from mae_for_eeg import MAEforEEG
from stad_model_CORRECT import STADModel

DEFAULT_OUTPUT_DIR = "/home/ab_students/EEG-MTP/new_SEED4_mfe"

# =========================================================================
# Channel index helpers
# =========================================================================

def get_seed4_channel_indices(target_channels):
    if target_channels == 62:
        return np.arange(62, dtype=int)
    if target_channels == 31:
        return np.array([0, 2, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                         31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 58, 60], dtype=int)
    if target_channels == 16:
        return np.array([0, 2, 5, 7, 9, 13, 17, 21, 23, 27, 31, 35, 39, 45, 53, 60], dtype=int)
    return np.linspace(0, 61, target_channels, dtype=int)

# =========================================================================
# Metrics
# =========================================================================

def compute_sr_metrics(pred_sr, target_sr, eps=1e-8):
    pred   = pred_sr.reshape(pred_sr.shape[0], -1)
    target = target_sr.reshape(target_sr.shape[0], -1)
    pred_centered   = pred   - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator   = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered.pow(2).sum(dim=1) + eps) *
        (target_centered.pow(2).sum(dim=1) + eps)
    )
    pcc = (numerator / denominator).mean().item()
    mse = (pred - target).pow(2).mean(dim=1)
    signal_power = target.pow(2).mean(dim=1)
    nmse = (mse / (signal_power + eps)).mean().item()
    snr  = (10.0 * torch.log10((signal_power + eps) / (mse + eps))).mean().item()
    return {'pcc': pcc, 'nmse': nmse, 'snr': snr}

# =========================================================================
# Dataset
# =========================================================================

class SEED4STADDataset(Dataset):
    def __init__(self, data_path, subjects, lr_channels=16, hr_channels=31, sr_channels=62, raw_data=False):
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels
        self.lr_indices  = get_seed4_channel_indices(lr_channels)
        self.hr_indices  = get_seed4_channel_indices(hr_channels)
        self.raw_data    = raw_data
        data_path = Path(data_path)

        if raw_data:
            self._load_raw_data(data_path, subjects)
        elif data_path.is_file() and data_path.suffix == '.npz':
            self._load_from_npz(data_path, subjects)
        else:
            self._load_processed_data(data_path, subjects)

    def _load_processed_data(self, data_path, subjects):
        all_windows = []
        for subject_id in subjects:
            for session in ['1', '2', '3']:
                session_path = data_path / session
                if not session_path.exists():
                    continue
                for folder in session_path.glob(f'{subject_id}_*'):
                    x_file = folder / 'X_prc1.npy'
                    if x_file.exists():
                        all_windows.append(np.load(x_file))
        if not all_windows:
            raise ValueError(f"No processed data found for subjects {subjects}")
        all_windows = np.concatenate(all_windows, axis=0)
        print(f"Loaded {len(all_windows)} windows from {len(subjects)} subjects")
        self.sr_samples = all_windows.astype(np.float32)
        self.hr_samples = self._downsample_channels(all_windows, self.hr_channels, self.hr_indices)
        self.lr_samples = self._downsample_channels(all_windows, self.lr_channels, self.lr_indices)

    def _load_raw_data(self, data_path, subjects):
        all_windows = []
        for subject_id in subjects:
            subject_path = data_path / f"subject_{subject_id}"
            if not subject_path.exists():
                continue
            for session_file in subject_path.glob("*.npy"):
                all_windows.append(np.load(session_file))
        if not all_windows:
            raise ValueError(f"No raw data found for subjects {subjects}")
        all_windows = np.concatenate(all_windows, axis=0)
        print(f"Loaded {len(all_windows)} windows from {len(subjects)} subjects (raw data)")
        self.sr_samples = all_windows.astype(np.float32)
        self.hr_samples = self._downsample_channels(all_windows, self.hr_channels, self.hr_indices)
        self.lr_samples = self._downsample_channels(all_windows, self.lr_channels, self.lr_indices)

    def _load_from_npz(self, npz_path, subjects):
        payload = np.load(npz_path, allow_pickle=True)
        if 'SR' not in payload or 'subject_ids' not in payload:
            raise KeyError("NPZ file must contain 'SR' and 'subject_ids' keys")
        sr_all = payload['SR']
        subject_ids = payload['subject_ids']
        if sr_all.ndim != 3:
            raise ValueError(f"Expected SR shape (N, C, T), got {sr_all.shape}")
        subject_ids_str = np.asarray(subject_ids).astype(str)
        subject_set = set(str(s) for s in subjects)
        mask = np.isin(subject_ids_str, list(subject_set))
        if not np.any(mask):
            raise ValueError(f"No data found for subjects {subjects}")
        selected = sr_all[mask]
        print(f"Loaded {selected.shape[0]} windows from {len(subjects)} subjects using npz")
        self.sr_samples = selected.astype(np.float32)
        self.hr_samples = self._downsample_channels(self.sr_samples, self.hr_channels, self.hr_indices)
        self.lr_samples = self._downsample_channels(self.sr_samples, self.lr_channels, self.lr_indices)

    def _downsample_channels(self, data, target_channels, indices=None):
        if target_channels == data.shape[1]:
            return data.astype(np.float32)
        if indices is None:
            indices = np.linspace(0, data.shape[1] - 1, target_channels, dtype=int)
        return data[:, indices, :].astype(np.float32)

    def __len__(self):
        return len(self.sr_samples)

    def __getitem__(self, idx):
        return {
            'lr': torch.from_numpy(self.lr_samples[idx]).float(),
            'hr': torch.from_numpy(self.hr_samples[idx]).float(),
            'sr': torch.from_numpy(self.sr_samples[idx]).float(),
        }

# =========================================================================
# MAE loading
# =========================================================================

def load_mae_from_kfold(checkpoint_path, fold_num=None, device='cuda', freeze_encoder=True):
    print(f"\n{'='*80}\nLoading Pretrained MAE\n{'='*80}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    mae_config = {
        'time_len': 1000, 'patch_size': 8, 'embed_dim': 768,
        'in_chans': 31, 'depth': 12, 'num_heads': 12,
        'decoder_embed_dim': 384, 'decoder_depth': 4, 'decoder_num_heads': 8, 'mlp_ratio': 4.0,
    }
    print(f"📊 MAE Config: {mae_config}")
    mae_model = MAEforEEG(**mae_config)
    if 'model_state_dict' in checkpoint:
        mae_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        mae_model.load_state_dict(checkpoint['model'])
    else:
        mae_model.load_state_dict(checkpoint)
    print(f"✅ MAE loaded")
    if freeze_encoder:
        for name, param in mae_model.named_parameters():
            if 'decoder' not in name:
                param.requires_grad = False
        print(f"🔒 Encoder frozen")
    mae_model = mae_model.to(device)
    print(f"{'='*80}\n")
    return mae_model, mae_config

def resolve_mae_checkpoint(mae_checkpoint, mae_kfold_dir, mae_fold):
    if mae_checkpoint and Path(mae_checkpoint).exists():
        return str(mae_checkpoint), None
    kfold_dir = Path(mae_kfold_dir)
    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold directory not found: {kfold_dir}")
    best_pth = kfold_dir / 'best_model.pth'
    if best_pth.exists():
        print(f"✅ Found MAE checkpoint: {best_pth}")
        return str(best_pth), None
    best_pt = kfold_dir / 'best_model.pt'
    if best_pt.exists():
        print(f"✅ Found MAE checkpoint: {best_pt}")
        return str(best_pt), None
    best_path = list(kfold_dir.glob('fold_*/best_model.pth'))
    if best_path:
        print(f"✅ Found MAE checkpoint: {best_path[0]}")
        return str(best_path[0]), None
    raise FileNotFoundError(f"No MAE checkpoint found in {kfold_dir}")

# =========================================================================
# MFE loss builder
# =========================================================================

def build_mfe_loss(m=2, n=2.0, tau_max=20, normalize_z=True, r_fixed=0.15, device=None):
    loss_fn = MFEProfileLoss(m=m, n=n, tau_max=tau_max, normalize_z=normalize_z, 
                             r_fixed=r_fixed, distance='mse', reduction='mean')
    if device is not None:
        loss_fn = loss_fn.to(device)
    print(f"\n📊 MFE Config: m={m}, n={n}, tau_max={tau_max}, r={r_fixed}")
    return loss_fn

# =========================================================================
# Training loop
# =========================================================================

def train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir):
    mfe_loss_fn = build_mfe_loss(m=args.mfe_m, n=args.mfe_n, tau_max=args.mfe_tau_max,
                                 normalize_z=True, r_fixed=args.mfe_r_fixed, device=device)
    
    trainable_params = [p for p in stad_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    best_val_loss = float('inf')
    history = []
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    start_epoch = 0
    mae_unfrozen = False

    if args.resume_stad_checkpoint and os.path.exists(args.resume_stad_checkpoint):
        print(f"\n🔁 Resuming from checkpoint: {args.resume_stad_checkpoint}")
        checkpoint = torch.load(args.resume_stad_checkpoint, map_location=device)

        stad_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if args.resume_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        if (args.freeze_mae and not mae_unfrozen and args.unfreeze_mae_epoch >= 0 
            and epoch >= args.unfreeze_mae_epoch):
            for param in stad_model.mae_encoder.parameters():
                param.requires_grad = True
            mae_unfrozen = True
            print(f"\n🔓 Unfroze MAE at epoch {epoch + 1}")

        # TRAIN
        stad_model.train()
        # CHANGED: 'mse' changed to 'sr' for consistency
        train_losses = {'total': [], 'diff': [], 'sr': [], 'mfe': []}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
            lr_eeg = batch['lr'].to(device)
            hr_eeg = batch['hr'].to(device)
            sr_eeg = batch['sr'].to(device)
            optimizer.zero_grad(set_to_none=True)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    diff_loss, pred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                    
                    # CHANGED: Baseline is L1
                    sr_loss = F.l1_loss(pred_sr.float(), sr_eeg.float())
                    
                    # CHANGED: Removed manual Z-score; MFE handles it
                    mfe_loss = 0
                    for ch in range(pred_sr.shape[1]):
                        mfe_loss += mfe_loss_fn(
                            pred_sr[:, ch:ch+1, :],
                            sr_eeg[:, ch:ch+1, :].detach()
                        )
                    mfe_loss /= pred_sr.shape[1]
                    if epoch < 20:
                        mfe_weight = 0.0
                    elif epoch < 50:
                        mfe_weight = args.mfe_loss_weight * (epoch - 20) / 30
                    else:
                        mfe_weight = args.mfe_loss_weight
                    mfe_loss = torch.clamp(mfe_loss, 0, 5.0)
                    total_loss = diff_loss + args.sr_loss_weight * sr_loss + mfe_weight * mfe_loss
            else:
                diff_loss, pred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                sr_loss = F.l1_loss(pred_sr.float(), sr_eeg.float())
                mfe_loss = 0
                for ch in range(pred_sr.shape[1]):
                    mfe_loss += mfe_loss_fn(
                        pred_sr[:, ch:ch+1, :],
                        sr_eeg[:, ch:ch+1, :].detach()
                    )
                mfe_loss /= pred_sr.shape[1]
                if epoch < 20:
                    mfe_weight = 0.0
                elif epoch < 50:
                    mfe_weight = args.mfe_loss_weight * (epoch - 20) / 30
                else:
                    mfe_weight = args.mfe_loss_weight

                mfe_loss = torch.clamp(mfe_loss, 0, 5.0)

                total_loss = diff_loss + args.sr_loss_weight * sr_loss + mfe_weight * mfe_loss

            if not torch.isfinite(total_loss):
                print("  ⚠️  Non-finite loss, skipping batch")
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in stad_model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses['total'].append(total_loss.item())
            train_losses['diff'].append(diff_loss.item())
            # CHANGED: Key is 'sr', appends sr_loss
            train_losses['sr'].append(sr_loss.item())
            train_losses['mfe'].append(mfe_loss.item())

        # VALIDATE
        stad_model.eval()
        # CHANGED: 'mse' changed to 'sr' for consistency
        val_losses = {'total': [], 'diff': [], 'sr': [], 'mfe': []}
        val_metrics = {'pcc': [], 'nmse': [], 'snr': []}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]"):
                lr_eeg = batch['lr'].to(device)
                hr_eeg = batch['hr'].to(device)
                sr_eeg = batch['sr'].to(device)

                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        vd_loss, vpred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                        
                        # CHANGED: Use L1 and vsr_loss naming
                        vsr_loss = F.l1_loss(vpred_sr.float(), sr_eeg.float())
                        vh_loss = 0
                        for ch in range(vpred_sr.shape[1]):
                            vh_loss += mfe_loss_fn(
                                vpred_sr[:, ch:ch+1, :],
                                sr_eeg[:, ch:ch+1, :].detach()
                            )
                        vh_loss /= vpred_sr.shape[1]
                        if epoch < 20:
                            mfe_weight = 0.0
                        elif epoch < 50:
                            mfe_weight = args.mfe_loss_weight * (epoch - 20) / 30
                        else:
                            mfe_weight = args.mfe_loss_weight
                        vt_loss = vd_loss + args.sr_loss_weight * vsr_loss + mfe_weight * vh_loss
                else:
                    vd_loss, vpred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                    vsr_loss = F.l1_loss(vpred_sr.float(), sr_eeg.float())
                    vh_loss = 0
                    for ch in range(vpred_sr.shape[1]):
                        vh_loss += mfe_loss_fn(
                            vpred_sr[:, ch:ch+1, :],
                            sr_eeg[:, ch:ch+1, :].detach()
                        )
                    vh_loss /= vpred_sr.shape[1]
                    if epoch < 20:
                        mfe_weight = 0.0
                    elif epoch < 50:
                        mfe_weight = args.mfe_loss_weight * (epoch - 20) / 30
                    else:
                        mfe_weight = args.mfe_loss_weight
                    vt_loss = vd_loss + args.sr_loss_weight * vsr_loss + mfe_weight * vh_loss

                if not torch.isfinite(vt_loss):
                    continue

                metrics = compute_sr_metrics(vpred_sr.float(), sr_eeg.float())
                val_losses['total'].append(vt_loss.item())
                val_losses['diff'].append(vd_loss.item())
                # CHANGED: Append vsr_loss
                val_losses['sr'].append(vsr_loss.item())
                val_losses['mfe'].append(vh_loss.item())
                val_metrics['pcc'].append(metrics['pcc'])
                val_metrics['nmse'].append(metrics['nmse'])
                val_metrics['snr'].append(metrics['snr'])

        # Aggregate metrics
        def _mean(lst):
            return float(np.mean(lst)) if lst else float('inf')

        train_loss = _mean(train_losses['total'])
        val_loss = _mean(val_losses['total'])
        mean_pcc = _mean(val_metrics['pcc']) if val_metrics['pcc'] else 0.0
        mean_nmse = _mean(val_metrics['nmse'])
        mean_snr = _mean(val_metrics['snr'])

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"PCC: {mean_pcc:.4f}, NMSE: {mean_nmse:.4f}, SNR: {mean_snr:.2f}dB"
        )

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_stad_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': stad_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'scaler_state_dict': scaler.state_dict(),
            }, save_path)
            print(f"  ✅ Saved best model → {save_path}")

        # ALWAYS save last checkpoint (for resume)
        last_ckpt_path = output_dir / 'last_checkpoint.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': stad_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_state_dict': scaler.state_dict(),
        }, last_ckpt_path)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_pcc': mean_pcc,
            'val_nmse': mean_nmse,
            'val_snr': mean_snr,
        })

    # Save history
    history_path = output_dir / 'training_history.npy'
    np.save(history_path, history, allow_pickle=True)
    print(f"\n📈 Training history saved → {history_path}")

# =========================================================================
# Data split
# =========================================================================

def create_split(data_path, n_folds=5, test_fold=0):
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

# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser('SEED-IV STAD Training — Baseline L1 + MFE Loss')
    parser.add_argument('--mae_checkpoint', type=str, default='best_model.pth')
    parser.add_argument('--mae_kfold_dir', type=str,
                       default='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_fixed')
    parser.add_argument('--mae_fold', type=int, default=None)
    parser.add_argument('--freeze_mae', action='store_true')
    parser.add_argument('--data_path', type=str, default='/DATA/EEG-MTP/seed4/eeg_processed_data')
    parser.add_argument('--test_fold', type=int, default=0)
    parser.add_argument('--raw_data', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--sr_loss_weight', type=float, default=1)
    parser.add_argument('--mfe_loss_weight', type=float, default=0.05)
    parser.add_argument('--mfe_m', type=int, default=2)
    parser.add_argument('--mfe_n', type=float, default=2.0)
    parser.add_argument('--mfe_tau_max', type=int, default=5)
    parser.add_argument('--mfe_r_fixed', type=float, default=0.15)
    parser.add_argument('--diffusion_schedule', type=str, default='cosine')
    parser.add_argument('--unfreeze_mae_epoch', type=int, default=50)
    parser.add_argument('--mae_finetune_lr', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--resume_stad_checkpoint', type=str, default='')
    parser.add_argument('--resume_optimizer', action='store_true')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n📁 Output directory: {output_dir}")

    config = Config_MAE_SEED4()
    print(config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Data splits
    print(f"\nCreating data splits...")
    splits = create_split(args.data_path, n_folds=5, test_fold=args.test_fold)
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # Datasets
    print("\nCreating datasets...")
    train_dataset = SEED4STADDataset(args.data_path, splits['train'], raw_data=args.raw_data)
    val_dataset = SEED4STADDataset(args.data_path, splits['val'], raw_data=args.raw_data)
    test_dataset = SEED4STADDataset(args.data_path, splits['test'], raw_data=args.raw_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # MAE
    try:
        checkpoint_path, _ = resolve_mae_checkpoint(args.mae_checkpoint, args.mae_kfold_dir, args.mae_fold)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return

    mae_model, mae_config = load_mae_from_kfold(checkpoint_path, device=device, freeze_encoder=args.freeze_mae)

    # STAD
    print("\n" + "="*80)
    print("Initializing STAD Model")
    print("="*80)
    test_batch = next(iter(test_loader))
    hr = test_batch['hr'].to(device)
    mae_model.eval()
    with torch.no_grad():
        latents, _, _ = mae_model.forward_encoder(hr, mask_ratio=0.0)
        latents = latents[:, 1:, :]
    print(f"✓ Latent shape: {latents.shape}")

    stad_model = STADModel(
        mae_encoder=mae_model, lr_channels=16, hr_channels=mae_config['in_chans'],
        sr_channels=62, latent_dim=mae_config['embed_dim'], num_patches=latents.shape[1],
        diffusion_schedule=args.diffusion_schedule, lr_channel_indices=train_dataset.lr_indices,
        device=device,
    )
    stad_model = stad_model.to(device)
    print("✅ STAD model initialized.")

    if args.test_only:
        print("\n✓ Test complete")
        return

    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Loss = diffusion + {args.sr_loss_weight} * L1 + {args.mfe_loss_weight} * MFE")

    print("\n" + "="*80)
    print("Training STAD (diffusion + L1 + MFE)")
    print("="*80)
    train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir)
    print("\n✅ Training finished.")
    print(f"   Results saved in: {output_dir}")

if __name__ == '__main__':
    main()