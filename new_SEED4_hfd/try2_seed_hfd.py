#!/usr/bin/env python3
"""
STAD Training for SEED-IV Dataset (62 channels, 250Hz)
Modified to use HFD Profile Loss + MSE instead of plain MSE/L1.

HFD loss file expected at:
  /home/ab_students/EEG-MTP/codes_sneha/harmonic_hfd/hfd_profile_loss.py

Results are saved to:
  /home/ab_students/EEG-MTP/new_SEED4_hfd/
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

# ---------------------------------------------------------------------------
# Add HFD loss module to path and import it
# ---------------------------------------------------------------------------
HFD_MODULE_DIR = "/home/ab_students/EEG-MTP/codes_sneha/harmonic_hfd"
if HFD_MODULE_DIR not in sys.path:
    sys.path.insert(0, HFD_MODULE_DIR)

try:
    from hfd_profile_loss import HFDProfileLoss, k_list_logspace
    print("✅ HFD Profile Loss imported successfully.")
except ImportError as e:
    raise ImportError(
        f"Could not import HFDProfileLoss from {HFD_MODULE_DIR}/hfd_profile_loss.py\n"
        f"Original error: {e}"
    )

from config_seed4 import Config_MAE_SEED4
from mae_for_eeg import MAEforEEG
from stad_model_CORRECT import STADModel


# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "/home/ab_students/EEG-MTP/new_SEED4_hfd"


# ---------------------------------------------------------------------------
# Channel index helpers
# ---------------------------------------------------------------------------

def get_seed4_channel_indices(target_channels):
    """
    Return fixed SEED-IV channel subsets.
    """
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sr_metrics(pred_sr, target_sr, eps=1e-8):
    """Compute batch-level SR metrics: PCC, NMSE, and SNR (dB)."""
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

    mse          = (pred - target).pow(2).mean(dim=1)
    signal_power = target.pow(2).mean(dim=1)
    nmse = (mse / (signal_power + eps)).mean().item()
    snr  = (10.0 * torch.log10((signal_power + eps) / (mse + eps))).mean().item()

    return {'pcc': pcc, 'nmse': nmse, 'snr': snr}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SEED4STADDataset(Dataset):
    """SEED-IV dataset for STAD training with built-in Z-Score Normalization"""

    def __init__(
        self,
        data_path,
        subjects,
        lr_channels=16,
        hr_channels=31,
        sr_channels=62,
        raw_data=False,
    ):
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
            
        # APPLIED FIX: Z-Score Normalization per channel
        # This prevents the diffusion model from being overwhelmed by high-amplitude raw EEG variances
        print(f"Applying Z-Score Normalization to data...")
        self.mean = np.mean(self.sr_samples, axis=(0, 2), keepdims=True)
        self.std = np.std(self.sr_samples, axis=(0, 2), keepdims=True) + 1e-6
        self.sr_samples = (self.sr_samples - self.mean) / self.std
        
        # Now derive hr and lr from the newly normalized sr_samples
        self.hr_samples = self._downsample_channels(self.sr_samples, self.hr_channels, self.hr_indices)
        self.lr_samples = self._downsample_channels(self.sr_samples, self.lr_channels, self.lr_indices)

    # ------------------------------------------------------------------
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

    def _load_from_npz(self, npz_path, subjects):
        payload = np.load(npz_path, allow_pickle=True)
        if 'SR' not in payload or 'subject_ids' not in payload:
            raise KeyError(f"NPZ file must contain 'SR' and 'subject_ids' keys")

        sr_all      = payload['SR']
        subject_ids = payload['subject_ids']
        subject_ids_str = np.asarray(subject_ids).astype(str)
        subject_set     = set(str(s) for s in subjects)
        mask            = np.isin(subject_ids_str, list(subject_set))

        selected = sr_all[mask]
        print(f"Loaded {selected.shape[0]} windows from {len(subjects)} subjects using npz")
        self.sr_samples = selected.astype(np.float32)

    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MAE loading helpers
# ---------------------------------------------------------------------------

def load_mae_from_kfold(checkpoint_path, fold_num=None, device='cuda', freeze_encoder=True):
    print(f"\n{'='*80}")
    print("Loading Pretrained MAE from K-Fold Training")
    print(f"{'='*80}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    mae_config = {
        'time_len':           1000,
        'patch_size':         8,
        'embed_dim':          768,
        'in_chans':           31,
        'depth':              12,
        'num_heads':          12,
        'decoder_embed_dim':  384,
        'decoder_depth':      4,
        'decoder_num_heads':  8,
        'mlp_ratio':          4.0,
    }

    mae_model = MAEforEEG(**mae_config)

    if 'model_state_dict' in checkpoint:
        mae_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        mae_model.load_state_dict(checkpoint['model'])
    else:
        mae_model.load_state_dict(checkpoint)

    if freeze_encoder:
        print(f"\n🔒 Freezing MAE encoder...")
        for name, param in mae_model.named_parameters():
            if 'decoder' not in name:
                param.requires_grad = False

    return mae_model.to(device), mae_config


def resolve_mae_checkpoint(mae_checkpoint, mae_kfold_dir, mae_fold):
    if mae_checkpoint and Path(mae_checkpoint).exists():
        return str(Path(mae_checkpoint)), None
    
    kfold_dir = Path(mae_kfold_dir)
    if mae_fold is not None:
        cands = [kfold_dir/f'fold_{mae_fold}'/'best_model.pth']
        for c in cands:
            if c.exists(): return str(c), mae_fold

    for ckpt in sorted(kfold_dir.glob('fold_*/best_model.pth')):
        return str(ckpt), None # Fallback to first found

    raise FileNotFoundError("Could not locate MAE checkpoint.")


# ---------------------------------------------------------------------------
# HFD loss builder
# ---------------------------------------------------------------------------

def build_hfd_loss(fs_hz: float = 250.0, device: torch.device = None) -> HFDProfileLoss:
    k_list = k_list_logspace(fs_hz=fs_hz, min_ms=4.0, max_ms=200.0, num_scales=16)
    loss_fn = HFDProfileLoss(k_list=k_list, distance='mse', reduction='mean')
    if device is not None: loss_fn = loss_fn.to(device)
    return loss_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir):
    hfd_loss_fn = build_hfd_loss(fs_hz=250.0, device=device)

    trainable_params = [p for p in stad_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_val_loss = float('inf')
    history, start_epoch = [], 0
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    mae_unfrozen = False

    if args.resume_stad_checkpoint and Path(args.resume_stad_checkpoint).exists():
        resume_payload = torch.load(args.resume_stad_checkpoint, map_location='cpu', weights_only=False)
        stad_model.load_state_dict(resume_payload['model_state_dict'], strict=False)
        start_epoch = int(resume_payload.get('epoch', 0))
        if args.resume_optimizer:
            try:
                optimizer.load_state_dict(resume_payload['optimizer_state_dict'])
                scheduler.load_state_dict(resume_payload['scheduler_state_dict'])
            except: pass
        print(f"\n🔁 Resumed STAD from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        
        # APPLIED FIX: Loss Warmup Schedule
        # Slowly introduce the HFD and MSE loss over the first 50 epochs
        warmup_factor = min(1.0, (epoch + 1) / args.warmup_epochs)
        current_sr_weight = args.sr_loss_weight * warmup_factor
        current_hfd_weight = args.hfd_loss_weight * warmup_factor

        # Unfreeze MAE Logic
        if args.freeze_mae and not mae_unfrozen and epoch >= args.unfreeze_mae_epoch:
            new_params = []
            for name, p in stad_model.mae_encoder.named_parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)
            if new_params:
                optimizer.add_param_group({'params': new_params, 'lr': args.mae_finetune_lr})
            mae_unfrozen = True

        stad_model.train()
        train_total, train_diff, train_mse, train_hfd = [], [], [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
            lr_eeg, hr_eeg, sr_eeg = batch['lr'].to(device), batch['hr'].to(device), batch['sr'].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                diff_loss, pred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                mse_loss = F.mse_loss(pred_sr.float(), sr_eeg.float())
                hfd_loss = hfd_loss_fn(pred_sr.float(), sr_eeg.float().detach())

                total_loss = diff_loss + (current_sr_weight * mse_loss) + (current_hfd_weight * hfd_loss)

            if not torch.isfinite(total_loss): continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(stad_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_total.append(total_loss.item()); train_diff.append(diff_loss.item())
            train_mse.append(mse_loss.item()); train_hfd.append(hfd_loss.item())

        # VALIDATE
        stad_model.eval()
        val_total, val_pcc, val_nmse, val_snr = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]"):
                lr_eeg, hr_eeg, sr_eeg = batch['lr'].to(device), batch['hr'].to(device), batch['sr'].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    vd_loss, vpred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                    
                    # APPLIED FIX: Check for proper generation method
                    # If STADModel has a proper sampling function, we use that for metrics instead of the 1-step guess
                    if hasattr(stad_model, 'sample'):
                        eval_sr = stad_model.sample(lr_eeg, hr_eeg)
                    else:
                        eval_sr = vpred_sr

                    vm_loss = F.mse_loss(vpred_sr.float(), sr_eeg.float())
                    vh_loss = hfd_loss_fn(vpred_sr.float(), sr_eeg.float())
                    vt_loss = vd_loss + (args.sr_loss_weight * vm_loss) + (args.hfd_loss_weight * vh_loss)

                if not torch.isfinite(vt_loss): continue

                metrics = compute_sr_metrics(eval_sr.float(), sr_eeg.float())
                val_total.append(vt_loss.item())
                val_pcc.append(metrics['pcc']); val_nmse.append(metrics['nmse']); val_snr.append(metrics['snr'])

        scheduler.step()
        
        # Logging
        mean_vt = np.mean(val_total) if val_total else float('inf')
        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {np.mean(train_total):.4f} | Val Loss: {mean_vt:.4f}")
        print(f"Metrics -> PCC: {np.mean(val_pcc):.4f} | NMSE: {np.mean(val_nmse):.4f} | SNR: {np.mean(val_snr):.2f} dB")
        print(f"Current Weights -> HFD: {current_hfd_weight:.4f} | MSE: {current_sr_weight:.4f}")

        if mean_vt < best_val_loss:
            best_val_loss = mean_vt
            torch.save({'epoch': epoch+1, 'model_state_dict': stad_model.state_dict()}, output_dir / 'best_stad_model.pth')
            
        torch.save({
            'epoch': epoch+1, 
            'model_state_dict': stad_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, output_dir / 'latest_stad_model.pth')

def create_split(data_path, n_folds=5, test_fold=0):
    all_subjects = [str(i) for i in range(1, 16)]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2024)
    splits = list(kf.split(all_subjects))
    train_val_idx, test_idx = splits[test_fold]
    val_size = len(train_val_idx) // 5
    return {
        'train': [all_subjects[i] for i in train_val_idx[val_size:]],
        'val':   [all_subjects[i] for i in train_val_idx[:val_size]],
        'test':  [all_subjects[i] for i in test_idx],
    }

def main():
    parser = argparse.ArgumentParser('SEED-IV STAD Training')
    parser.add_argument('--mae_checkpoint', type=str, default='')
    parser.add_argument('--mae_kfold_dir', type=str, default='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_fixed')
    parser.add_argument('--mae_fold', type=int, default=0)
    parser.add_argument('--freeze_mae', action='store_true', default=True)
    parser.add_argument('--data_path', type=str, default='/DATA/EEG-MTP/seed4/eeg_processed_data')
    parser.add_argument('--test_fold', type=int, default=0)
    parser.add_argument('--raw_data', action='store_true')
    
    # Updated Training Defaults
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr',       type=float, default=1e-6)
    
    # Loss Weights and Warmup
    parser.add_argument('--sr_loss_weight', type=float, default=0.1)
    parser.add_argument('--hfd_loss_weight', type=float, default=0.3)
    parser.add_argument('--warmup_epochs', type=int, default=50, help='Epochs to ramp up HFD/MSE loss')
    
    parser.add_argument('--diffusion_schedule', type=str, default='cosine')
    parser.add_argument('--unfreeze_mae_epoch', type=int, default=50)
    parser.add_argument('--mae_finetune_lr', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_stad_checkpoint', type=str, default='')
    parser.add_argument('--resume_optimizer', action='store_true')

    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    splits = create_split(args.data_path, n_folds=5, test_fold=args.test_fold)
    train_dataset = SEED4STADDataset(args.data_path, splits['train'], raw_data=args.raw_data)
    val_dataset   = SEED4STADDataset(args.data_path, splits['val'], raw_data=args.raw_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    checkpoint_path, _ = resolve_mae_checkpoint(args.mae_checkpoint, args.mae_kfold_dir, args.mae_fold)
    mae_model, mae_config = load_mae_from_kfold(checkpoint_path, args.mae_fold, device, args.freeze_mae)

    # Note: Extracting num_patches from a dummy tensor
    dummy_hr = torch.randn(2, mae_config['in_chans'], mae_config['time_len']).to(device)
    with torch.no_grad():
        latents, _, _ = mae_model.forward_encoder(dummy_hr, mask_ratio=0.0)
    num_patches = latents.shape[1] - 1

    stad_model = STADModel(
        mae_encoder=mae_model, lr_channels=16, hr_channels=mae_config['in_chans'],
        sr_channels=62, latent_dim=mae_config['embed_dim'], num_patches=num_patches,
        diffusion_schedule=args.diffusion_schedule, lr_channel_indices=train_dataset.lr_indices,
        device=device
    ).to(device)

    train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir)

if __name__ == '__main__':
    main()