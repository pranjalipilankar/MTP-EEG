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
# Default output directory (all results go here)
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "/home/ab_students/EEG-MTP/new_SEED4_hfd"


# ---------------------------------------------------------------------------
# Channel index helpers
# ---------------------------------------------------------------------------

def get_seed4_channel_indices(target_channels):
    """
    Return fixed SEED-IV channel subsets (indices over 62-channel ordering).
    These subsets cover frontal/central/parietal/occipital regions more evenly
    than simple linear index sampling.
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
    """SEED-IV dataset for STAD training"""

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

        if 'SR' not in payload:
            raise KeyError(f"NPZ file must contain 'SR' key: {npz_path}")
        if 'subject_ids' not in payload:
            raise KeyError(f"NPZ file must contain 'subject_ids' key: {npz_path}")

        sr_all      = payload['SR']
        subject_ids = payload['subject_ids']

        if sr_all.ndim != 3:
            raise ValueError(f"Expected SR shape (N, C, T), got {sr_all.shape}")

        subject_ids_str = np.asarray(subject_ids).astype(str)
        subject_set     = set(str(s) for s in subjects)
        mask            = np.isin(subject_ids_str, list(subject_set))

        if not np.any(mask):
            available = sorted(np.unique(subject_ids_str).tolist())
            raise ValueError(
                f"No data found for subjects {subjects}. "
                f"Available: {available[:20]}"
            )

        selected = sr_all[mask]
        print(
            f"Loaded {selected.shape[0]} windows from {len(subjects)} subjects "
            f"using npz: {npz_path}"
        )

        self.sr_samples = selected.astype(np.float32)
        self.hr_samples = self._downsample_channels(self.sr_samples, self.hr_channels, self.hr_indices)
        self.lr_samples = self._downsample_channels(self.sr_samples, self.lr_channels, self.lr_indices)

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
    """Load MAE from k-fold checkpoint (31-channel model)."""
    print(f"\n{'='*80}")
    print("Loading Pretrained MAE from K-Fold Training")
    print(f"{'='*80}")
    if fold_num:
        print(f"Fold: {fold_num}")
    print(f"Checkpoint: {checkpoint_path}")

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

    print(f"\n📊 Model Configuration:")
    for key, value in mae_config.items():
        print(f"   {key}: {value}")

    mae_model = MAEforEEG(
        time_len          = mae_config['time_len'],
        patch_size        = mae_config['patch_size'],
        embed_dim         = mae_config['embed_dim'],
        in_chans          = mae_config['in_chans'],
        depth             = mae_config['depth'],
        num_heads         = mae_config['num_heads'],
        decoder_embed_dim = mae_config['decoder_embed_dim'],
        decoder_depth     = mae_config['decoder_depth'],
        decoder_num_heads = mae_config['decoder_num_heads'],
        mlp_ratio         = mae_config['mlp_ratio'],
    )

    if 'model_state_dict' in checkpoint:
        mae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✅ Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        val_loss = checkpoint.get('val_loss', None)
        val_cor  = checkpoint.get('val_corr', checkpoint.get('val_correlation', None))
        if isinstance(val_loss, (int, float)):
            print(f"   Val Loss: {val_loss:.6f}")
        if isinstance(val_cor, (int, float)):
            print(f"   Val Corr: {val_cor:.6f}")
    elif 'model' in checkpoint:
        mae_model.load_state_dict(checkpoint['model'])
        print(f"\n✅ Loaded model state dict")
    else:
        mae_model.load_state_dict(checkpoint)
        print(f"\n✅ Loaded model weights")

    if freeze_encoder:
        print(f"\n🔒 Freezing MAE encoder...")
        frozen = trainable = 0
        for name, param in mae_model.named_parameters():
            if 'decoder' not in name:
                param.requires_grad = False
                frozen += param.numel()
            else:
                trainable += param.numel()
        print(f"   Frozen params:    {frozen:,}")
        print(f"   Trainable params: {trainable:,}")

    mae_model = mae_model.to(device)
    print(f"{'='*80}\n")
    return mae_model, mae_config


def _fold_checkpoint_candidates(kfold_dir, fold_num):
    kfold_dir  = Path(kfold_dir)
    candidates = [
        kfold_dir / f'fold_{fold_num}'     / 'best_model.pth',
        kfold_dir / f'fold_{fold_num - 1}' / 'best_model.pth',
    ]
    seen, unique = set(), []
    for c in candidates:
        if str(c) not in seen:
            unique.append(c)
            seen.add(str(c))
    return unique


def resolve_mae_checkpoint(mae_checkpoint, mae_kfold_dir, mae_fold):
    """Resolve MAE checkpoint from explicit path, fold selection, or best-fold auto-select."""
    if mae_checkpoint:
        path = Path(mae_checkpoint)
        if path.exists():
            return str(path), None
        raise FileNotFoundError(f"MAE checkpoint not found: {mae_checkpoint}")

    kfold_dir = Path(mae_kfold_dir)
    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold results directory not found: {kfold_dir}")

    if mae_fold is not None:
        for candidate in _fold_checkpoint_candidates(kfold_dir, mae_fold):
            if candidate.exists():
                return str(candidate), mae_fold
        raise FileNotFoundError(
            f"No checkpoint found for fold={mae_fold} in {kfold_dir}."
        )

    def _to_python_float(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if torch.is_tensor(value) and value.numel() == 1:
            return float(value.item())
        return None

    best_path, best_score = None, -float('inf')
    for ckpt in sorted(kfold_dir.glob('fold_*/best_model.pth')):
        try:
            payload    = torch.load(ckpt, map_location='cpu', weights_only=False)
            score_raw  = payload.get('val_cor', payload.get('val_corr', None))
            score      = _to_python_float(score_raw)
            if score is not None and score > best_score:
                best_score = score
                best_path  = ckpt
        except Exception:
            continue

    if best_path is not None:
        return str(best_path), None

    raise FileNotFoundError(
        f"Could not auto-select checkpoint from {kfold_dir}. "
        f"Pass --mae_checkpoint explicitly."
    )


# ---------------------------------------------------------------------------
# HFD loss builder
# ---------------------------------------------------------------------------

def build_hfd_loss(fs_hz: float = 250.0, device: torch.device = None) -> HFDProfileLoss:
    """
    Build HFDProfileLoss tuned for 1000-sample EEG windows at 250 Hz.

    Covers 4–200 ms (delta → gamma) with 16 log-spaced scales.
    Distance = 'mse' for smooth gradients.
    """
    k_list = k_list_logspace(
        fs_hz      = fs_hz,
        min_ms     = 4.0,
        max_ms     = 200.0,
        num_scales = 16,
    )
    loss_fn = HFDProfileLoss(k_list=k_list, distance='mse', reduction='mean')
    if device is not None:
        loss_fn = loss_fn.to(device)
    print(f"\n📐 HFD k_list ({len(k_list)} scales, 4–200 ms @ {fs_hz} Hz): {k_list.tolist()}")
    return loss_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir):
    """
    Train STAD with combined loss:

        total_loss = diffusion_loss
                   + args.sr_loss_weight  * MSE(pred_sr, sr_eeg)
                   + args.hfd_loss_weight * HFD(pred_sr, sr_eeg)

    Best checkpoint (lowest val total loss) → output_dir/best_stad_model.pth
    Latest checkpoint every epoch           → output_dir/latest_stad_model.pth
    Training history                        → output_dir/training_history.npy
    """

    # ---- Build HFD loss (stays on device for entire training) ----
    hfd_loss_fn = build_hfd_loss(fs_hz=250.0, device=device)

    # ---- Optimizer / scheduler ----
    trainable_params = [p for p in stad_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = args.min_lr,
    )

    best_val_loss = float('inf')
    history       = []
    scaler        = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    start_epoch   = 0
    mae_unfrozen  = False

    # ---- Optional: resume from checkpoint ----
    if args.resume_stad_checkpoint:
        resume_path = Path(args.resume_stad_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resume_payload = torch.load(resume_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' not in resume_payload:
            raise KeyError(f"Checkpoint missing model_state_dict: {resume_path}")

        missing_keys, unexpected_keys = stad_model.load_state_dict(
            resume_payload['model_state_dict'], strict=False
        )
        if missing_keys:
            print(f"   ⚠️  Missing keys while resuming: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys while resuming: {len(unexpected_keys)}")

        start_epoch   = int(resume_payload.get('epoch', 0))
        best_val_loss = float(
            resume_payload.get('best_val_loss',
                               resume_payload.get('val_total_loss', float('inf')))
        )

        if args.resume_optimizer:
            for key, obj in [
                ('optimizer_state_dict', optimizer),
                ('scheduler_state_dict', scheduler),
                ('scaler_state_dict',    scaler),
            ]:
                if key in resume_payload:
                    try:
                        obj.load_state_dict(resume_payload[key])
                    except Exception as exc:
                        print(f"   ⚠️  Could not load {key}: {exc}")

        # Restore history
        history_path = output_dir / 'training_history.npy'
        if history_path.exists():
            try:
                loaded = np.load(history_path, allow_pickle=True).tolist()
                if isinstance(loaded, list):
                    history = loaded
            except Exception:
                pass

        print(f"\n🔁 Resumed STAD from: {resume_path}")
        print(f"   Starting at epoch:        {start_epoch + 1}/{args.epochs}")
        print(f"   Best val total loss so far: {best_val_loss:.6f}")

        # Re-apply MAE unfreeze if we're past the unfreeze epoch
        if (
            args.freeze_mae
            and args.unfreeze_mae_epoch >= 0
            and start_epoch >= args.unfreeze_mae_epoch
        ):
            for param in stad_model.mae_encoder.parameters():
                param.requires_grad = True
            mae_unfrozen = True

    if start_epoch >= args.epochs:
        print(f"\n✓ Already at epoch {start_epoch}, nothing to train.")
        return

    # ========================================================================
    # Epoch loop
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):

        # ---- Optional: unfreeze MAE encoder mid-training ----
        if (
            args.freeze_mae
            and not mae_unfrozen
            and args.unfreeze_mae_epoch >= 0
            and epoch >= args.unfreeze_mae_epoch
        ):
            newly_trainable = []
            for name, param in stad_model.mae_encoder.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    newly_trainable.append(param)
            if newly_trainable:
                optimizer.add_param_group({
                    'params':       newly_trainable,
                    'lr':           args.mae_finetune_lr,
                    'weight_decay': args.weight_decay,
                })
                print(
                    f"\n🔓 Unfroze MAE encoder at epoch {epoch + 1} | "
                    f"new params: {sum(p.numel() for p in newly_trainable):,} | "
                    f"lr={args.mae_finetune_lr}"
                )
            mae_unfrozen = True

        # ====================================================================
        # TRAIN
        # ====================================================================
        stad_model.train()
        train_total_losses = []
        train_diff_losses  = []
        train_mse_losses   = []
        train_hfd_losses   = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
            lr_eeg = batch['lr'].to(device)
            hr_eeg = batch['hr'].to(device)
            sr_eeg = batch['sr'].to(device)

            optimizer.zero_grad(set_to_none=True)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    diff_loss, pred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)

                    # MSE reconstruction loss  (pred vs 62-ch target)
                    mse_loss = F.mse_loss(pred_sr.float(), sr_eeg.float())

                    # HFD complexity-matching loss
                    # pred_sr shape: (B, 62, T) — HFDProfileLoss handles (B, C, T) natively
                    hfd_loss = hfd_loss_fn(
                        pred_sr.float(),
                        sr_eeg.float().detach(),   # no gradient needed on the target
                    )

                    total_loss = (
                        diff_loss
                        + args.sr_loss_weight  * mse_loss
                        + args.hfd_loss_weight * hfd_loss
                    )
            else:
                diff_loss, pred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                mse_loss  = F.mse_loss(pred_sr.float(), sr_eeg.float())
                hfd_loss  = hfd_loss_fn(pred_sr.float(), sr_eeg.float().detach())
                total_loss = (
                    diff_loss
                    + args.sr_loss_weight  * mse_loss
                    + args.hfd_loss_weight * hfd_loss
                )

            if not torch.isfinite(total_loss):
                print(
                    f"  ⚠️  Non-finite total loss "
                    f"(diff={diff_loss.item():.4f}, "
                    f"mse={mse_loss.item():.4f}, "
                    f"hfd={hfd_loss.item():.4f}), skipping batch."
                )
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in stad_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            train_total_losses.append(total_loss.item())
            train_diff_losses.append(diff_loss.item())
            train_mse_losses.append(mse_loss.item())
            train_hfd_losses.append(hfd_loss.item())

        # ====================================================================
        # VALIDATE
        # ====================================================================
        stad_model.eval()
        val_total_losses = []
        val_diff_losses  = []
        val_mse_losses   = []
        val_hfd_losses   = []
        val_pcc  = []
        val_nmse = []
        val_snr  = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]"):
                lr_eeg = batch['lr'].to(device)
                hr_eeg = batch['hr'].to(device)
                sr_eeg = batch['sr'].to(device)

                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        vd_loss, vpred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                        vm_loss = F.mse_loss(vpred_sr.float(), sr_eeg.float())
                        vh_loss = hfd_loss_fn(vpred_sr.float(), sr_eeg.float())
                        vt_loss = (
                            vd_loss
                            + args.sr_loss_weight  * vm_loss
                            + args.hfd_loss_weight * vh_loss
                        )
                else:
                    vd_loss, vpred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
                    vm_loss = F.mse_loss(vpred_sr.float(), sr_eeg.float())
                    vh_loss = hfd_loss_fn(vpred_sr.float(), sr_eeg.float())
                    vt_loss = (
                        vd_loss
                        + args.sr_loss_weight  * vm_loss
                        + args.hfd_loss_weight * vh_loss
                    )

                if not torch.isfinite(vt_loss):
                    continue

                metrics = compute_sr_metrics(vpred_sr.float(), sr_eeg.float())
                val_total_losses.append(vt_loss.item())
                val_diff_losses.append(vd_loss.item())
                val_mse_losses.append(vm_loss.item())
                val_hfd_losses.append(vh_loss.item())
                val_pcc.append(metrics['pcc'])
                val_nmse.append(metrics['nmse'])
                val_snr.append(metrics['snr'])

        # ---- Aggregate ----
        def _mean(lst, default=float('inf')):
            return float(np.mean(lst)) if lst else default

        train_loss = _mean(train_total_losses)
        train_diff = _mean(train_diff_losses)
        train_mse  = _mean(train_mse_losses)
        train_hfd  = _mean(train_hfd_losses)

        val_loss = _mean(val_total_losses)
        val_diff = _mean(val_diff_losses)
        val_mse  = _mean(val_mse_losses)
        val_hfd  = _mean(val_hfd_losses)

        mean_pcc  = float(np.mean(val_pcc))  if val_pcc  else 0.0
        mean_nmse = float(np.mean(val_nmse)) if val_nmse else float('inf')
        mean_snr  = float(np.mean(val_snr))  if val_snr  else -float('inf')

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train → Total: {train_loss:.6f}  Diff: {train_diff:.6f}  "
            f"MSE: {train_mse:.6f}  HFD: {train_hfd:.6f} | "
            f"Val → Total: {val_loss:.6f}  Diff: {val_diff:.6f}  "
            f"MSE: {val_mse:.6f}  HFD: {val_hfd:.6f} | "
            f"PCC: {mean_pcc:.4f}  NMSE: {mean_nmse:.4f}  SNR: {mean_snr:.2f} dB"
        )

        # ---- History entry ----
        history.append({
            'epoch':            epoch + 1,
            # train
            'train_total_loss': train_loss,
            'train_diff_loss':  train_diff,
            'train_mse_loss':   train_mse,
            'train_hfd_loss':   train_hfd,
            # val
            'val_total_loss':   val_loss,
            'val_diff_loss':    val_diff,
            'val_mse_loss':     val_mse,
            'val_hfd_loss':     val_hfd,
            # metrics
            'val_pcc':   mean_pcc,
            'val_nmse':  mean_nmse,
            'val_snr_db': mean_snr,
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        # ---- Checkpoint helpers ----
        def _build_ckpt(epoch_idx):
            payload = {
                'epoch':            epoch_idx + 1,
                'model_state_dict': stad_model.state_dict(),
                'best_val_loss':    best_val_loss,
                # losses saved for reference
                'val_total_loss': val_loss,
                'val_diff_loss':  val_diff,
                'val_mse_loss':   val_mse,
                'val_hfd_loss':   val_hfd,
                'val_pcc':        mean_pcc,
                'val_nmse':       mean_nmse,
                'val_snr_db':     mean_snr,
                'train_total_loss': train_loss,
                'train_diff_loss':  train_diff,
                'train_mse_loss':   train_mse,
                'train_hfd_loss':   train_hfd,
                # hyperparams for provenance
                'sr_loss_weight':  args.sr_loss_weight,
                'hfd_loss_weight': args.hfd_loss_weight,
            }
            if args.save_optimizer_state:
                payload['optimizer_state_dict'] = optimizer.state_dict()
                payload['scheduler_state_dict'] = scheduler.state_dict()
                payload['scaler_state_dict']    = scaler.state_dict()
            return payload

        def _safe_save(payload, path, label):
            try:
                torch.save(payload, path)
                return True
            except Exception as exc:
                print(f"  ⚠️  Could not save {label} checkpoint: {exc}")
                return False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_stad_model.pth'
            if _safe_save(_build_ckpt(epoch), save_path, 'best'):
                print(f"  ✅ Saved best STAD checkpoint → {save_path}")

        latest_path = output_dir / 'latest_stad_model.pth'
        _safe_save(_build_ckpt(epoch), latest_path, 'latest')

    # ---- Persist history ----
    history_path = output_dir / 'training_history.npy'
    np.save(history_path, history, allow_pickle=True)
    print(f"\n📈 Training history saved → {history_path}")


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def create_split(data_path, n_folds=5, test_fold=0):
    """Create train/val/test splits over 15 SEED-IV subjects."""
    all_subjects = [str(i) for i in range(1, 16)]
    from sklearn.model_selection import KFold
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=2024)
    splits = list(kf.split(all_subjects))
    train_val_idx, test_idx = splits[test_fold]
    val_size  = len(train_val_idx) // 5
    val_idx   = train_val_idx[:val_size]
    train_idx = train_val_idx[val_size:]
    return {
        'train': [all_subjects[i] for i in train_idx],
        'val':   [all_subjects[i] for i in val_idx],
        'test':  [all_subjects[i] for i in test_idx],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser('SEED-IV STAD Training — HFD + MSE Loss')

    # ---- MAE checkpoint ----
    parser.add_argument('--mae_checkpoint', type=str, default='',
                        help='Path to pretrained MAE checkpoint (.pth). '
                             'If empty, auto-selects best fold from --mae_kfold_dir.')
    parser.add_argument('--mae_kfold_dir', type=str,
                        default='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_fixed',
                        help='Directory with fold_*/best_model.pth from 31-ch MAE k-fold training')
    parser.add_argument('--mae_fold', type=int, default=None,
                        help='Specific fold to use (0- or 1-based). '
                             'Omit to auto-select by best val_cor.')
    parser.add_argument('--freeze_mae', action='store_true',
                        help='Freeze MAE encoder weights initially')

    # ---- Data ----
    parser.add_argument('--data_path', type=str,
                        default='/DATA/EEG-MTP/seed4/eeg_processed_data',
                        help='Path to SEED-IV data: processed folder or .npz with SR+subject_ids')
    parser.add_argument('--test_fold', type=int, default=0,
                        help='K-fold test split index (0–4)')
    parser.add_argument('--raw_data', action='store_true',
                        help='Use raw data loader instead of processed folder')

    # ---- Training ----
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr',       type=float, default=1e-6)

    # ---- Loss weights ----
    parser.add_argument('--sr_loss_weight', type=float, default=0.1,
                        help='Weight λ_mse for MSE reconstruction loss '
                             '(total = diff + λ_mse*MSE + λ_hfd*HFD)')
    parser.add_argument('--hfd_loss_weight', type=float, default=0.01,
                        help='Weight λ_hfd for HFD profile complexity-matching loss')

    # ---- Diffusion ----
    parser.add_argument('--diffusion_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'])

    # ---- MAE fine-tuning ----
    parser.add_argument('--unfreeze_mae_epoch', type=int, default=50,
                        help='Epoch to unfreeze MAE encoder (-1 = keep frozen)')
    parser.add_argument('--mae_finetune_lr', type=float, default=2e-5,
                        help='LR for MAE encoder after unfreezing')

    # ---- Output / device ----
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory for all outputs '
                             '(default: /home/ab_students/EEG-MTP/new_SEED4_hfd)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_only', action='store_true',
                        help='Test MAE latent extraction only, skip STAD training')

    # ---- Resume ----
    parser.add_argument('--resume_stad_checkpoint', type=str, default='',
                        help='STAD checkpoint path to resume from '
                             '(e.g. .../new_SEED4_hfd/latest_stad_model.pth)')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Also restore optimizer/scheduler/scaler from checkpoint')
    parser.add_argument('--save_optimizer_state', action='store_true',
                        help='Include optimizer/scheduler/scaler in saved checkpoints '
                             '(much larger files, enables full resume)')

    args = parser.parse_args()

    # ---- Create output directory ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n📁 Output directory: {output_dir}")

    # ---- Config / device ----
    config = Config_MAE_SEED4()
    print(config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ---- Data splits ----
    print(f"\nCreating data splits (test_fold={args.test_fold})...")
    splits = create_split(args.data_path, n_folds=5, test_fold=args.test_fold)
    print(f"  Train subjects ({len(splits['train'])}): {splits['train']}")
    print(f"  Val subjects   ({len(splits['val'])}):   {splits['val']}")
    print(f"  Test subjects  ({len(splits['test'])}):  {splits['test']}")

    # ---- Datasets & loaders ----
    print("\nCreating datasets...")
    train_dataset = SEED4STADDataset(args.data_path, splits['train'], raw_data=args.raw_data)
    val_dataset   = SEED4STADDataset(args.data_path, splits['val'],   raw_data=args.raw_data)
    test_dataset  = SEED4STADDataset(args.data_path, splits['test'],  raw_data=args.raw_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"\n  Train: {len(train_dataset)} windows ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} windows ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_dataset)} windows ({len(test_loader)} batches)")

    # ---- Resolve MAE checkpoint ----
    try:
        checkpoint_path, resolved_fold = resolve_mae_checkpoint(
            mae_checkpoint = args.mae_checkpoint,
            mae_kfold_dir  = args.mae_kfold_dir,
            mae_fold       = args.mae_fold,
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return

    print(f"\n💡 Using MAE checkpoint: {checkpoint_path}")

    # ---- Load MAE ----
    mae_model, mae_config = load_mae_from_kfold(
        checkpoint_path = checkpoint_path,
        fold_num        = args.mae_fold,
        device          = device,
        freeze_encoder  = args.freeze_mae,
    )

    # ---- Quick latent extraction sanity check ----
    print("\n" + "="*80)
    print("Testing MAE Latent Extraction")
    print("="*80)
    test_batch = next(iter(test_loader))
    hr = test_batch['hr'].to(device)

    mae_model.eval()
    with torch.no_grad():
        latents, _, _ = mae_model.forward_encoder(hr, mask_ratio=0.0)
        latents = latents[:, 1:, :]  # remove CLS token

    print(f"✓ Input shape:  {hr.shape}")
    print(f"✓ Latent shape: {latents.shape}")
    expected_patches = mae_config['time_len'] // mae_config['patch_size']
    assert latents.shape[1] == expected_patches, \
        f"Patch count mismatch: got {latents.shape[1]}, expected {expected_patches}"
    print(f"✓ Patch count matches: {expected_patches}")

    # ---- Initialise STAD ----
    print("\n" + "="*80)
    print("Initializing STAD Model")
    print("="*80)

    stad_model = STADModel(
        mae_encoder         = mae_model,
        lr_channels         = 16,
        hr_channels         = mae_config['in_chans'],
        sr_channels         = 62,
        latent_dim          = mae_config['embed_dim'],
        num_patches         = latents.shape[1],
        diffusion_schedule  = args.diffusion_schedule,
        lr_channel_indices  = train_dataset.lr_indices,
        device              = device,
    )
    stad_model = stad_model.to(device)
    print("✅ STAD model initialized.")

    if args.test_only:
        print("\n✓ Test complete (--test_only mode)")
        return

    # ---- Training summary ----
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"  Output directory:   {output_dir}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Weight decay:       {args.weight_decay}")
    print(f"  Diffusion schedule: {args.diffusion_schedule}")
    print(f"  Loss = diffusion + {args.sr_loss_weight} * MSE + {args.hfd_loss_weight} * HFD")
    print(f"  MAE frozen:         {args.freeze_mae}")
    if args.freeze_mae:
        print(f"  MAE unfreeze epoch: {args.unfreeze_mae_epoch}")
        print(f"  MAE finetune lr:    {args.mae_finetune_lr}")
    if args.resume_stad_checkpoint:
        print(f"  Resume checkpoint:  {args.resume_stad_checkpoint}")
        print(f"  Resume optimizer:   {args.resume_optimizer}")

    # ---- Train ----
    print("\n" + "="*80)
    print("Training STAD (diffusion + MSE + HFD)")
    print("="*80)
    train_stad_model(
        stad_model   = stad_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        args         = args,
        device       = device,
        output_dir   = output_dir,
    )
    print("\n✅ STAD training finished.")
    print(f"   All results saved in: {output_dir}")


if __name__ == '__main__':
    main()