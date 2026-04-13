#!/usr/bin/env python3
"""
K-Fold MAE training for DEAP in a style similar to SEED4 k-fold scripts.

The script prefers subject-based folds when subject IDs are available in the
NPZ; otherwise it falls back to sample-based K-Fold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from config_deap import Config_MAE_DEAP
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler


FIXED_MASK_RATIO = 0.75


def _to_str_array(arr):
    return np.asarray(arr).astype(str)


def _load_deap_array(data, exclude_test_split=True):
    """Load DEAP array and optional IDs from NPZ with multiple schema options."""
    test_meta = {
        'used_fixed_test_split': False,
        'heldout_test_samples': 0,
        'heldout_test_subjects': None,
    }

    if 'X' in data.files:
        x = data['X']
        subject_ids = data['subject_ids'] if 'subject_ids' in data.files else None
        return x, subject_ids, test_meta

    split_keys = [k for k in ('X_train', 'X_val', 'X_test') if k in data.files]
    if len(split_keys) == 0:
        raise KeyError(
            "Could not find EEG data in NPZ. Expected one of: X, X_train/X_val/X_test"
        )

    if exclude_test_split and 'X_train' in data.files and 'X_val' in data.files and 'X_test' in data.files:
        x = np.concatenate([data['X_train'], data['X_val']], axis=0)
        test_meta['used_fixed_test_split'] = True
        test_meta['heldout_test_samples'] = int(len(data['X_test']))
        if 'subject_ids_test' in data.files:
            test_meta['heldout_test_subjects'] = sorted(np.unique(_to_str_array(data['subject_ids_test'])).tolist())

        if 'subject_ids_train' in data.files and 'subject_ids_val' in data.files:
            subject_ids = np.concatenate([data['subject_ids_train'], data['subject_ids_val']], axis=0)
        elif 'subject_ids' in data.files and len(data['subject_ids']) == len(x):
            subject_ids = data['subject_ids']
        else:
            subject_ids = None

        return x, subject_ids, test_meta

    x = np.concatenate([data[k] for k in split_keys], axis=0)

    subj_split_keys = [k for k in ('subject_ids_train', 'subject_ids_val', 'subject_ids_test') if k in data.files]
    if len(subj_split_keys) == len(split_keys):
        subject_ids = np.concatenate([data[k] for k in ('subject_ids_train', 'subject_ids_val', 'subject_ids_test')], axis=0)
    elif 'subject_ids' in data.files and len(data['subject_ids']) == len(x):
        subject_ids = data['subject_ids']
    else:
        subject_ids = None

    return x, subject_ids, test_meta


class DEAPKFoldDataset(Dataset):
    """Minimal dataset wrapper for MAE pretraining."""

    def __init__(
        self,
        eeg_array,
        indices,
        norm_mode='global_zscore',
        norm_clip=0.0,
        segment_len=1024,
        is_train=True,
        seed=2024,
    ):
        self.eeg = eeg_array[indices].astype(np.float32)
        self.norm_mode = norm_mode
        self.norm_clip = norm_clip
        self.segment_len = int(segment_len) if segment_len is not None else 0
        self.is_train = is_train
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        sample = self.eeg[idx]

        if self.segment_len > 0 and sample.shape[1] > self.segment_len:
            max_start = sample.shape[1] - self.segment_len
            if self.is_train:
                start = self.rng.randint(0, max_start + 1)
            else:
                start = max_start // 2
            sample = sample[:, start:start + self.segment_len]

        if self.norm_mode == 'global_zscore':
            mean = sample.mean()
            std = sample.std() + 1e-8
            sample = (sample - mean) / std
        elif self.norm_mode == 'channel_zscore':
            mean = sample.mean(axis=1, keepdims=True)
            std = sample.std(axis=1, keepdims=True) + 1e-8
            sample = (sample - mean) / std
        elif self.norm_mode == 'none':
            pass
        else:
            raise ValueError(f"Unsupported norm_mode: {self.norm_mode}")

        if self.norm_mode != 'none' and self.norm_clip is not None and self.norm_clip > 0:
            sample = np.clip(sample, -self.norm_clip, self.norm_clip)

        return {'eeg': torch.from_numpy(sample).float()}


def create_fold_splits(subject_ids, num_samples, n_folds=5, seed=2024):
    """
    Create fold splits.
    - subject_ids provided: subject-based folds (no leakage)
    - subject_ids missing: sample-based folds
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if subject_ids is None:
        indices = np.arange(num_samples)
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            folds.append({
                'fold': fold_idx + 1,
                'train_indices': train_idx,
                'val_indices': val_idx,
                'train_subjects': None,
                'val_subjects': None,
                'split_type': 'sample_based',
            })
        return folds

    subject_ids = _to_str_array(subject_ids)
    unique_subjects = np.unique(subject_ids)

    folds = []
    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(kf.split(unique_subjects)):
        train_subjects = unique_subjects[train_sub_idx]
        val_subjects = unique_subjects[val_sub_idx]

        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
        val_idx = np.where(np.isin(subject_ids, val_subjects))[0]

        folds.append({
            'fold': fold_idx + 1,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'train_subjects': train_subjects.tolist(),
            'val_subjects': val_subjects.tolist(),
            'split_type': 'subject_based',
        })

    return folds


def validate_fold_integrity(fold_info, subject_ids=None):
    """Guard against train/val leakage inside each fold."""
    train_idx = np.asarray(fold_info['train_indices'])
    val_idx = np.asarray(fold_info['val_indices'])

    overlap = np.intersect1d(train_idx, val_idx)
    if overlap.size > 0:
        raise ValueError(
            f"Fold {fold_info['fold']} has {overlap.size} overlapping train/val samples."
        )

    if subject_ids is not None and fold_info['split_type'] == 'subject_based':
        sid = _to_str_array(subject_ids)
        train_subj = set(np.unique(sid[train_idx]).tolist())
        val_subj = set(np.unique(sid[val_idx]).tolist())
        shared = sorted(train_subj.intersection(val_subj))
        if shared:
            raise ValueError(
                f"Fold {fold_info['fold']} has subject leakage across train/val: {shared}"
            )


def _mean_reconstruction_correlation(pred_patches, target_signals, model):
    """Compute mean Pearson correlation on full reconstructed signals."""
    pred = model.unpatchify(pred_patches).detach().cpu()
    target = target_signals.detach().cpu()

    correlations = []
    for p, t in zip(pred, target):
        p_flat = p.reshape(-1)
        t_flat = t.reshape(-1)

        if torch.std(p_flat) <= 1e-6 or torch.std(t_flat) <= 1e-6:
            continue

        p_norm = (p_flat - p_flat.mean()) / (p_flat.std() + 1e-8)
        t_norm = (t_flat - t_flat.mean()) / (t_flat.std() + 1e-8)
        correlations.append((p_norm * t_norm).mean())

    if len(correlations) == 0:
        return 0.0
    return float(torch.stack(correlations).mean().item())


def _masked_patch_correlation(pred_patches, target_patches, mask):
    """Compute mean Pearson correlation on masked patches only, matching SEED4."""
    pred_np = pred_patches.detach().cpu().numpy()
    target_np = target_patches.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    correlations = []
    for i in range(pred_np.shape[0]):
        masked_patches = mask_np[i] == 1
        if masked_patches.sum() <= 0:
            continue

        pred_masked = pred_np[i][masked_patches].flatten()
        target_masked = target_np[i][masked_patches].flatten()

        if len(pred_masked) <= 1:
            continue

        pred_mean = pred_masked.mean()
        target_mean = target_masked.mean()
        numerator = ((pred_masked - pred_mean) * (target_masked - target_mean)).sum()
        denom_pred = ((pred_masked - pred_mean) ** 2).sum()
        denom_target = ((target_masked - target_mean) ** 2).sum()
        denominator = np.sqrt(denom_pred * denom_target)

        if denominator > 1e-8:
            correlations.append(numerator / denominator)

    if len(correlations) == 0:
        return 0.0
    return float(np.mean(correlations))


def _compact_state_dict(model):
    """Create a smaller CPU state dict for checkpointing."""
    compact = {}
    for name, tensor in model.state_dict().items():
        tensor_cpu = tensor.detach().cpu()
        if tensor_cpu.is_floating_point():
            compact[name] = tensor_cpu.to(dtype=torch.float16)
        else:
            compact[name] = tensor_cpu
    return compact


def validate_model(model, val_loader, device, mask_ratio=0.0):
    """Validate MAE using masked patches, matching the SEED4 validation path."""
    model.eval()

    total_loss = 0.0
    all_correlations = []

    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)

            loss, pred, mask = model(eeg, mask_ratio=mask_ratio)
            target = model.patchify(eeg)
            all_correlations.append(_masked_patch_correlation(pred, target, mask))

            total_loss += loss.item()

    avg_loss = total_loss / max(len(val_loader), 1)
    avg_corr = float(np.mean(all_correlations)) if all_correlations else 0.0
    return float(avg_loss), avg_corr


def build_model(config, device):
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        norm_pix_loss=getattr(config, 'norm_pix_loss', True),
    ).to(device)
    return model


def save_val_corr_curve(history, out_path):
    """Save per-epoch validation correlation curve for quick diagnosis."""
    if plt is None:
        return

    epochs = [h['epoch'] for h in history]
    vals = [h['val_corr'] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, vals, linewidth=2)
    plt.axhline(y=0.02, linestyle='--', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Correlation')
    plt.title('Validation Correlation Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_one_fold(fold_info, eeg_array, config, args, device, output_dir):
    fold_num = fold_info['fold']
    train_idx = fold_info['train_indices']
    val_idx = fold_info['val_indices']

    print("\n" + "=" * 80)
    print(f"Fold {fold_num}/{args.n_folds}")
    print("=" * 80)
    print(f"Split type: {fold_info['split_type']}")
    if fold_info['train_subjects'] is not None:
        print(f"Train subjects: {fold_info['train_subjects']}")
        print(f"Val subjects: {fold_info['val_subjects']}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")

    train_ds = DEAPKFoldDataset(
        eeg_array,
        train_idx,
        norm_mode=args.norm_mode,
        norm_clip=args.norm_clip,
        segment_len=args.segment_len,
        is_train=True,
        seed=args.seed + fold_num,
    )
    val_ds = DEAPKFoldDataset(
        eeg_array,
        val_idx,
        norm_mode=args.norm_mode,
        norm_clip=args.norm_clip,
        segment_len=args.segment_len,
        is_train=False,
        seed=args.seed + 10_000 + fold_num,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = build_model(config, device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    loss_scaler = NativeScaler()

    best_val_corr = -float('inf')
    best_epoch = -1
    history = []
    stopped_early_by_gate = False

    fold_dir = output_dir / f"fold_{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_corr = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            config=config,
            model_without_ddp=model,
        )
        scheduler.step()

        val_loss, val_corr = validate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            mask_ratio=args.val_mask_ratio,
        )

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_corr': float(train_corr),
            'val_loss': float(val_loss),
            'val_corr': float(val_corr),
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss {train_loss:.6f}, Train Corr {train_corr:.4f} | "
            f"Val Loss {val_loss:.6f}, Val Corr {val_corr:.4f}"
        )

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch + 1
            ckpt = {
                'epoch': epoch + 1,
                'checkpoint_format': 'model_state_dict_fp16',
                'model_state_dict': _compact_state_dict(model),
                'val_loss': float(val_loss),
                'val_corr': float(val_corr),
                'fold': fold_num,
                'split_type': fold_info['split_type'],
                'train_subjects': fold_info['train_subjects'],
                'val_subjects': fold_info['val_subjects'],
                'config': {
                    'time_len': config.time_len,
                    'patch_size': config.patch_size,
                    'embed_dim': config.embed_dim,
                    'in_chans': config.num_channels,
                    'depth': config.depth,
                    'num_heads': config.num_heads,
                    'decoder_embed_dim': config.decoder_embed_dim,
                    'decoder_depth': config.decoder_depth,
                    'decoder_num_heads': config.decoder_num_heads,
                    'mlp_ratio': config.mlp_ratio,
                    'norm_pix_loss': bool(config.norm_pix_loss),
                    'mask_ratio': config.mask_ratio,
                },
            }
            torch.save(ckpt, fold_dir / 'best_model.pth')
            print(f"Saved best model at epoch {best_epoch} with val corr {best_val_corr:.4f}")

        if args.early_gate_epoch > 0 and (epoch + 1) == args.early_gate_epoch and best_val_corr < args.early_gate_corr:
            stopped_early_by_gate = True
            print(
                f"Stopping fold early: best val corr {best_val_corr:.4f} < "
                f"gate {args.early_gate_corr:.4f} at epoch {args.early_gate_epoch}"
            )
            break

    with open(fold_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    save_val_corr_curve(history, fold_dir / 'val_corr_curve.png')

    return {
        'fold': fold_num,
        'split_type': fold_info['split_type'],
        'best_epoch': best_epoch,
        'best_val_corr': float(best_val_corr),
        'stopped_early_by_gate': bool(stopped_early_by_gate),
        'train_samples': int(len(train_idx)),
        'val_samples': int(len(val_idx)),
        'train_subjects': fold_info['train_subjects'],
        'val_subjects': fold_info['val_subjects'],
    }


def parse_args():
    parser = argparse.ArgumentParser('DEAP MAE K-Fold Training')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to DEAP NPZ (supports X or X_train/X_val/X_test format)')
    parser.add_argument('--output_dir', type=str, default='./results_kfold')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Temporal patch size (larger patches reduce sequence length)')
    parser.add_argument('--segment_len', type=int, default=8064,
                        help='Temporal segment length sampled from DEAP trials (0 to use full length)')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='MAE encoder embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='MAE encoder depth')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='MAE encoder attention heads')
    parser.add_argument('--decoder_embed_dim', type=int, default=384,
                        help='MAE decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=4,
                        help='MAE decoder depth')
    parser.add_argument('--decoder_num_heads', type=int, default=8,
                        help='MAE decoder attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='Transformer MLP expansion ratio')
    parser.add_argument('--disable_norm_pix_loss', action='store_true',
                        help='Disable MAE normalized-pixel loss (enabled by default)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mask_ratio', type=float, default=FIXED_MASK_RATIO,
                        help='MAE mask ratio; lower values often help EEG reconstruction')
    parser.add_argument('--val_mask_ratio', type=float, default=FIXED_MASK_RATIO,
                        help='Mask ratio used only during validation (set to 0.75 to match SEED4-style validation)')
    parser.add_argument('--corr_loss_weight', type=float, default=0.1,
                        help='Weight for differentiable correlation auxiliary loss (0 to disable)')
    parser.add_argument('--norm_mode', type=str, default='global_zscore',
                        choices=['none', 'global_zscore', 'channel_zscore'],
                        help='Input normalization used in the k-fold dataset loader')
    parser.add_argument('--norm_clip', type=float, default=5.0,
                        help='Clip threshold applied after normalization (set <=0 to disable)')
    parser.add_argument('--even_channels_only', action='store_true',
                        help='Use only even-indexed EEG channels (0,2,...,30) resulting in 16 channels for DEAP')
    parser.add_argument('--include_test_in_kfold_pool', action='store_true',
                        help='If set, include X_test in the k-fold pool. Default excludes test split to prevent leakage.')
    parser.add_argument('--early_gate_epoch', type=int, default=0,
                        help='Epoch at which to apply minimum validation correlation gate (<=0 disables gate)')
    parser.add_argument('--early_gate_corr', type=float, default=0.02,
                        help='Minimum best validation correlation required by early_gate_epoch')
    parser.add_argument('--sanity_run', action='store_true',
                        help='Run only one fixed fold for quick sanity check')
    parser.add_argument('--sanity_fold', type=int, default=1,
                        help='1-based fold index used when --sanity_run is enabled')
    parser.add_argument('--sanity_epochs', type=int, default=20,
                        help='Epochs used when --sanity_run is enabled')
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config_MAE_DEAP()
    if args.data_path is not None:
        config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.num_epoch = args.epochs
    config.lr = args.lr
    config.min_lr = args.min_lr
    config.weight_decay = args.weight_decay
    config.seed = args.seed
    config.mask_ratio = args.mask_ratio
    config.corr_loss_weight = args.corr_loss_weight
    config.patch_size = args.patch_size
    config.time_len = args.segment_len if args.segment_len > 0 else config.time_len
    config.embed_dim = args.embed_dim
    config.depth = args.depth
    config.num_heads = args.num_heads
    config.decoder_embed_dim = args.decoder_embed_dim
    config.decoder_depth = args.decoder_depth
    config.decoder_num_heads = args.decoder_num_heads
    config.mlp_ratio = args.mlp_ratio
    config.norm_pix_loss = not args.disable_norm_pix_loss

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data path: {config.data_path}")
    print(f"Normalization mode: {args.norm_mode} (clip={args.norm_clip})")
    print(f"Train mask ratio: {config.mask_ratio}")
    print(f"Validation mask ratio: {args.val_mask_ratio}")
    print(
        f"Model config: patch_size={config.patch_size}, mask_ratio={config.mask_ratio}, "
        f"segment_len={args.segment_len}, "
        f"embed_dim={config.embed_dim}, depth={config.depth}, num_heads={config.num_heads}, "
        f"decoder_embed_dim={config.decoder_embed_dim}, decoder_depth={config.decoder_depth}, "
        f"decoder_num_heads={config.decoder_num_heads}, mlp_ratio={config.mlp_ratio}, "
        f"norm_pix_loss={config.norm_pix_loss}, corr_loss_weight={config.corr_loss_weight}"
    )

    npz_path = Path(config.data_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"DEAP NPZ not found: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    eeg_array, subject_ids, test_meta = _load_deap_array(
        payload,
        exclude_test_split=(not args.include_test_in_kfold_pool),
    )
    print(f"Loaded EEG data: {eeg_array.shape}")
    if test_meta['used_fixed_test_split']:
        print(
            f"Using train+val pool only. Held out fixed test split: "
            f"{test_meta['heldout_test_samples']} samples"
        )
        if test_meta['heldout_test_subjects'] is not None:
            print(f"Held-out test subjects: {test_meta['heldout_test_subjects']}")

    if subject_ids is None:
        print("No subject IDs found in NPZ. Falling back to sample-based K-Fold.")
    else:
        print(f"Subject IDs found. Unique subjects: {len(np.unique(_to_str_array(subject_ids)))}")

    if eeg_array.ndim != 3:
        raise ValueError(f"Expected EEG shape (N, C, T), got {eeg_array.shape}")

    if args.even_channels_only:
        if eeg_array.shape[1] < 2:
            raise ValueError(f"Need at least 2 channels for even-channel selection, got C={eeg_array.shape[1]}")
        even_idx = np.arange(0, eeg_array.shape[1], 2)
        eeg_array = eeg_array[:, even_idx, :]
        print(f"Using even-indexed channels only: {even_idx.tolist()}")
        print(f"EEG shape after channel selection: {eeg_array.shape}")

    if eeg_array.shape[1] != config.num_channels:
        if args.even_channels_only:
            print(
                f"Info: config num_channels={config.num_channels}, "
                f"even-channel mode produced C={eeg_array.shape[1]}. Using data channels."
            )
        else:
            print(
                f"Warning: config num_channels={config.num_channels}, "
                f"but data has C={eeg_array.shape[1]}. Using data channels."
            )
        config.num_channels = eeg_array.shape[1]

    if args.segment_len <= 0 and eeg_array.shape[2] != config.time_len:
        print(
            f"Warning: config time_len={config.time_len}, "
            f"but data has T={eeg_array.shape[2]}. Using data time length."
        )
        config.time_len = eeg_array.shape[2]

    folds = create_fold_splits(
        subject_ids=subject_ids,
        num_samples=len(eeg_array),
        n_folds=args.n_folds,
        seed=args.seed,
    )
    for fold_info in folds:
        validate_fold_integrity(fold_info, subject_ids=subject_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'fold_splits.json', 'w', encoding='utf-8') as f:
        json.dump([
            {
                'fold': int(info['fold']),
                'split_type': info['split_type'],
                'train_subjects': info['train_subjects'],
                'val_subjects': info['val_subjects'],
                'train_size': int(len(info['train_indices'])),
                'val_size': int(len(info['val_indices'])),
            }
            for info in folds
        ], f, indent=2)

    if args.sanity_run:
        sanity_fold = max(1, min(args.sanity_fold, len(folds)))
        folds = [folds[sanity_fold - 1]]
        args.epochs = args.sanity_epochs
        print(
            f"Sanity run active: fold={sanity_fold}, epochs={args.epochs}, "
            f"early_gate=({args.early_gate_epoch}, {args.early_gate_corr})"
        )

    fold_results = []
    for fold_info in folds:
        result = train_one_fold(
            fold_info=fold_info,
            eeg_array=eeg_array,
            config=config,
            args=args,
            device=device,
            output_dir=output_dir,
        )
        fold_results.append(result)

    avg_corr = float(np.mean([x['best_val_corr'] for x in fold_results]))
    std_corr = float(np.std([x['best_val_corr'] for x in fold_results]))

    summary = {
        'n_folds': int(args.n_folds),
        'sanity_run': bool(args.sanity_run),
        'even_channels_only': bool(args.even_channels_only),
        'norm_mode': args.norm_mode,
        'norm_clip': float(args.norm_clip),
        'segment_len': int(args.segment_len),
        'model': {
            'patch_size': int(config.patch_size),
            'mask_ratio': float(config.mask_ratio),
            'embed_dim': int(config.embed_dim),
            'depth': int(config.depth),
            'num_heads': int(config.num_heads),
            'decoder_embed_dim': int(config.decoder_embed_dim),
            'decoder_depth': int(config.decoder_depth),
            'decoder_num_heads': int(config.decoder_num_heads),
            'mlp_ratio': float(config.mlp_ratio),
            'norm_pix_loss': bool(config.norm_pix_loss),
        },
        'early_gate_epoch': int(args.early_gate_epoch),
        'early_gate_corr': float(args.early_gate_corr),
        'used_fixed_test_split': bool(test_meta['used_fixed_test_split']),
        'heldout_test_samples': int(test_meta['heldout_test_samples']),
        'heldout_test_subjects': test_meta['heldout_test_subjects'],
        'average_best_val_corr': avg_corr,
        'std_best_val_corr': std_corr,
        'fold_results': fold_results,
    }
    with open(output_dir / 'kfold_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DEAP K-Fold Training Summary")
    print("=" * 80)
    for fr in fold_results:
        print(
            f"Fold {fr['fold']}: best val corr={fr['best_val_corr']:.4f} "
            f"(epoch {fr['best_epoch']}), train={fr['train_samples']}, val={fr['val_samples']}"
        )
    print(f"Average best val corr: {avg_corr:.4f} +/- {std_corr:.4f}")
    print(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
