#!/usr/bin/env python3
"""
Train MAE on DEAP data with K-Fold Cross-Validation.
Uses subject-based k-fold splits for robust evaluation.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path

# Assuming similar project structure for config and model
from config_deap import Config_MAE_DEAP
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


class DEAPSubjectDataset(Dataset):
    """Dataset built from raw DEAP subject files (sXX.dat)."""

    def __init__(self, data_dir, subject_ids, eeg_only=True, normalize_per_trial=True):
        self.data_dir = Path(data_dir)
        self.subject_ids = [str(s) for s in subject_ids]
        self.eeg_only = eeg_only
        self.normalize_per_trial = normalize_per_trial

        all_trials = []
        all_sample_subjects = []

        for sid in self.subject_ids:
            file_name = sid if sid.endswith('.dat') else f"{sid}.dat"
            file_path = self.data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Missing DEAP subject file: {file_path}")

            with open(file_path, 'rb') as f:
                payload = pickle.load(f, encoding='latin1')

            if 'data' not in payload:
                raise KeyError(f"'data' key not found in {file_path}")

            trials = np.asarray(payload['data'], dtype=np.float32)
            if trials.ndim != 3:
                raise ValueError(f"Expected shape (N, C, T) in {file_path}, got {trials.shape}")

            # DEAP standard files contain 40 channels; first 32 are EEG.
            if self.eeg_only:
                if trials.shape[1] < 32:
                    raise ValueError(f"Expected at least 32 channels in {file_path}, got {trials.shape[1]}")
                trials = trials[:, :32, :]

            all_trials.append(trials)
            all_sample_subjects.extend([sid.replace('.dat', '')] * trials.shape[0])

        if len(all_trials) == 0:
            raise ValueError(f"No DEAP trials loaded from {self.data_dir}")

        self.eeg_data = np.concatenate(all_trials, axis=0)
        self.sample_subject_ids = np.asarray(all_sample_subjects)

        if self.normalize_per_trial:
            mean = self.eeg_data.mean(axis=(1, 2), keepdims=True)
            std = self.eeg_data.std(axis=(1, 2), keepdims=True) + 1e-8
            self.eeg_data = (self.eeg_data - mean) / std

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx].astype(np.float32)
        return {
            'eeg': torch.from_numpy(eeg),
            'subject_id': self.sample_subject_ids[idx],
        }


def discover_deap_subjects(data_dir):
    """Discover DEAP subject IDs from sXX.dat files."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"DEAP data directory not found: {data_dir}")

    subject_ids = []
    for p in sorted(data_dir.glob('s*.dat')):
        subject_ids.append(p.stem)

    if len(subject_ids) == 0:
        raise ValueError(f"No DEAP subject files found in {data_dir} (expected sXX.dat)")

    return subject_ids


def create_kfold_splits(subject_ids, n_folds=5, seed=42):
    """
    Create subject-based k-fold splits.
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold.
    """
    unique_subjects = np.unique(subject_ids)
    print(f"   Total subjects: {len(unique_subjects)}")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_splits = []
    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(kfold.split(unique_subjects)):
        train_subjects = unique_subjects[train_sub_idx]
        val_subjects = unique_subjects[val_sub_idx]

        fold_splits.append((train_subjects, val_subjects))
        
        print(f"   Fold {fold_idx+1}: Train subjects={len(train_subjects)}, Val subjects={len(val_subjects)}")
    
    return fold_splits


def validate_model(model, val_loader, device, mask_ratio=0.75):
    """Validate model and compute metrics"""
    model.eval()
    
    total_loss = 0.0
    all_correlations = []
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            
            loss, pred, mask = model(eeg, mask_ratio=mask_ratio)
            
            target = model.patchify(eeg)
            
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i in range(eeg.size(0)):
                masked_patches = mask_np[i] == 1
                
                if masked_patches.sum() > 0:
                    pred_masked = pred_np[i][masked_patches].flatten()
                    target_masked = target_np[i][masked_patches].flatten()
                    
                    if len(pred_masked) > 1:
                        corr, _ = pearsonr(pred_masked, target_masked)
                        if not np.isnan(corr):
                            all_correlations.append(corr)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_correlation = np.mean(all_correlations) if all_correlations else 0.0
    
    return avg_loss, avg_correlation


def train_fold(fold_idx, train_subjects, val_subjects, data_dir, config, args, device):
    """Train one fold"""
    
    print(f"\n{'='*80}")
    print(f"🔄 FOLD {fold_idx + 1}/{args.n_folds}")
    print(f"{'='*80}")
    print(f"Train subjects: {sorted(train_subjects)}")
    print(f"Val subjects: {sorted(val_subjects)}\n")
    
    fold_output_dir = Path(args.output_dir) / f'fold_{fold_idx}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = DEAPSubjectDataset(
        data_dir=data_dir,
        subject_ids=train_subjects,
        eeg_only=(not args.keep_non_eeg_channels),
        normalize_per_trial=True,
    )
    val_dataset = DEAPSubjectDataset(
        data_dir=data_dir,
        subject_ids=val_subjects,
        eeg_only=(not args.keep_non_eeg_channels),
        normalize_per_trial=True,
    )

    # Align model dimensions to actual loaded data.
    config.num_channels = int(train_dataset.eeg_data.shape[1])
    config.time_len = int(train_dataset.eeg_data.shape[2])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    
    print(f"\n🏗️  Building MAE model...")
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    ).to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_parameters / 1e6:.1f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_scaler = NativeScaler()
    
    best_val_cor = -float('inf')
    best_epoch = 0
    fold_history = []
    
    for epoch in range(args.epochs):
        train_loss, train_cor = train_one_epoch(
            model=model, data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, loss_scaler=loss_scaler, config=config
        )
        scheduler.step()
        
        val_loss, val_cor = validate_model(model, val_loader, device, mask_ratio=config.mask_ratio)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.6f}, Cor: {train_cor:.6f}")
        print(f"  Val   - Loss: {val_loss:.6f}, Cor: {val_cor:.6f}")
        
        fold_history.append({
            'epoch': epoch, 'train_loss': float(train_loss), 'train_cor': float(train_cor),
            'val_loss': float(val_loss), 'val_cor': float(val_cor)
        })
        
        if val_cor > best_val_cor:
            best_val_cor = val_cor
            best_epoch = epoch
            
            checkpoint = {
                'model': model.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'val_loss': val_loss, 'val_cor': val_cor, 'fold': fold_idx,
                'train_subjects': train_subjects.tolist(), 'val_subjects': val_subjects.tolist(),
                'config': {
                    'time_len': config.time_len, 'patch_size': config.patch_size,
                    'embed_dim': config.embed_dim, 'in_chans': config.num_channels,
                    'depth': config.depth, 'num_heads': config.num_heads,
                    'decoder_embed_dim': config.decoder_embed_dim, 'decoder_depth': config.decoder_depth,
                    'mlp_ratio': config.mlp_ratio
                }
            }
            
            save_path = fold_output_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"  ✅ SAVED best model (cor={val_cor:.6f})")
    
    history_path = fold_output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(fold_history, f, indent=2)
    
    print(f"\n✅ Fold {fold_idx+1} complete!")
    print(f"   Best Val Cor: {best_val_cor:.6f} (Epoch {best_epoch+1})")
    
    return {
        'fold': fold_idx, 'best_val_cor': best_val_cor, 'best_epoch': best_epoch,
        'train_subjects': train_subjects.tolist(), 'val_subjects': val_subjects.tolist()
    }


def main():
    parser = argparse.ArgumentParser('MAE DEAP K-Fold training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./results_deap_kfold')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='/DATA/EEG-MTP/DEAP',
                        help='Path to raw DEAP subject files (s01.dat ... s32.dat)')
    parser.add_argument('--keep_non_eeg_channels', action='store_true',
                        help='Use all channels from DEAP file instead of first 32 EEG channels')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MAE Training with {args.n_folds}-Fold Cross-Validation for DEAP")
    print(f"   Device: {device}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config_MAE_DEAP()
    config.batch_size = args.batch_size
    config.num_epoch = args.epochs
    config.lr = args.lr
    config.warmup_epochs = args.warmup_epochs
    config.data_path = args.data_dir
    
    print(f"📊 Discovering DEAP subjects from {args.data_dir}...")
    subject_ids = discover_deap_subjects(args.data_dir)
    print(f"   Found subjects: {len(subject_ids)}")
    print(f"   Subject IDs: {subject_ids}\n")
    
    print(f"🔀 Creating {args.n_folds}-fold splits...")
    fold_splits = create_kfold_splits(subject_ids, n_folds=args.n_folds, seed=args.seed)
    
    fold_split_records = []
    all_fold_results = []
    for fold_idx, (train_subjects, val_subjects) in enumerate(fold_splits):
        fold_split_records.append({
            'fold': fold_idx + 1,
            'train_subjects': train_subjects.tolist(),
            'val_subjects': val_subjects.tolist(),
        })
        fold_result = train_fold(
            fold_idx=fold_idx,
            train_subjects=train_subjects, val_subjects=val_subjects,
            data_dir=args.data_dir,
            config=config, args=args, device=device,
        )
        all_fold_results.append(fold_result)

    fold_splits_path = output_dir / 'fold_splits.json'
    with open(fold_splits_path, 'w') as f:
        json.dump(fold_split_records, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎉 K-FOLD TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    for result in all_fold_results:
        print(f"Fold {result['fold']+1}: Best Val Cor = {result['best_val_cor']:.6f} (Epoch {result['best_epoch']+1})")
    
    avg_cor = np.mean([r['best_val_cor'] for r in all_fold_results])
    std_cor = np.std([r['best_val_cor'] for r in all_fold_results])
    
    print(f"\nAverage Val Correlation: {avg_cor:.6f} ± {std_cor:.6f}")
    
    summary = {
        'n_folds': args.n_folds, 'avg_val_cor': float(avg_cor), 'std_val_cor': float(std_cor),
        'data_dir': args.data_dir,
        'fold_results': [
            {'fold': int(r['fold']), 'best_val_cor': float(r['best_val_cor']), 'best_epoch': int(r['best_epoch']),
             'train_subjects': r['train_subjects'], 'val_subjects': r['val_subjects']}
            for r in all_fold_results
        ],
        'args': vars(args)
    }
    
    summary_path = output_dir / 'kfold_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()