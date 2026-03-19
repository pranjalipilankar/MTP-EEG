#!/usr/bin/env python3
"""
Train MAE on 31-channel HR data with K-Fold Cross-Validation
Uses subject-based k-fold splits for robust evaluation
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path

from config_seed4 import Config_MAE_SEED4
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler
from scipy.stats import pearsonr


class HR31ChannelDataset(Dataset):
    """Dataset for 31-channel HR data from preprocessed NPZ with subject tracking"""
    
    def __init__(self, npz_path, indices, subject_ids=None):
        """
        Args:
            npz_path: Path to preprocessed_data.npz
            indices: Array of indices to use
            subject_ids: Optional array of subject IDs for each sample
        """
        data = np.load(npz_path)
        self.hr_data = data['HR']  # (14280, 31, 1000)
        self.indices = indices
        self.subject_ids = subject_ids
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        eeg = self.hr_data[real_idx].astype(np.float32)
        result = {'eeg': torch.from_numpy(eeg)}
        if self.subject_ids is not None:
            result['subject_id'] = self.subject_ids[idx]
        return result


def create_subject_ids_for_preprocessed_data(data_path):
    """
    Create subject IDs for the preprocessed data by loading the original files
    Returns array of subject IDs matching the order in preprocessed_data.npz
    """
    from tqdm import tqdm
    
    data_path = Path(data_path)
    all_subject_ids = []
    
    # Load in same order as prepare_dataset.py
    for session in ['1', '2', '3']:
        session_path = data_path / session
        if not session_path.exists():
            continue
        
        subject_folders = sorted([f for f in session_path.iterdir() if f.is_dir()])
        
        for folder in tqdm(subject_folders, desc=f"Session {session}"):
            x_file = folder / 'X_prc1.npy'
            if x_file.exists():
                # Extract subject ID from folder name (e.g., '1_20160518' -> '1')
                subject_id = folder.name.split('_')[0]
                
                # Load to get number of windows
                data = np.load(x_file)
                num_windows = len(data)
                
                # Add subject ID for each window
                all_subject_ids.extend([subject_id] * num_windows)
    
    return np.array(all_subject_ids)


def create_kfold_splits(subject_ids, n_folds=5, seed=42):
    """
    Create subject-based k-fold splits
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    from sklearn.model_selection import KFold
    
    # Get unique subjects
    unique_subjects = np.unique(subject_ids)
    print(f"   Total subjects: {len(unique_subjects)}")
    
    # Create k-fold splitter
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_splits = []
    for fold_idx, (train_sub_idx, val_sub_idx) in enumerate(kfold.split(unique_subjects)):
        train_subjects = unique_subjects[train_sub_idx]
        val_subjects = unique_subjects[val_sub_idx]
        
        # Get sample indices for these subjects
        train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
        val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
        
        fold_splits.append((train_indices, val_indices, train_subjects, val_subjects))
        
        print(f"   Fold {fold_idx+1}: Train subjects={len(train_subjects)}, Val subjects={len(val_subjects)}")
        print(f"              Train samples={len(train_indices)}, Val samples={len(val_indices)}")
    
    return fold_splits


def validate_model(model, val_loader, device, mask_ratio=0.75):
    """Validate model and compute metrics"""
    model.eval()
    
    total_loss = 0.0
    all_correlations = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            batch_size = eeg.size(0)
            
            # Forward pass returns: loss, pred, mask (NOT target!)
            loss, pred, mask = model(eeg, mask_ratio=mask_ratio)
            
            # Get actual target by patchifying input
            target = model.patchify(eeg)  # [B, num_patches, C*patch_size]
            
            # Convert to numpy for correlation computation
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Compute per-sample correlation on masked patches only
            for i in range(batch_size):
                # Get masked patches (mask=1 means masked/removed)
                masked_patches = mask_np[i] == 1
                
                if masked_patches.sum() > 0:
                    pred_masked = pred_np[i][masked_patches].flatten()
                    target_masked = target_np[i][masked_patches].flatten()
                    
                    # Compute Pearson correlation manually
                    if len(pred_masked) > 1:
                        pred_mean = pred_masked.mean()
                        target_mean = target_masked.mean()
                        
                        numerator = ((pred_masked - pred_mean) * (target_masked - target_mean)).sum()
                        denom_pred = ((pred_masked - pred_mean) ** 2).sum()
                        denom_target = ((target_masked - target_mean) ** 2).sum()
                        denominator = np.sqrt(denom_pred * denom_target)
                        
                        if denominator > 1e-8:
                            corr = numerator / denominator
                            all_correlations.append(corr)
            
            total_loss += loss.item()
            total_samples += batch_size
    
    avg_loss = total_loss / len(val_loader)
    avg_correlation = np.mean(all_correlations) if all_correlations else 0.0
    
    return avg_loss, avg_correlation


def train_fold(fold_idx, train_indices, val_indices, train_subjects, val_subjects, 
               npz_path, subject_ids, config, args, device):
    """Train one fold"""
    
    print(f"\n{'='*80}")
    print(f"🔄 FOLD {fold_idx + 1}/{args.n_folds}")
    print(f"{'='*80}")
    print(f"Train subjects: {sorted(train_subjects)}")
    print(f"Val subjects: {sorted(val_subjects)}\n")
    
    # Create output directory for this fold
    fold_output_dir = Path(args.output_dir) / f'fold_{fold_idx}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_dataset = HR31ChannelDataset(npz_path, train_indices, subject_ids[train_indices])
    val_dataset = HR31ChannelDataset(npz_path, val_indices, subject_ids[val_indices])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📊 Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    
    # Build model
    print(f"\n🏗️  Building MAE model...")
    model = MAEforEEG(
        time_len=1000,
        patch_size=8,
        embed_dim=768,
        in_chans=31,  # 31 channels
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0
    ).to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_parameters / 1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Scaler
    loss_scaler = NativeScaler()
    
    # Training loop
    best_val_cor = -float('inf')
    best_epoch = 0
    fold_history = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_cor = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            config=config
        )
        
        scheduler.step()
        
        # Validate
        val_loss, val_cor = validate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.6f}, Cor: {train_cor:.6f}")
        print(f"  Val   - Loss: {val_loss:.6f}, Cor: {val_cor:.6f}")
        
        # Save history
        fold_history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_cor': float(train_cor),
            'val_loss': float(val_loss),
            'val_cor': float(val_cor)
        })
        
        # Save best model
        if val_cor > best_val_cor:
            best_val_cor = val_cor
            best_epoch = epoch
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_cor': val_cor,
                'fold': fold_idx,
                'train_subjects': train_subjects.tolist(),
                'val_subjects': val_subjects.tolist(),
                'config': {
                    'time_len': 1000,
                    'patch_size': 8,
                    'embed_dim': 768,
                    'in_chans': 31,
                    'depth': 12,
                    'num_heads': 12,
                    'decoder_embed_dim': 384,
                    'decoder_depth': 4,
                    'mlp_ratio': 4.0
                }
            }
            
            save_path = fold_output_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"  ✅ SAVED best model (cor={val_cor:.6f})")
    
    # Save fold history
    history_path = fold_output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(fold_history, f, indent=2)
    
    print(f"\n✅ Fold {fold_idx+1} complete!")
    print(f"   Best Val Cor: {best_val_cor:.6f} (Epoch {best_epoch+1})")
    
    return {
        'fold': fold_idx,
        'best_val_cor': best_val_cor,
        'best_epoch': best_epoch,
        'train_subjects': train_subjects.tolist(),
        'val_subjects': val_subjects.tolist()
    }


def main():
    parser = argparse.ArgumentParser('MAE 31-channel K-Fold training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./results_31ch_kfold')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MAE Training with {args.n_folds}-Fold Cross-Validation")
    print(f"   31-channel HR data (evenly distributed)")
    print(f"   Device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = Config_MAE_SEED4()
    config.num_channels = 31
    config.batch_size = args.batch_size
    config.num_epoch = args.epochs
    config.lr = args.lr
    config.warmup_epochs = args.warmup_epochs
    config.accum_iter = 1
    
    # Paths
    npz_path = '/home/ab_students/EEG-MTP/DATA/seed4/preprocessed_data.npz'
    data_root = '/home/ab_students/EEG-MTP/DATA/seed4/eeg_processed_data'
    
    # Create subject IDs
    print(f"📊 Creating subject IDs from original data...")
    subject_ids = create_subject_ids_for_preprocessed_data(data_root)
    print(f"   Total samples: {len(subject_ids)}")
    print(f"   Unique subjects: {len(np.unique(subject_ids))}\n")
    
    # Create k-fold splits
    print(f"🔀 Creating {args.n_folds}-fold splits...")
    fold_splits = create_kfold_splits(subject_ids, n_folds=args.n_folds, seed=args.seed)
    
    # Train each fold
    all_fold_results = []
    
    for fold_idx, (train_indices, val_indices, train_subjects, val_subjects) in enumerate(fold_splits):
        fold_result = train_fold(
            fold_idx=fold_idx,
            train_indices=train_indices,
            val_indices=val_indices,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            npz_path=npz_path,
            subject_ids=subject_ids,
            config=config,
            args=args,
            device=device
        )
        all_fold_results.append(fold_result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"🎉 K-FOLD TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    for result in all_fold_results:
        print(f"Fold {result['fold']+1}: Best Val Cor = {result['best_val_cor']:.6f} (Epoch {result['best_epoch']+1})")
    
    avg_cor = np.mean([r['best_val_cor'] for r in all_fold_results])
    std_cor = np.std([r['best_val_cor'] for r in all_fold_results])
    
    print(f"\nAverage Val Correlation: {avg_cor:.6f} ± {std_cor:.6f}")
    
    # Save summary
    summary = {
        'n_folds': args.n_folds,
        'avg_val_cor': float(avg_cor),
        'std_val_cor': float(std_cor),
        'fold_results': [
            {
                'fold': int(r['fold']),
                'best_val_cor': float(r['best_val_cor']),
                'best_epoch': int(r['best_epoch']),
                'train_subjects': r['train_subjects'],
                'val_subjects': r['val_subjects']
            }
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
