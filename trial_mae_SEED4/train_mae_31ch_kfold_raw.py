#!/usr/bin/env python3
"""
K-Fold Cross-Validation Training for MAE on RAW SEED-IV (31-Channel HR)
Comparison with preprocessed data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local imports
sys.path.append('/home/ab_students/EEG-MTP/New_SEED4')
from mae_for_eeg import MAEforEEG


# ===== Dataset =====
class SEED4RawHRDataset(Dataset):
    """31-channel HR dataset from raw SEED-IV data"""
    
    def __init__(self, data_path, subject_ids=None):
        self.data_path = Path(data_path)
        
        # Load raw multi-resolution data
        print(f"Loading RAW data from {self.data_path}")
        data = np.load(self.data_path)
        
        # Use HR (31 channels)
        self.hr_data = data['HR']  # (N, 31, 1000)
        self.labels = data['labels']
        all_subject_ids = data['subject_ids']
        
        # Filter by subject_ids if provided
        if subject_ids is not None:
            subject_ids_str = [str(sid) for sid in subject_ids]
            mask = np.isin(all_subject_ids, subject_ids_str)
            self.hr_data = self.hr_data[mask]
            self.labels = self.labels[mask]
            self.subject_ids = all_subject_ids[mask]
        else:
            self.subject_ids = all_subject_ids
        
        # Normalize per-sample
        self.hr_data = self.normalize_data(self.hr_data)
        
        print(f"  HR data shape: {self.hr_data.shape}")
        print(f"  Subjects: {np.unique(self.subject_ids)}")
        print(f"  Samples: {len(self.hr_data)}")
    
    def normalize_data(self, data):
        """Normalize to zero mean, unit variance per sample"""
        # data: (N, C, T)
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = data.std(axis=(1, 2), keepdims=True) + 1e-8
        return (data - mean) / std
    
    def __len__(self):
        return len(self.hr_data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.hr_data[idx]).float(), self.labels[idx]


# ===== K-Fold Splits =====
def create_kfold_splits(n_subjects=15, n_folds=5, seed=42):
    """Create subject-based k-fold splits"""
    np.random.seed(seed)
    subjects = np.arange(1, n_subjects + 1)
    np.random.shuffle(subjects)
    
    fold_splits = []
    fold_size = n_subjects // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_subjects
        
        val_subjects = subjects[val_start:val_end]
        train_subjects = np.concatenate([subjects[:val_start], subjects[val_end:]])
        
        fold_splits.append({
            'fold': fold + 1,
            'train_subjects': train_subjects.tolist(),
            'val_subjects': val_subjects.tolist()
        })
    
    return fold_splits


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_x, _ in tqdm(dataloader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_size = batch_x.size(0)  # FIX: Get batch size from data
        
        # ADD: Additional per-batch channel normalization
        # Normalize each channel in the batch independently
        mean_ch = batch_x.mean(dim=(0, 2), keepdim=True)  # (1, C, 1)
        std_ch = batch_x.std(dim=(0, 2), keepdim=True) + 1e-8
        batch_x = (batch_x - mean_ch) / std_ch
        
        optimizer.zero_grad()
        
        # Forward
        loss, pred, mask = model(batch_x, mask_ratio=0.75)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_correlations = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, _ in tqdm(dataloader, desc="Validation", leave=False):
            batch_x = batch_x.to(device)
            batch_size = batch_x.size(0)
            
            # Forward
            loss, pred, mask = model(batch_x, mask_ratio=0.75)
            
            # Patchify the input to get target
            target = model.patchify(batch_x)  # [B, num_patches, C*patch_size]
            
            # Compute per-sample correlation on masked patches
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for i in range(batch_size):
                # Get masked patches only (mask=1 means masked)
                masked_patches = mask_np[i] == 1
                
                if masked_patches.sum() > 0:
                    pred_masked = pred_np[i][masked_patches].flatten()
                    target_masked = target_np[i][masked_patches].flatten()
                    
                    # Compute correlation
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
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_corr = np.mean(all_correlations) if all_correlations else 0.0
    
    return avg_loss, avg_corr


# ===== Main Training =====
def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def train_fold(fold_info, args, save_dir):
    fold_num = fold_info['fold']
    train_subjects = fold_info['train_subjects']
    val_subjects = fold_info['val_subjects']
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}/{args.n_folds}")
    print(f"{'='*60}")
    print(f"Train subjects: {train_subjects}")
    print(f"Val subjects: {val_subjects}")
    
    # Datasets
    train_dataset = SEED4RawHRDataset(args.data_path, subject_ids=train_subjects)
    val_dataset = SEED4RawHRDataset(args.data_path, subject_ids=val_subjects)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MAEforEEG(
        in_chans=31,
        time_len=1000,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads
    ).to(device)
    
    print(f"\n📊 Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_corr = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        
        # Validate
        val_loss, val_corr = validate_epoch(model, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        
        # History
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_corr'].append(float(val_corr))
        
        # Print
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f} | Val Corr: {val_corr:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_corr = val_corr
            
            fold_dir = save_dir / f'fold_{fold_num}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_corr': val_corr,
            }, fold_dir / 'best_model.pt')
    
    # Save history after training loop completes
    fold_dir = save_dir / f'fold_{fold_num}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'fold': fold_num,
        'best_val_loss': float(best_val_loss),
        'best_val_corr': float(best_val_corr),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_path', type=str, default='/home/ab_students/EEG-MTP/DATA/seed4/raw_data.npz')
    parser.add_argument('--n_folds', type=int, default=5)
    
    # Model
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--decoder_embed_dim', type=int, default=384)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--decoder_num_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # Output
    parser.add_argument('--save_dir', type=str, default='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_raw')
    
    args = parser.parse_args()
    
    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create k-fold splits
    print("🔄 Creating k-fold splits...")
    fold_splits = create_kfold_splits(n_subjects=15, n_folds=args.n_folds, seed=42)
    
    with open(save_dir / 'fold_splits.json', 'w') as f:
        json.dump(fold_splits, f, indent=2)
    
    # Train each fold
    fold_results = []
    
    for fold_info in fold_splits:
        result = train_fold(fold_info, args, save_dir)
        fold_results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("K-FOLD SUMMARY (RAW DATA)")
    print(f"{'='*60}")
    
    avg_loss = np.mean([r['best_val_loss'] for r in fold_results])
    avg_corr = np.mean([r['best_val_corr'] for r in fold_results])
    std_loss = np.std([r['best_val_loss'] for r in fold_results])
    std_corr = np.std([r['best_val_corr'] for r in fold_results])
    
    for result in fold_results:
        print(f"Fold {result['fold']}: Val Loss = {result['best_val_loss']:.6f}, Val Corr = {result['best_val_corr']:.4f}")
    
    print(f"\nAverage: Val Loss = {avg_loss:.6f} ± {std_loss:.6f}, Val Corr = {avg_corr:.4f} ± {std_corr:.4f}")
    
    # Save summary
    summary = {
        'fold_results': [
            {
                'fold': int(r['fold']),
                'best_val_loss': float(r['best_val_loss']),
                'best_val_corr': float(r['best_val_corr']),
                'train_samples': int(r['train_samples']),
                'val_samples': int(r['val_samples'])
            }
            for r in fold_results
        ],
        'average_val_loss': float(avg_loss),
        'average_val_corr': float(avg_corr),
        'std_val_loss': float(std_loss),
        'std_val_corr': float(std_corr)
    }
    
    with open(save_dir / 'kfold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ K-fold training complete! Results saved to {save_dir}")

    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
