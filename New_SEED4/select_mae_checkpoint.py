#!/usr/bin/env python3
"""
Select best MAE checkpoint from k-fold results based on training logs
"""
import shutil
from pathlib import Path

# K-fold results from your training
FOLD_RESULTS = {
    1: {'val_cor': 0.856729, 'epoch': 73},
    2: {'val_cor': 0.797216, 'epoch': 64},
    3: {'val_cor': 0.789385, 'epoch': 56},
    4: {'val_cor': 0.787286, 'epoch': 66},
    5: {'val_cor': 0.821233, 'epoch': 64},
}

def select_best_fold(results_dir='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold'):
    """Select and copy the best fold checkpoint"""
    
    results_dir = Path(results_dir)
    
    print("="*80)
    print("K-Fold MAE Training Results")
    print("="*80)
    print("\n📊 All Folds:")
    
    best_fold = None
    best_cor = -1
    
    for fold_num, metrics in FOLD_RESULTS.items():
        marker = ""
        if metrics['val_cor'] > best_cor:
            best_cor = metrics['val_cor']
            best_fold = fold_num
            marker = "⭐"
        
        print(f"{marker:2} Fold {fold_num}: Cor={metrics['val_cor']:.6f} (Epoch {metrics['epoch']})")
    
    print(f"\n✅ Best Fold: {best_fold}")
    print(f"   Val Correlation: {best_cor:.6f}")
    print(f"   Epoch: {FOLD_RESULTS[best_fold]['epoch']}")
    
    # Copy checkpoint
    checkpoint_path = results_dir / f'fold_{best_fold}' / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Output paths
    output_paths = {
        'stad': Path('../New_SEED4/pretrained_mae_31ch_best.pth'),
        'local': Path('pretrained_mae_31ch_fold{best_fold}.pth')
    }
    
    print(f"\n💾 Copying checkpoint...")
    for name, output_path in output_paths.items():
        try:
            shutil.copy(checkpoint_path, output_path)
            print(f"   ✓ {name}: {output_path}")
        except Exception as e:
            print(f"   ✗ {name}: Failed ({e})")
    
    return best_fold, best_cor


def copy_all_folds(results_dir='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold',
                   output_dir='./stad_checkpoints'):
    """Copy all fold checkpoints for ensemble or selection"""
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📦 Copying all fold checkpoints to: {output_dir}")
    
    for fold_num in FOLD_RESULTS.keys():
        src = results_dir / f'fold_{fold_num}' / 'best_model.pth'
        dst = output_dir / f'mae_fold{fold_num}_cor{FOLD_RESULTS[fold_num]["val_cor"]:.4f}.pth'
        
        if src.exists():
            shutil.copy(src, dst)
            print(f"   ✓ Fold {fold_num}: {dst.name}")
        else:
            print(f"   ✗ Fold {fold_num}: Not found at {src}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Select best MAE checkpoint from k-fold results')
    parser.add_argument('--results_dir', type=str,
                       default='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold',
                       help='K-fold results directory')
    parser.add_argument('--copy_all', action='store_true',
                       help='Copy all fold checkpoints')
    args = parser.parse_args()
    
    if args.copy_all:
        copy_all_folds(args.results_dir)
    else:
        select_best_fold(args.results_dir)
    
    print("\n" + "="*80 + "\n")
