#!/usr/bin/env python3
"""
Helper script to select the best checkpoint from k-fold training
"""
import json
from pathlib import Path
import shutil

def select_best_checkpoint(kfold_results_dir, output_path='pretrained_mae_seed4.pth'):
    """
    Select best checkpoint from k-fold results
    
    Args:
        kfold_results_dir: Path to k-fold results directory
        output_path: Where to copy the best checkpoint
    """
    results_dir = Path(kfold_results_dir)
    
    # Load summary
    summary_path = results_dir / 'kfold_summary.json'
    if not summary_path.exists():
        print(f"❌ Summary not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Find best fold
    best_fold = max(summary['fold_results'], key=lambda x: x['best_val_corr'])
    
    print("="*80)
    print("K-Fold Training Summary")
    print("="*80)
    print(f"\n📊 All Folds:")
    for result in summary['fold_results']:
        marker = "⭐" if result['fold'] == best_fold['fold'] else "  "
        print(f"{marker} Fold {result['fold']}: Cor={result['best_val_corr']:.6f} (Epoch {result.get('best_epoch', 'N/A')})")
    
    print(f"\n✅ Best Fold: {best_fold['fold']}")
    print(f"   Val Correlation: {best_fold['best_val_corr']:.6f}")
    print(f"   Epoch: {best_fold.get('best_epoch', 'N/A')}")
    
    # Copy checkpoint
    checkpoint_path = results_dir / f"fold_{best_fold['fold']}" / 'best_model.pt'
    
    if not checkpoint_path.exists():
        # Try .pth extension
        checkpoint_path = results_dir / f"fold_{best_fold['fold']}" / 'best_model.pth'
    
    if checkpoint_path.exists():
        output_path = Path(output_path)
        shutil.copy(checkpoint_path, output_path)
        print(f"\n💾 Checkpoint copied to: {output_path}")
        return str(output_path)
    else:
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Select best MAE checkpoint from k-fold results')
    parser.add_argument('--results_dir', type=str, 
                       default='/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold',
                       help='Path to k-fold results directory')
    parser.add_argument('--output', type=str,
                       default='../New_SEED4/pretrained_mae_31ch.pth',
                       help='Output path for best checkpoint')
    args = parser.parse_args()
    
    select_best_checkpoint(args.results_dir, args.output)
