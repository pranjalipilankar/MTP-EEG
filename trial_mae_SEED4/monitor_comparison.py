#!/usr/bin/env python3
"""
Monitor and compare preprocessed vs raw data training
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_history(fold_dir):
    """Load training history from a fold - handle incomplete files"""
    history_path = fold_dir / 'history.json'
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                # Handle two formats: list of dicts or dict of lists
                if isinstance(data, list):
                    return {
                        'train_loss': [d['train_loss'] for d in data],
                        'val_loss': [d['val_loss'] for d in data],
                        'val_corr': [d['val_cor'] for d in data]
                    }
                else:
                    return data
        except json.JSONDecodeError as e:
            # File is incomplete (training in progress)
            print(f"    ⚠️  Warning: Incomplete history file (training in progress)")
            return None
        except Exception as e:
            print(f"    ❌ Error reading history: {e}")
            return None
    return None


def check_best_model(fold_dir):
    """Check if best model exists and get its info"""
    best_model_path = fold_dir / 'best_model.pth'
    if best_model_path.exists():
        return True
    return False


def count_checkpoints(fold_dir):
    """Count number of checkpoint files"""
    checkpoints = list(fold_dir.glob('checkpoint_epoch_*.pth'))
    return len(checkpoints)


def monitor_progress(results_dir_preprocessed, results_dir_raw):
    """Monitor both training runs"""
    
    print("="*80)
    print("MAE K-FOLD TRAINING COMPARISON: PREPROCESSED vs RAW")
    print("="*80)
    
    # Check preprocessed
    print("\n📊 PREPROCESSED DATA:")
    preproc_dir = Path(results_dir_preprocessed)
    if preproc_dir.exists():
        for fold_num in range(1, 6):
            fold_dir = preproc_dir / f'fold_{fold_num}'
            if fold_dir.exists():
                history = load_history(fold_dir)
                has_best = check_best_model(fold_dir)
                n_checkpoints = count_checkpoints(fold_dir)
                
                if history and len(history.get('train_loss', [])) > 0:
                    epochs_done = len(history['train_loss'])
                    latest_val_loss = history['val_loss'][-1] if history['val_loss'] else None
                    latest_val_corr = history['val_corr'][-1] if history['val_corr'] else None
                    best_val_loss = min(history['val_loss']) if history['val_loss'] else None
                    best_val_corr = max(history['val_corr']) if history['val_corr'] else None
                    
                    print(f"  Fold {fold_num}: Epoch {epochs_done}/100 {'✅' if epochs_done >= 100 else '🔄'}")
                    print(f"    Latest - Loss: {latest_val_loss:.6f}, Corr: {latest_val_corr:.6f}")
                    print(f"    Best   - Loss: {best_val_loss:.6f}, Corr: {best_val_corr:.6f}")
                    print(f"    Files  - Best model: {'✓' if has_best else '✗'}, Checkpoints: {n_checkpoints}")
                else:
                    print(f"  Fold {fold_num}: 🔄 In progress... (checkpoints: {n_checkpoints})")
            else:
                print(f"  Fold {fold_num}: ⏳ Not started")
    else:
        print("  ❌ Directory not found")
    
    # Check raw
    print("\n📊 RAW DATA:")
    raw_dir = Path(results_dir_raw)
    if raw_dir.exists():
        for fold_num in range(1, 6):
            fold_dir = raw_dir / f'fold_{fold_num}'
            if fold_dir.exists():
                history = load_history(fold_dir)
                has_best = check_best_model(fold_dir)
                n_checkpoints = count_checkpoints(fold_dir)
                
                if history and len(history.get('train_loss', [])) > 0:
                    epochs_done = len(history['train_loss'])
                    latest_val_loss = history['val_loss'][-1] if history['val_loss'] else None
                    latest_val_corr = history['val_corr'][-1] if history['val_corr'] else None
                    best_val_loss = min(history['val_loss']) if history['val_loss'] else None
                    best_val_corr = max(history['val_corr']) if history['val_corr'] else None
                    
                    print(f"  Fold {fold_num}: Epoch {epochs_done}/100 {'✅' if epochs_done >= 100 else '🔄'}")
                    print(f"    Latest - Loss: {latest_val_loss:.6f}, Corr: {latest_val_corr:.6f}")
                    print(f"    Best   - Loss: {best_val_loss:.6f}, Corr: {best_val_corr:.6f}")
                    print(f"    Files  - Best model: {'✓' if has_best else '✗'}, Checkpoints: {n_checkpoints}")
                else:
                    print(f"  Fold {fold_num}: 🔄 In progress... (checkpoints: {n_checkpoints})")
            else:
                print(f"  Fold {fold_num}: ⏳ Not started")
    else:
        print("  ❌ Directory not found")
    
    # Load summaries if available
    preproc_summary_path = preproc_dir / 'kfold_summary.json' if preproc_dir.exists() else None
    raw_summary_path = raw_dir / 'kfold_summary.json' if raw_dir.exists() else None
    
    if preproc_summary_path and preproc_summary_path.exists() and \
       raw_summary_path and raw_summary_path.exists():
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        
        try:
            with open(preproc_summary_path, 'r') as f:
                preproc_summary = json.load(f)
            
            with open(raw_summary_path, 'r') as f:
                raw_summary = json.load(f)
            
            print(f"\nPreprocessed:")
            print(f"  Avg Val Loss: {preproc_summary['average_val_loss']:.6f} ± {preproc_summary['std_val_loss']:.6f}")
            print(f"  Avg Val Corr: {preproc_summary['average_val_corr']:.6f} ± {preproc_summary['std_val_corr']:.6f}")
            
            print(f"\nRaw:")
            print(f"  Avg Val Loss: {raw_summary['average_val_loss']:.6f} ± {raw_summary['std_val_loss']:.6f}")
            print(f"  Avg Val Corr: {raw_summary['average_val_corr']:.6f} ± {raw_summary['std_val_corr']:.6f}")
            
            # Comparison
            loss_diff = raw_summary['average_val_loss'] - preproc_summary['average_val_loss']
            corr_diff = raw_summary['average_val_corr'] - preproc_summary['average_val_corr']
            
            print(f"\n{'='*80}")
            print("WINNER:")
            print(f"{'='*80}")
            
            if loss_diff < 0:
                print(f"✅ RAW data: {abs(loss_diff):.6f} LOWER val loss")
            else:
                print(f"✅ PREPROCESSED data: {abs(loss_diff):.6f} LOWER val loss")
            
            if corr_diff > 0:
                print(f"✅ RAW data: {abs(corr_diff):.6f} HIGHER val correlation")
            else:
                print(f"✅ PREPROCESSED data: {abs(corr_diff):.6f} HIGHER val correlation")
        
        except Exception as e:
            print(f"\n⚠️  Could not load summaries: {e}")
    
    elif raw_summary_path and raw_summary_path.exists():
        print("\n" + "="*80)
        print("RAW DATA SUMMARY (Preprocessed training not complete)")
        print("="*80)
        
        try:
            with open(raw_summary_path, 'r') as f:
                raw_summary = json.load(f)
            
            print(f"\nRaw:")
            print(f"  Avg Val Loss: {raw_summary['average_val_loss']:.6f} ± {raw_summary['std_val_loss']:.6f}")
            print(f"  Avg Val Corr: {raw_summary['average_val_corr']:.6f} ± {raw_summary['std_val_corr']:.6f}")
        except Exception as e:
            print(f"\n⚠️  Could not load raw summary: {e}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    results_dir_preprocessed = '/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold'
    results_dir_raw = '/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_raw'
    
    monitor_progress(results_dir_preprocessed, results_dir_raw)
