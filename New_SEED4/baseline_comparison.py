#!/usr/bin/env python3
"""
Linear Interpolation Baseline Comparison for STAD Model

Implements a simple linear interpolation baseline (16 → 62 channels)
and compares against STAD model to diagnose poor performance.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import torch
from pathlib import Path
from tqdm import tqdm
import json


def get_seed4_channel_indices(target_channels):
    """Get fixed SEED-IV channel indices."""
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


def linear_interpolation_baseline(lr_eeg, lr_indices, sr_indices):
    """
    Linear interpolation baseline from LR to SR space.
    
    Args:
        lr_eeg: (batch, 16, 1000) - low resolution input
        lr_indices: (16,) - indices of LR channels in 62-channel space
        sr_indices: (62,) - indices of SR channels in 62-channel space
        
    Returns:
        pred_sr: (batch, 62, 1000) - interpolated SR prediction
    """
    batch_size, _, time_steps = lr_eeg.shape
    pred_sr = np.zeros((batch_size, 62, time_steps), dtype=np.float32)
    
    for b in range(batch_size):
        for t in range(time_steps):
            # Get LR values at this time step
            lr_values = lr_eeg[b, :, t]  # (16,)
            
            # Create interpolation function: LR channel indices → channel values
            f = interp1d(lr_indices, lr_values, kind='linear', 
                        fill_value='extrapolate', assume_sorted=True)
            
            # Evaluate at all 62 channel positions
            pred_sr[b, :, t] = f(np.arange(62))
    
    return pred_sr


def compute_sr_metrics(pred_sr, target_sr, eps=1e-8):
    """Compute SR metrics: PCC, NMSE, SNR."""
    pred = pred_sr.reshape(pred_sr.shape[0], -1)
    target = target_sr.reshape(target_sr.shape[0], -1)
    
    # PCC: Pearson Correlation Coefficient
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    
    numerator = (pred_centered * target_centered).sum(axis=1)
    denominator = np.sqrt(
        (pred_centered**2).sum(axis=1) * (target_centered**2).sum(axis=1) + eps
    )
    pcc = np.mean(numerator / (denominator + eps))
    
    # NMSE
    mse = ((pred - target) ** 2).mean(axis=1)
    signal_power = (target ** 2).mean(axis=1)
    nmse = np.mean(mse / (signal_power + eps))
    
    # SNR (dB)
    snr = 10.0 * np.log10((signal_power + eps) / (mse + eps))
    snr_db = np.mean(snr)
    
    return {
        'pcc': pcc,
        'nmse': nmse,
        'snr': snr_db,
    }


def main():
    print("\n" + "="*80)
    print("BASELINE COMPARISON: Linear Interpolation vs STAD Model")
    print("="*80)
    
    # Load test data
    data_path = Path('/DATA/EEG-MTP/seed4/raw_data.npz')
    print(f"\nLoading test data from {data_path}...")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    lr_all = data['LR'].astype(np.float32)  # (7035, 16, 1000)
    sr_all = data['SR'].astype(np.float32)  # (7035, 62, 1000)
    subject_ids = data['subject_ids']  # (7035,)
    
    print(f"✓ Loaded LR shape: {lr_all.shape}")
    print(f"✓ Loaded SR shape: {sr_all.shape}")
    print(f"✓ Loaded subject IDs shape: {subject_ids.shape}")
    
    # Filter test subjects
    test_subjects = ['7', '10', '15']
    test_mask = np.isin(subject_ids, test_subjects)
    lr_test = lr_all[test_mask]
    sr_test = sr_all[test_mask]
    
    print(f"\nFiltered test data:")
    print(f"  Test subjects: {test_subjects}")
    print(f"  Test samples: {len(lr_test)} / {len(lr_all)}")
    print(f"  LR shape: {lr_test.shape}")
    print(f"  SR shape: {sr_test.shape}")
    
    # Get channel indices
    lr_indices = get_seed4_channel_indices(16)
    sr_indices = get_seed4_channel_indices(62)
    
    print(f"\n{'='*80}")
    print("BASELINE 1: LINEAR INTERPOLATION")
    print("="*80)
    
    print(f"Computing linear interpolation predictions...")
    pred_sr_interp = linear_interpolation_baseline(lr_test, lr_indices, sr_indices)
    
    print(f"  Prediction shape: {pred_sr_interp.shape}")
    print(f"  Prediction range: [{pred_sr_interp.min():.4f}, {pred_sr_interp.max():.4f}]")
    
    metrics_interp = compute_sr_metrics(pred_sr_interp, sr_test)
    
    print(f"\nLinear Interpolation Metrics:")
    print(f"  PCC: {metrics_interp['pcc']:+.6f}")
    print(f"  NMSE: {metrics_interp['nmse']:.6f}")
    print(f"  SNR: {metrics_interp['snr']:.2f} dB")
    
    # Load STAD predictions
    print(f"\n{'='*80}")
    print("BASELINE 2: STAD MODEL (from evaluation)")
    print("="*80)
    
    results_file = Path('stad_raw_evaluation/results_summary.npz')
    if not results_file.exists():
        raise FileNotFoundError(f"STAD results not found: {results_file}")
    
    print(f"Loading STAD results from {results_file}...")
    stad_results = np.load(results_file, allow_pickle=True)
    
    pred_sr_stad_batched = stad_results['pred_sr']  # (batches, 32, 62, 1000)
    sr_test_batched = stad_results['target_sr']      # (batches, 32, 62, 1000)
    
    # Reshape to single batch
    pred_sr_stad = pred_sr_stad_batched.reshape(-1, 62, 1000)
    sr_test_batched_flat = sr_test_batched.reshape(-1, 62, 1000)
    
    # Trim to same number of samples as linear interp
    n_samples = min(len(pred_sr_stad), len(pred_sr_interp))
    pred_sr_stad = pred_sr_stad[:n_samples]
    sr_test_batched_flat = sr_test_batched_flat[:n_samples]
    pred_sr_interp = pred_sr_interp[:n_samples]
    sr_test = sr_test[:n_samples]
    
    print(f"  STAD predictions shape: {pred_sr_stad.shape}")
    print(f"  Prediction range: [{pred_sr_stad.min():.4f}, {pred_sr_stad.max():.4f}]")
    
    metrics_stad = compute_sr_metrics(pred_sr_stad, sr_test)
    
    print(f"\nSTAD Model Metrics:")
    print(f"  PCC: {metrics_stad['pcc']:+.6f}")
    print(f"  NMSE: {metrics_stad['nmse']:.6f}")
    print(f"  SNR: {metrics_stad['snr']:.2f} dB")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print("="*80)
    
    pcc_diff = metrics_stad['pcc'] - metrics_interp['pcc']
    nmse_diff = metrics_stad['nmse'] - metrics_interp['nmse']
    snr_diff = metrics_stad['snr'] - metrics_interp['snr']
    
    print(f"\n{'Metric':<20} {'Linear Interp':<20} {'STAD':<20} {'Difference':<15}")
    print(f"{'-'*75}")
    print(f"{'PCC':<20} {metrics_interp['pcc']:+.6f}{'':<13} {metrics_stad['pcc']:+.6f}{'':<13} {pcc_diff:+.6f}")
    print(f"{'NMSE':<20} {metrics_interp['nmse']:+.6f}{'':<13} {metrics_stad['nmse']:+.6f}{'':<13} {nmse_diff:+.6f}")
    print(f"{'SNR (dB)':<20} {metrics_interp['snr']:+.6f}{'':<13} {metrics_stad['snr']:+.6f}{'':<13} {snr_diff:+.6f}")
    
    print(f"\n{'='*80}")
    print("DIAGNOSTIC FINDINGS")
    print("="*80)
    
    if abs(pcc_diff) < 0.01 and abs(nmse_diff) < 0.01:
        print(f"\n🚨 CRITICAL ISSUE FOUND:")
        print(f"   STAD model performs SIMILAR to random linear interpolation!")
        print(f"   • PCC difference: {pcc_diff:.6f} (negligible)")
        print(f"   • NMSE difference: {nmse_diff:.6f} (negligible)")
        print(f"\n   This suggests:")
        print(f"   1. Model is not learning meaningful representations")
        print(f"   2. Loss function may be misaligned with task objective")
        print(f"   3. Model capacity or training hyperparameters may be wrong")
    elif pcc_diff > 0:
        print(f"\n✓ STAD outperforms baseline:")
        print(f"   PCC improvement: {pcc_diff:+.6f}")
        print(f"   However, absolute STAD PCC is still very low ({metrics_stad['pcc']:.6f})")
    else:
        print(f"\n❌ STAD UNDERPERFORMS baseline linear interpolation!")
        print(f"   PCC degradation: {pcc_diff:+.6f}")
        print(f"   This is a major red flag")
    
    # Check training issues
    print(f"\n{'='*80}")
    print("TRAINING DIAGNOSIS (from history)")
    print("="*80)
    
    hist_path = Path('results_stad_raw/training_history.npy')
    if hist_path.exists():
        hist_raw = np.load(hist_path, allow_pickle=True)
        history_list = list(hist_raw)
        
        val_pcc = np.array([h['val_pcc'] for h in history_list])
        val_nmse = np.array([h['val_nmse'] for h in history_list])
        val_total_loss = np.array([h['val_total_loss'] for h in history_list])
        train_total_loss = np.array([h['train_total_loss'] for h in history_list])
        
        # Overfitting check
        gap = val_total_loss - train_total_loss
        gap_first = gap[0]
        gap_last = gap[-1]
        
        print(f"\nOverfitting indicators:")
        print(f"  Train-Val gap (epoch 1): {gap_first:.4f}")
        print(f"  Train-Val gap (last):    {gap_last:.4f}")
        
        if gap_last > gap_first:
            print(f"  ⚠️  Gap increased by {((gap_last - gap_first) / gap_first * 100):.1f}% → STRONG OVERFITTING")
        else:
            print(f"  ✓ Gap decreased → Good generalization")
        
        print(f"\nReconstruction metric trends:")
        print(f"  Max val PCC during training: {np.max(val_pcc):.6f}")
        print(f"  Final val PCC: {val_pcc[-1]:.6f}")
        print(f"  Improvement: {(val_pcc[-1] - val_pcc[0]):.6f}")
        
        if np.max(val_pcc) < 0.1:
            print(f"  ⚠️  PCC never improved significantly")
            print(f"      This indicates loss function doesn't correlate with reconstruction quality")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)
    
    print(f"""
1. CHECK LOSS FUNCTION ALIGNMENT:
   • Current metrics (PCC, NMSE) show near-zero performance
   • Yet training loss decreases (diffusion loss: 1.09 → 0.06)
   • Hypothesis: Loss function (diffusion + L2) may not optimize for reconstruction quality
   • Action: Visualize sample reconstructions to see if model is learning _something_

2. VERIFY MODEL ARCHITECTURE:
   • Check if MAE encoder is properly frozen/unfrozen
   • Verify STC (Spatio-Temporal Conditioner) is receiving correct conditioning
   • Check if MTD (Multi-scale Transformer Denoising) output is properly scaled

3. INVESTIGATE DATA ISSUES:
   • Data leakage: ✅ VERIFIED - test subjects NOT in training
   • Data normalization: Verify LR/SR consistent preprocessing
   • Data distribution: Check if test set differs from training

4. TRY ALTERNATIVE APPROACHES:
   • Replace diffusion loss with direct L2/L1 reconstruction loss
   • Simplify model (remove MTD, use MAE directly)
   • Increase training duration or adjust hyperparameters
   • Reduce model complexity to see if simpler model performs better

5. NEXT DEBUG STEPS:
   • Visualize 3-5 sample reconstructions (STAD vs linear interp)
   • Check gradients during training (vanishing/exploding)
   • Run inference on training set (should have better metrics)
   • Compare against simpler baseline (e.g., spline interpolation)
""")
    
    # Save comparison results
    output_file = Path('baseline_comparison_results.json')
    results_comparison = {
        'baseline': {
            'method': 'Linear Interpolation',
            'pcc': float(metrics_interp['pcc']),
            'nmse': float(metrics_interp['nmse']),
            'snr_db': float(metrics_interp['snr']),
        },
        'stad': {
            'method': 'STAD Model',
            'pcc': float(metrics_stad['pcc']),
            'nmse': float(metrics_stad['nmse']),
            'snr_db': float(metrics_stad['snr']),
        },
        'comparison': {
            'pcc_diff': float(pcc_diff),
            'nmse_diff': float(nmse_diff),
            'snr_diff': float(snr_diff),
        },
        'test_samples': int(n_samples),
        'test_subjects': test_subjects,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_comparison, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
