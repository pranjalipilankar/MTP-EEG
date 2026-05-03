#!/usr/bin/env python3
"""
Analyse training_history.npy from SEED-IV STAD + MFE training.
Usage: python analyse_training.py --path /home/ab_students/EEG-MTP/new_SEED4_mfe/training_history.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path

def load_history(path):
    data = np.load(path, allow_pickle=True)
    # Could be an array of dicts or a single dict
    if data.ndim == 0:
        data = data.item()
        if isinstance(data, list):
            return data
        return [data]
    return list(data)

def extract_metric(history, key):
    return [h[key] for h in history if key in h]

def print_summary(history):
    epochs      = extract_metric(history, 'epoch')
    train_loss  = extract_metric(history, 'train_loss')
    val_loss    = extract_metric(history, 'val_loss')
    val_pcc     = extract_metric(history, 'val_pcc')
    val_nmse    = extract_metric(history, 'val_nmse')
    val_snr     = extract_metric(history, 'val_snr')

    best_epoch_idx = int(np.argmin(val_loss))

    print("\n" + "="*60)
    print("  TRAINING HISTORY SUMMARY")
    print("="*60)
    print(f"  Total epochs logged   : {len(history)}")
    print(f"  Epoch range           : {epochs[0]} → {epochs[-1]}")
    print()
    print(f"  --- Final epoch ({epochs[-1]}) ---")
    print(f"  Train Loss            : {train_loss[-1]:.6f}")
    print(f"  Val Loss              : {val_loss[-1]:.6f}")
    if val_pcc:  print(f"  Val PCC               : {val_pcc[-1]:.4f}")
    if val_nmse: print(f"  Val NMSE              : {val_nmse[-1]:.4f}")
    if val_snr:  print(f"  Val SNR               : {val_snr[-1]:.2f} dB")
    print()
    print(f"  --- Best epoch (lowest val loss) ---")
    print(f"  Epoch                 : {epochs[best_epoch_idx]}")
    print(f"  Val Loss              : {val_loss[best_epoch_idx]:.6f}")
    if val_pcc:  print(f"  Val PCC               : {val_pcc[best_epoch_idx]:.4f}")
    if val_nmse: print(f"  Val NMSE              : {val_nmse[best_epoch_idx]:.4f}")
    if val_snr:  print(f"  Val SNR               : {val_snr[best_epoch_idx]:.2f} dB")
    print()
    print(f"  --- Val Loss Stats ---")
    print(f"  Min                   : {min(val_loss):.6f}")
    print(f"  Max                   : {max(val_loss):.6f}")
    print(f"  Mean                  : {np.mean(val_loss):.6f}")
    print(f"  Std                   : {np.std(val_loss):.6f}")
    print("="*60)

    # Overfitting check
    gap = np.array(val_loss) - np.array(train_loss)
    last10_gap = gap[-10:] if len(gap) >= 10 else gap
    print(f"\n  Overfitting check (val - train loss, last 10 epochs):")
    print(f"    Mean gap = {np.mean(last10_gap):.4f} | "
          f"Positive = overfitting, Negative = underfitting")
    print("="*60 + "\n")

def plot_history(history, save_path=None):
    epochs   = extract_metric(history, 'epoch')
    train_l  = extract_metric(history, 'train_loss')
    val_l    = extract_metric(history, 'val_loss')
    val_pcc  = extract_metric(history, 'val_pcc')
    val_nmse = extract_metric(history, 'val_nmse')
    val_snr  = extract_metric(history, 'val_snr')

    best_idx = int(np.argmin(val_l))

    n_plots = 1 + bool(val_pcc) + bool(val_nmse) + bool(val_snr)
    fig = plt.figure(figsize=(7 * min(n_plots, 2), 5 * ((n_plots + 1) // 2)))
    gs  = gridspec.GridSpec((n_plots + 1) // 2, min(n_plots, 2), figure=fig)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_plots)]

    # MFE warmup shading (epochs 20-50)
    def shade_warmup(ax):
        ep = np.array(epochs)
        if ep[-1] > 20:
            ax.axvspan(20, min(50, ep[-1]), alpha=0.08, color='orange', label='MFE warmup (ep 20–50)')

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, train_l, label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_l,   label='Val Loss',   linewidth=1.5)
    ax.axvline(epochs[best_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Best epoch {epochs[best_idx]}')
    shade_warmup(ax)
    ax.set_title('Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plot_i = 1
    if val_pcc:
        ax = axes[plot_i]; plot_i += 1
        ax.plot(epochs, val_pcc, color='green', linewidth=1.5)
        ax.axvline(epochs[best_idx], color='red', linestyle='--', alpha=0.7)
        shade_warmup(ax)
        ax.set_title('Val PCC (↑ better)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('PCC')
        ax.grid(True, alpha=0.3)

    if val_nmse:
        ax = axes[plot_i]; plot_i += 1
        ax.plot(epochs, val_nmse, color='purple', linewidth=1.5)
        ax.axvline(epochs[best_idx], color='red', linestyle='--', alpha=0.7)
        shade_warmup(ax)
        ax.set_title('Val NMSE (↓ better)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('NMSE')
        ax.grid(True, alpha=0.3)

    if val_snr:
        ax = axes[plot_i]; plot_i += 1
        ax.plot(epochs, val_snr, color='darkorange', linewidth=1.5)
        ax.axvline(epochs[best_idx], color='red', linestyle='--', alpha=0.7)
        shade_warmup(ax)
        ax.set_title('Val SNR dB (↑ better)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('SNR (dB)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('SEED-IV STAD + MFE — Training History', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default='/home/ab_students/EEG-MTP/new_SEED4_mfe/training_history.npy')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Optional path to save the figure, e.g. training_plot.png')
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nLoading: {path}")
    history = load_history(path)
    print(f"Loaded {len(history)} epoch records.")
    print(f"Keys in each record: {list(history[0].keys())}")

    print_summary(history)
    plot_history(history, save_path=args.save_plot)

if __name__ == '__main__':
    main()