"""
PrC-1 Reconstruction Script
----------------------------
Inverts PrC-1 transformations on super-resolution model output
to recover uV-scale EEG signals.

Pipeline assumed: Raw EEG -> PrC-1 -> SR Model -> This script

Usage:
    Configure the settings below and run:
        python reconstruct_prc1.py

Inputs:
    SR_OUTPUT_PATH:  .npy file from the SR model, shape (N, C_hr, T)
    PRC1_DIR:        PrC-1 output directory containing:
                       - X_prc1_norm_stats.npy   (N, 2) with [mu, sigma] per window
                       - prc1_meta.json          config metadata

Output:
    Reconstructed signal in uV, saved as .npy
"""

import json
import os
import numpy as np

# ========================== CONFIG ==========================

SR_OUTPUT_PATH = ""                # Path to SR model .npy output
PRC1_DIR = ""                      # PrC-1 output directory (contains stats + meta)
OUTPUT_PATH = ""                   # Output .npy path (leave empty for auto: <sr_dir>/reconstructed.npy)

INVERT_SOFT_CLIP = True            # Invert soft clipping (if applied during PrC-1)
SAVE_CONTINUOUS = False            # Also save concatenated continuous signal

# ============================================================


def load_prc1_artifacts(prc1_dir):
    """Load normalization stats and metadata from PrC-1 output directory."""
    stats = np.load(os.path.join(prc1_dir, "X_prc1_norm_stats.npy"))
    with open(os.path.join(prc1_dir, "prc1_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return stats, meta


def invert_soft_clip(x, clip_val):
    """Approximate inverse of soft clipping: x = clip_val * arctanh(x / clip_val)."""
    limit = clip_val * (1 - 1e-7)
    x_clamped = np.clip(x, -limit, limit)
    return clip_val * np.arctanh(x_clamped / clip_val)


def invert_normalization(x, mu, sigma):
    """Invert per-window z-score: w_original = w_norm * sigma + mu."""
    return x * sigma + mu


def reconstruct(sr_output, stats, meta, skip_soft_clip=False):
    """Invert PrC-1 transforms on SR output."""
    n_windows = sr_output.shape[0]
    if n_windows != len(stats):
        raise ValueError(
            f"Window count mismatch: SR output has {n_windows}, PrC-1 stats has {len(stats)}"
        )

    out = sr_output.copy()

    if meta.get("soft_clip_applied", False) and not skip_soft_clip:
        clip_val = meta.get("soft_clip_val", 5.0)
        print(f"Inverting soft clip (a={clip_val})")
        out = invert_soft_clip(out, clip_val)
    elif meta.get("soft_clip_applied", False) and skip_soft_clip:
        print("Soft clip inversion skipped (INVERT_SOFT_CLIP = False)")

    for i in range(n_windows):
        mu, sigma = stats[i]
        out[i] = invert_normalization(out[i], mu, sigma)

    return out


def windows_to_continuous(windows):
    """Concatenate non-overlapping windows: (N, C, T) -> (C, N*T)."""
    return np.concatenate([windows[i] for i in range(windows.shape[0])], axis=1)


def main():
    if not SR_OUTPUT_PATH or not PRC1_DIR:
        raise ValueError("Set SR_OUTPUT_PATH and PRC1_DIR in the config section before running.")

    sr = np.load(SR_OUTPUT_PATH)
    stats, meta = load_prc1_artifacts(PRC1_DIR)

    print(f"SR output shape: {sr.shape}")
    print(f"PrC-1 stats: {stats.shape}, soft_clip={meta.get('soft_clip_applied')}")

    recon = reconstruct(sr, stats, meta, skip_soft_clip=not INVERT_SOFT_CLIP)

    out_path = OUTPUT_PATH or os.path.join(os.path.dirname(SR_OUTPUT_PATH), "reconstructed.npy")
    np.save(out_path, recon)
    print(f"Saved reconstructed signal: {recon.shape} -> {out_path}")

    if SAVE_CONTINUOUS:
        cont = windows_to_continuous(recon)
        cont_path = out_path.replace(".npy", "_continuous.npy")
        np.save(cont_path, cont)
        print(f"Saved continuous signal: {cont.shape} -> {cont_path}")

    print("\nSanity check (first window):")
    print(f"  Range: [{recon[0].min():.4f}, {recon[0].max():.4f}] uV")
    print(f"  Mean:  {recon[0].mean():.6f} uV")
    print(f"  Std:   {recon[0].std():.6f} uV")


if __name__ == "__main__":
    main()
