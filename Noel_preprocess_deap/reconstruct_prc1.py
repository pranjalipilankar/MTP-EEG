"""
PrC-1 Reconstruction Script
----------------------------
Inverts PrC-1 transformations on super-resolution model output
to recover µV-scale EEG signals.

Pipeline assumed: Raw EEG → PrC-1 → SR Model → This script

Usage:
    Configure the settings below and run:
        python reconstruct_prc1.py

Inputs:
    SR_OUTPUT_PATH:  .npy file from the SR model, shape (N, C_hr, T)
                     C_hr >= C_lr (62). Extra channels are SR-generated.
    PRC1_DIR:        PrC-1 output directory containing:
                       - X_prc1_norm_stats.npy   (N, 2) with [µ, σ] per window
                       - prc1_meta.json          config metadata

Output:
    Reconstructed signal in µV, saved as .npy
"""

import json
import os
import numpy as np

# ========================== CONFIG ==========================

SR_OUTPUT_PATH = ""                # Path to SR model .npy output
PRC1_DIR       = ""                # PrC-1 output directory (contains stats + meta)
OUTPUT_PATH    = ""                # Output .npy path (leave empty for auto: <sr_dir>/reconstructed.npy)

INVERT_SOFT_CLIP = True            # Invert soft clipping (if applied during PrC-1)
SAVE_CONTINUOUS  = False           # Also save concatenated continuous signal

# ============================================================


def load_prc1_artifacts(prc1_dir):
    """Load normalization stats and metadata from PrC-1 output directory."""
    stats = np.load(os.path.join(prc1_dir, "X_prc1_norm_stats.npy"))
    with open(os.path.join(prc1_dir, "prc1_meta.json")) as f:
        meta = json.load(f)
    return stats, meta


def invert_soft_clip(x, clip_val):
    """Approximate inverse of soft clipping: x = clip_val * arctanh(x / clip_val).
    Clamps input to avoid arctanh divergence at ±clip_val."""
    limit = clip_val * (1 - 1e-7)
    x_clamped = np.clip(x, -limit, limit)
    return clip_val * np.arctanh(x_clamped / clip_val)


def invert_normalization(x, mu, sigma):
    """Invert per-window z-score: w_original = w_norm * σ + µ."""
    return x * sigma + mu


def reconstruct(sr_output, stats, meta, target_channels=None, skip_soft_clip=False):
    """
    Invert PrC-1 transformations on SR model output.

    Parameters
    ----------
    sr_output : ndarray, shape (N, C_hr, T)
        Super-resolution model output. May have more channels than PrC-1 input.
    stats : ndarray, shape (N, 2)
        Per-window [µ, σ] from PrC-1.
    meta : dict
        PrC-1 metadata (soft_clip_applied, soft_clip_val, etc.)
    target_channels : list of int, optional
        Which channels in sr_output correspond to the original 62.
        If None, assumes the first meta['n_channels'] channels are the originals.
    skip_soft_clip : bool, optional
        If True, skip soft-clip inversion even if it was applied during PrC-1.

    Returns
    -------
    reconstructed : ndarray, same shape as sr_output, in µV scale
    """
    n_windows = sr_output.shape[0]

    if n_windows != len(stats):
        raise ValueError(
            f"Window count mismatch: SR output has {n_windows}, "
            f"PrC-1 stats has {len(stats)}")

    out = sr_output.copy()

    # Step 1: Invert soft clipping (if it was applied during PrC-1)
    if meta.get("soft_clip_applied", False) and not skip_soft_clip:
        clip_val = meta["soft_clip_val"]
        print(f"Inverting soft clip (a={clip_val})")
        out = invert_soft_clip(out, clip_val)
    elif meta.get("soft_clip_applied", False) and skip_soft_clip:
        print("Soft clip inversion SKIPPED (INVERT_SOFT_CLIP = False)")

    # Step 2: Invert z-score normalization per window
    # The SR model may output more channels than the original 62.
    # The same µ/σ (computed globally across the original window) is used
    # because the SR-generated channels live in the same normalized space.
    for i in range(n_windows):
        mu, sigma = stats[i]
        out[i] = invert_normalization(out[i], mu, sigma)

    return out


def windows_to_continuous(windows):
    """Concatenate non-overlapping windows back into a continuous signal.
    Input: (N, C, T) → Output: (C, N*T)"""
    return np.concatenate([windows[i] for i in range(windows.shape[0])], axis=1)


def main():
    if not SR_OUTPUT_PATH or not PRC1_DIR:
        raise ValueError("Set SR_OUTPUT_PATH and PRC1_DIR in the config section before running.")

    # Load inputs
    sr = np.load(SR_OUTPUT_PATH)
    stats, meta = load_prc1_artifacts(PRC1_DIR)

    print(f"SR output shape: {sr.shape}")
    print(f"PrC-1 stats: {stats.shape}, soft_clip={meta.get('soft_clip_applied')}")

    # Reconstruct
    recon = reconstruct(sr, stats, meta, skip_soft_clip=not INVERT_SOFT_CLIP)

    # Save
    out_path = OUTPUT_PATH or os.path.join(os.path.dirname(SR_OUTPUT_PATH), "reconstructed.npy")
    np.save(out_path, recon)
    print(f"Saved reconstructed signal: {recon.shape} → {out_path}")

    if SAVE_CONTINUOUS:
        cont = windows_to_continuous(recon)
        cont_path = out_path.replace(".npy", "_continuous.npy")
        np.save(cont_path, cont)
        print(f"Saved continuous signal: {cont.shape} → {cont_path}")

    # Quick sanity check
    print(f"\nSanity check (first window):")
    print(f"  Range: [{recon[0].min():.2f}, {recon[0].max():.2f}] µV")
    print(f"  Mean:  {recon[0].mean():.4f} µV")
    print(f"  Std:   {recon[0].std():.4f} µV")


if __name__ == "__main__":
    main()
