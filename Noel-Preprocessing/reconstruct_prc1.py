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
import shutil
import numpy as np
from scipy.io import savemat

# ========================== CONFIG ==========================
PRC1_DIR       = ""                # PrC-1 output directory (contains stats + meta)
SR_OUTPUT_PATH = "/home/ab_students/EEG-MTP/New_SEED4/test_outputs/sr_pred_test.npy"
STATS_OVERRIDE_PATH = "/home/ab_students/EEG-MTP/New_SEED4/test_outputs/test_norm_stats.npy"
META_OVERRIDE_PATH = "/home/ab_students/EEG-MTP/New_SEED4/test_outputs/test_prc1_meta.json"
OUTPUT_PATH = "/home/ab_students/EEG-MTP/New_SEED4/test_outputs/sr_pred_test_reconstructed.npy"

INVERT_SOFT_CLIP = True            # Invert soft clipping (if applied during PrC-1)
SAVE_CONTINUOUS  = False           # Also save concatenated continuous signal

# Optional export: write reconstructed TEST EEG as raw-data style folders/files.
EXPORT_EEG_RAW_DATA_FORMAT = True
RAW_EXPORT_ROOT = "/home/ab_students/EEG-MTP/New_SEED4/test_outputs/eeg_raw_data"  # e.g. /path/to/eeg_raw_data
REFERENCE_DATA_ROOT = "/DATA/EEG-MTP/seed4"                                         # Folder that contains Channel Order.xlsx and ReadMe.txt
TEST_METADATA_PATH = "/home/ab_students/EEG-MTP/New_SEED4/test_outputs/test_metadata.npz"  # Optional .npz/.npy/.json with subject/session metadata
TEST_INDICES_PATH = ""             # Optional .npz/.npy with test indices to subset windows

# ============================================================


def load_prc1_artifacts(prc1_dir, stats_override_path="", meta_override_path=""):
    """Load normalization stats and metadata from PrC-1 output directory."""
    if stats_override_path:
        stats = np.load(stats_override_path)
    else:
        stats = np.load(os.path.join(prc1_dir, "X_prc1_norm_stats.npy"))

    if meta_override_path:
        with open(meta_override_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    elif prc1_dir:
        with open(os.path.join(prc1_dir, "prc1_meta.json"), 'r', encoding='utf-8') as f:
            meta = json.load(f)
    else:
        meta = {
            "soft_clip_applied": False,
            "soft_clip_val": 5.0,
            "n_channels": int(stats.shape[1]) if stats.ndim > 1 else None,
        }
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


def _load_indices(indices_path):
    """Load index array from .npy or .npz (key: test_indices/indices)."""
    if not indices_path:
        return None

    if indices_path.endswith('.npy'):
        return np.load(indices_path)

    if indices_path.endswith('.npz'):
        z = np.load(indices_path, allow_pickle=True)
        for key in ('test_indices', 'indices'):
            if key in z:
                return z[key]
        raise KeyError(f"No test_indices/indices key found in {indices_path}")

    raise ValueError(f"Unsupported indices file type: {indices_path}")


def _load_metadata(meta_path, n_windows):
    """
    Load window-level metadata with optional keys:
      - subject_ids: shape (N,)
      - session_ids: shape (N,) values typically '1'/'2'/'3'
      - trial_ids:   shape (N,) optional trial grouping
    """
    if not meta_path:
        # Fallback: place everything under session 1, subject "reconstructed".
        return {
            'subject_ids': np.array(['reconstructed'] * n_windows),
            'session_ids': np.array(['1'] * n_windows),
            'trial_ids': None,
        }

    if meta_path.endswith('.npz'):
        z = np.load(meta_path, allow_pickle=True)
        subject_ids = z['subject_ids'] if 'subject_ids' in z else np.array(['reconstructed'] * n_windows)
        session_ids = z['session_ids'] if 'session_ids' in z else np.array(['1'] * n_windows)
        trial_ids = z['trial_ids'] if 'trial_ids' in z else None
    elif meta_path.endswith('.npy'):
        obj = np.load(meta_path, allow_pickle=True).item()
        subject_ids = np.asarray(obj.get('subject_ids', ['reconstructed'] * n_windows))
        session_ids = np.asarray(obj.get('session_ids', ['1'] * n_windows))
        trial_ids = obj.get('trial_ids', None)
        if trial_ids is not None:
            trial_ids = np.asarray(trial_ids)
    elif meta_path.endswith('.json'):
        with open(meta_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        subject_ids = np.asarray(obj.get('subject_ids', ['reconstructed'] * n_windows))
        session_ids = np.asarray(obj.get('session_ids', ['1'] * n_windows))
        trial_ids = obj.get('trial_ids', None)
        if trial_ids is not None:
            trial_ids = np.asarray(trial_ids)
    else:
        raise ValueError(f"Unsupported metadata file type: {meta_path}")

    if len(subject_ids) != n_windows or len(session_ids) != n_windows:
        raise ValueError(
            f"Metadata length mismatch with windows. windows={n_windows}, "
            f"subject_ids={len(subject_ids)}, session_ids={len(session_ids)}"
        )
    if trial_ids is not None and len(trial_ids) != n_windows:
        raise ValueError(f"trial_ids length mismatch: {len(trial_ids)} vs {n_windows}")

    return {
        'subject_ids': subject_ids.astype(str),
        'session_ids': session_ids.astype(str),
        'trial_ids': trial_ids,
    }


def _copy_reference_files(reference_data_root, raw_export_root):
    """Copy Channel Order.xlsx and ReadMe.txt next to eeg_raw_data folder if available."""
    if not reference_data_root:
        return

    for name in ("Channel Order.xlsx", "ReadMe.txt"):
        src = os.path.join(reference_data_root, name)
        dst = os.path.join(os.path.dirname(raw_export_root), name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {name} -> {dst}")
        else:
            print(f"Warning: reference file not found: {src}")


def export_reconstructed_as_raw_structure(recon, raw_export_root, metadata, test_indices=None):
    """
    Export reconstructed windows to:
      raw_export_root/1/*.mat
      raw_export_root/2/*.mat
      raw_export_root/3/*.mat

    Each .mat will contain trial variables:
      - If trial_ids are provided: cz_eeg<trial_id>
      - Otherwise: single variable cz_eeg1 containing concatenated windows.
    """
    if not raw_export_root:
        raise ValueError("Set RAW_EXPORT_ROOT when EXPORT_EEG_RAW_DATA_FORMAT is True")

    subject_ids = metadata['subject_ids']
    session_ids = metadata['session_ids']
    trial_ids = metadata['trial_ids']

    if test_indices is not None:
        test_indices = np.asarray(test_indices).astype(int)
        recon = recon[test_indices]
        subject_ids = subject_ids[test_indices]
        session_ids = session_ids[test_indices]
        if trial_ids is not None:
            trial_ids = np.asarray(trial_ids)[test_indices]
        print(f"Exporting only test subset: {len(test_indices)} windows")

    for session in ('1', '2', '3'):
        os.makedirs(os.path.join(raw_export_root, session), exist_ok=True)

    unique_subject_session = sorted(set(zip(subject_ids.tolist(), session_ids.tolist())))

    for subject, session in unique_subject_session:
        mask = (subject_ids == subject) & (session_ids == session)
        windows = recon[mask]
        mat_path = os.path.join(raw_export_root, session, f"{subject}.mat")

        if windows.size == 0:
            continue

        mat_dict = {}
        if trial_ids is not None:
            trials = np.asarray(trial_ids)[mask]
            trial_offset = 1 if np.min(trials) == 0 else 0
            for tid in sorted(np.unique(trials)):
                tmask = (trials == tid)
                trial_windows = windows[tmask]
                # Convert (N, C, T) to continuous (T_total, C) as common .mat style.
                cont = windows_to_continuous(trial_windows).T
                mat_dict[f"cz_eeg{int(tid) + trial_offset}"] = cont
        else:
            cont = windows_to_continuous(windows).T
            mat_dict["cz_eeg1"] = cont

        savemat(mat_path, mat_dict, do_compression=True)
        print(f"Saved {mat_path} ({len(windows)} windows)")


def main():
    if not SR_OUTPUT_PATH:
        raise ValueError("Set SR_OUTPUT_PATH in the config section before running.")
    if not PRC1_DIR and not STATS_OVERRIDE_PATH:
        raise ValueError("Set PRC1_DIR or STATS_OVERRIDE_PATH in the config section before running.")

    # Load inputs
    sr = np.load(SR_OUTPUT_PATH)
    stats, meta = load_prc1_artifacts(
        PRC1_DIR,
        stats_override_path=STATS_OVERRIDE_PATH,
        meta_override_path=META_OVERRIDE_PATH,
    )

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

    if EXPORT_EEG_RAW_DATA_FORMAT:
        metadata = _load_metadata(TEST_METADATA_PATH, n_windows=recon.shape[0])
        test_indices = _load_indices(TEST_INDICES_PATH)
        export_reconstructed_as_raw_structure(
            recon=recon,
            raw_export_root=RAW_EXPORT_ROOT,
            metadata=metadata,
            test_indices=test_indices,
        )
        _copy_reference_files(REFERENCE_DATA_ROOT, RAW_EXPORT_ROOT)
        print(
            f"Export complete in raw-data format: {RAW_EXPORT_ROOT} "
            f"(sessions 1/2/3 with .mat files)."
        )

    # Quick sanity check
    print(f"\nSanity check (first window):")
    print(f"  Range: [{recon[0].min():.2f}, {recon[0].max():.2f}] µV")
    print(f"  Mean:  {recon[0].mean():.4f} µV")
    print(f"  Std:   {recon[0].std():.4f} µV")


if __name__ == "__main__":
    main()
