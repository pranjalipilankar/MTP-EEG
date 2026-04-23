import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# --- Config ---
RAW_FS = 8000
TARGET_FS = 8000
HPF = 1.0
LPF = 100.0
NOTCH_FREQS = (50.0, 100.0, 150.0, 200.0)
NOTCH_Q = 30.0

EPOCH_SAMPLES = 2080
SOFT_CLIP_VAL = 5.0
EPS = 1e-6

ENABLE_SOFT_CLIP = True
SPIKE_SAFE_MODE = False

# Input Localize-MI epochs: derivatives/epochs/sub-XX/eeg/*_epochs.npy
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "DATA" / "Localize-MI" / "derivatives" / "epochs"
# Output directory in PrC-1 style
OUTPUT_ROOT = PROJECT_ROOT / "DATA" / "Localize-MI" / "derivatives" / "epochs_prc1"


def bandpass_filter(x, fs, low_hz, high_hz):
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, x, axis=1)


def notch_filter(x, fs, freq_hz, q):
    b, a = iirnotch(freq_hz / (fs / 2.0), q)
    return filtfilt(b, a, x, axis=1)


def global_normalize(window):
    mu = window.mean()
    sigma = window.std() + EPS
    return (window - mu) / sigma, mu, sigma


def soft_clip(x, clip_val):
    return clip_val * np.tanh(x / clip_val)


def preprocess_epoch(epoch):
    """Apply PrC-1 style preprocessing to one epoch (C, T)."""
    x = epoch
    if x.ndim != 2:
        raise ValueError(f"Expected 2D epoch (C, T), got shape={x.shape}")

    if x.shape[1] < EPOCH_SAMPLES:
        pad = EPOCH_SAMPLES - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad)), mode="edge")
    else:
        x = x[:, :EPOCH_SAMPLES]

    x = bandpass_filter(x, RAW_FS, HPF, LPF)
    for freq in NOTCH_FREQS:
        if freq < RAW_FS / 2.0:
            x = notch_filter(x, RAW_FS, freq, NOTCH_Q)

    x_norm, mu, sigma = global_normalize(x)
    if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
        x_norm = soft_clip(x_norm, SOFT_CLIP_VAL)

    return x_norm.astype(np.float32), np.array([mu, sigma], dtype=np.float32)


def invert_prc1(x_norm, stats):
    """Reverse soft-clip and normalization for reference output."""
    x = x_norm.copy()
    if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
        limit = SOFT_CLIP_VAL * (1 - 1e-7)
        x = SOFT_CLIP_VAL * np.arctanh(np.clip(x, -limit, limit) / SOFT_CLIP_VAL)

    for i in range(x.shape[0]):
        mu, sigma = stats[i]
        x[i] = x[i] * sigma + mu
    return x


def process_subject(subject_dir, output_root):
    eeg_dir = subject_dir / "eeg"
    if not eeg_dir.exists():
        return None

    epoch_files = sorted(eeg_dir.glob("*_epochs.npy"))
    if not epoch_files:
        return None

    out_dir = output_root / subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_stats = []
    file_index = []

    for epoch_file in epoch_files:
        data = np.load(epoch_file)  # expected: (N, C, T)
        if data.ndim != 3:
            raise ValueError(f"Unexpected shape in {epoch_file}: {data.shape}")

        for i in range(data.shape[0]):
            x_norm, stats = preprocess_epoch(data[i])
            all_windows.append(x_norm)
            all_stats.append(stats)
            file_index.append({
                "source_file": epoch_file.name,
                "epoch_index": int(i),
            })

    X = np.stack(all_windows, axis=0)
    norm_stats = np.stack(all_stats, axis=0)
    X_reversed = invert_prc1(X, norm_stats)

    np.save(out_dir / "X_prc1.npy", X)
    np.save(out_dir / "X_prc1_norm_stats.npy", norm_stats)
    np.save(out_dir / "X_prc1_reversed.npy", X_reversed)

    with open(out_dir / "window_index.json", "w", encoding="utf-8") as f:
        json.dump(file_index, f, indent=2)

    meta = {
        "dataset": "Localize-MI",
        "subject": subject_dir.name,
        "raw_fs": RAW_FS,
        "target_fs": TARGET_FS,
        "n_channels": int(X.shape[1]),
        "window_samples": int(X.shape[2]),
        "n_windows": int(X.shape[0]),
        "soft_clip_applied": bool(ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE),
        "soft_clip_val": SOFT_CLIP_VAL,
        "bandpass_hz": [HPF, LPF],
        "notch_freqs_hz": list(NOTCH_FREQS),
    }
    with open(out_dir / "prc1_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir, X.shape


def main():
    print("=" * 70)
    print("LOCALIZE-MI DATASET PROCESSING - PrC-1 Pipeline")
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Input directory not found: {DATA_ROOT}")

    subject_dirs = sorted(DATA_ROOT.glob("sub-*"))
    print(f"Found {len(subject_dirs)} subjects")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    processed = 0
    for subject_dir in subject_dirs:
        result = process_subject(subject_dir, OUTPUT_ROOT)
        if result is None:
            print(f"- Skipping {subject_dir.name}: no epoch files")
            continue

        out_dir, shape = result
        processed += 1
        print(f"- {subject_dir.name}: saved {shape} -> {out_dir}")

    print("=" * 70)
    print(f"Completed. Processed subjects: {processed}")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
