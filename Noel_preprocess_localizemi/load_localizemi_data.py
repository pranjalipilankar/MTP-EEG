"""
Helper script to load processed Localize-MI PrC-1 data.

Usage example:
    from load_localizemi_data import load_subject, load_all_subjects

    X, meta = load_subject('sub-01')
    all_data = load_all_subjects()
"""

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/ab_students/EEG-MTP/DATA/Localize-MI/derivatives/epochs_prc1")


def load_subject(subject_id, base_dir=BASE_DIR, load_reversed=False):
    """Load one subject's preprocessed Localize-MI windows."""
    subject_path = Path(base_dir) / subject_id
    if not subject_path.exists():
        raise FileNotFoundError(f"Data not found: {subject_path}")

    x = np.load(subject_path / "X_prc1.npy")
    stats = np.load(subject_path / "X_prc1_norm_stats.npy")

    with open(subject_path / "prc1_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["stats_shape"] = list(stats.shape)

    if load_reversed:
        x_reversed = np.load(subject_path / "X_prc1_reversed.npy")
        return x, stats, meta, x_reversed

    return x, stats, meta


def load_all_subjects(base_dir=BASE_DIR, load_reversed=False):
    """Load all subjects under Localize-MI processed directory."""
    base_path = Path(base_dir)
    all_data = {}

    for subject_dir in sorted(base_path.glob("sub-*")):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        try:
            if load_reversed:
                x, stats, meta, x_reversed = load_subject(subject_id, base_dir, load_reversed=True)
                all_data[subject_id] = {
                    "X": x,
                    "stats": stats,
                    "meta": meta,
                    "X_reversed": x_reversed,
                }
            else:
                x, stats, meta = load_subject(subject_id, base_dir, load_reversed=False)
                all_data[subject_id] = {
                    "X": x,
                    "stats": stats,
                    "meta": meta,
                }
        except Exception as exc:
            print(f"Skipping {subject_id}: {exc}")

    return all_data


def summary(base_dir=BASE_DIR):
    """Print a compact summary of processed Localize-MI data."""
    all_data = load_all_subjects(base_dir=base_dir, load_reversed=False)
    if not all_data:
        print("No processed data found.")
        return

    total_windows = 0
    for subject_id, payload in all_data.items():
        n_windows = payload["X"].shape[0]
        total_windows += n_windows
        print(f"{subject_id}: {payload['X'].shape} (windows={n_windows})")

    print(f"Total subjects: {len(all_data)}")
    print(f"Total windows: {total_windows}")


if __name__ == "__main__":
    summary()
