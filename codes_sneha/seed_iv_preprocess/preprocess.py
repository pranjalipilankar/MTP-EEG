import os
import numpy as np
import scipy.io as sio
from scipy.signal import iirnotch, filtfilt
from tqdm import tqdm

# ================= CONFIG ================= #
RAW_ROOT = "/home/ab_students/EEG-MTP/DATA/dataset/eeg_raw_data"
OUT_ROOT = "/home/ab_students/EEG-MTP/DATA/dataset_preprocessed/eeg"

FS = 200.0          # sampling rate
NOTCH_FREQ = 50.0   # power line frequency
Q = 30.0            # notch sharpness

MAX_BAD_CHANNELS = 10    # drop trial if exceeded
BAD_Z_THRESH = 5.0       # robust z-score threshold
# ========================================== #


def notch_filter(eeg, fs, f0=50.0, q=30.0):
    """Apply narrow 50 Hz notch filter channel-wise"""
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, eeg, axis=1)


def detect_bad_channels(eeg, z_thresh=5.0):
    """
    Detect bad channels using robust variance statistics.
    Returns mask: True = good channel
    """
    var = np.var(eeg, axis=1)

    median = np.median(var)
    mad = np.median(np.abs(var - median)) + 1e-6

    z_score = (var - median) / mad
    good_mask = np.abs(z_score) < z_thresh

    # remove NaN / Inf channels
    good_mask &= np.isfinite(var)

    return good_mask


def normalize(eeg):
    """Z-score normalization per channel"""
    mean = np.mean(eeg, axis=1, keepdims=True)
    std = np.std(eeg, axis=1, keepdims=True) + 1e-6
    return (eeg - mean) / std


def process_mat_file(mat_path, save_dir):
    mat = sio.loadmat(mat_path)
    base_name = os.path.splitext(os.path.basename(mat_path))[0]

    for key in mat:
        if not key.startswith("cz_eeg"):
            continue

        trial_id = key.replace("cz_eeg", "")
        eeg = mat[key]  # shape: (62, T)

        # ---------- preprocessing ----------
        eeg = notch_filter(eeg, FS, NOTCH_FREQ, Q)

        good_mask = detect_bad_channels(eeg, BAD_Z_THRESH)

        # skip trial if too many bad channels
        if np.sum(~good_mask) > MAX_BAD_CHANNELS:
            continue

        # zero-out bad channels (SR-safe)
        eeg[~good_mask, :] = 0.0

        eeg = normalize(eeg)

        # ---------- save ----------
        eeg_path = os.path.join(
            save_dir, f"{base_name}_trial{trial_id}.npy"
        )
        mask_path = eeg_path.replace(".npy", "_mask.npy")

        np.save(eeg_path, eeg.astype(np.float32))
        np.save(mask_path, good_mask.astype(np.bool_))


def main():
    for session in ["1", "2", "3"]:
        in_dir = os.path.join(RAW_ROOT, session)
        out_dir = os.path.join(OUT_ROOT, session)
        os.makedirs(out_dir, exist_ok=True)

        mat_files = [f for f in os.listdir(in_dir) if f.endswith(".mat")]

        print(f"\nProcessing session {session} | Files: {len(mat_files)}")
        for mf in tqdm(mat_files):
            process_mat_file(os.path.join(in_dir, mf), out_dir)


if __name__ == "__main__":
    main()
