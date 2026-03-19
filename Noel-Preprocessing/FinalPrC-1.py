import os, json, time
import numpy as np
from scipy.io import loadmat
from scipy.signal import firwin, filtfilt, sosfiltfilt, butter, resample_poly, welch, hilbert, iirnotch
from scipy.stats import skew, kurtosis, ks_2samp, ttest_rel, ttest_1samp, wilcoxon, shapiro, kstest, wasserstein_distance
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mne

# --- Config ---

RAW_FS = 512
TARGET_FS = 250

HPF = 0.1
LPF = 100
NOTCH_FREQ = 50
NOTCH_Q = 30

WINDOW_SEC = 4

SOFT_CLIP_VAL = 5.0
EPS = 1e-6

ENABLE_SOFT_CLIP = True
SPIKE_SAFE_MODE = False     # disables soft clipping when True

ENABLE_BAD_CHANNEL_INTERP = True
ENABLE_VALIDATION = False

FLAT_STD_THRESH = 1e-6        # flatline detection threshold
VAR_Z_THRESH = 5.0            # variance outlier z-score threshold
CORR_THRESH = 0.15            # min mean abs correlation with other channels
MAX_BAD_CHANNELS = 0.2        # max fraction of channels allowed to be bad

# -------- SEED IV CHANNEL NAMES (62ch, 10-20 system) --------
SEED_CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2'
]

# MNE-compatible names (mixed case to match MNE conventions).
# SEED-IV labels CB1/CB2 are an alternative naming convention for the I1/I2
# positions defined by Oostenveld & Praamstra (2001).  MNE's standard_1005
# montage uses I1/I2, so we map CB1→I1 and CB2→I2.
MNE_CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'I1', 'O1', 'Oz',
    'O2', 'I2'
]

# Root directories for SEED-IV dataset
BASE_DIR = "/home/ab_students/EEG-MTP/DATA/seed4"
RAW_DATA_DIR = f"{BASE_DIR}/eeg_raw_data"
PROCESSED_DATA_DIR = f"{BASE_DIR}/eeg_processed_data"

# --- SEED-IV Emotion Labels ---
# Labels for the 24 trials in each session
# 0=neutral, 1=sad, 2=fear, 3=happy
SESSION_LABELS = {
    '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

EMOTION_LABELS = {
    0: 'neutral',
    1: 'sad', 
    2: 'fear',
    3: 'happy'
}

# --- Signal operations ---

def zero_phase_bandpass(x, fs, l_freq, h_freq):
    """IIR high-pass (Butterworth, sosfiltfilt) + FIR low-pass.

    Using IIR for the high-pass avoids the massive FIR tap count that
    a 0.1 Hz cutoff would require (~16 900 taps at 512 Hz).  sosfiltfilt
    gives zero-phase response and second-order-section form ensures
    numerical stability even at very low Wn.
    """
    # --- IIR high-pass (zero-phase via sosfiltfilt) ---
    sos_hp = butter(N=4, Wn=l_freq, btype='high', fs=fs, output='sos')
    x = sosfiltfilt(sos_hp, x, axis=1)

    # --- FIR low-pass (zero-phase via filtfilt) ---
    nyq = fs / 2
    trans_bw = min(5.0, (nyq - h_freq) * 0.5)  # transition bandwidth (Hz)
    n = int(np.ceil(3.3 * fs / trans_bw))       # ~337 taps for 5 Hz transition
    max_n = (x.shape[1] - 1) // 3              # filtfilt padding constraint
    n = min(n, max_n)
    n = n if n % 2 == 1 else n - 1             # firwin needs odd length
    taps = firwin(numtaps=n, cutoff=h_freq / nyq)
    return filtfilt(taps, [1.0], x, axis=1)

def notch_filter(x, fs, freq, q):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, x, axis=1)

def global_normalize(w):
    mu = w.mean()
    sigma = w.std() + EPS
    return (w - mu) / sigma, mu, sigma

def soft_clip(w, clip_val):
    return clip_val * np.tanh(w / clip_val)

def window_signal(x):
    win_len = int(WINDOW_SEC * TARGET_FS)
    windows, stats = [], []

    for start in range(0, x.shape[1] - win_len, win_len):
        w = x[:, start:start + win_len]
        w_norm, mu, sigma = global_normalize(w)

        if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
            w_norm = soft_clip(w_norm, SOFT_CLIP_VAL)

        windows.append(w_norm)
        stats.append((mu, sigma))

    return np.stack(windows), np.array(stats)

# --- Bad channel detection & interpolation ---

def detect_bad_channels(x):
    """Returns boolean mask of bad channels (flatline | variance outlier | uncorrelated)."""
    ch_std = x.std(axis=1)
    ch_var = ch_std ** 2

    flat = ch_std < FLAT_STD_THRESH

    z = (ch_var - np.median(ch_var)) / (np.std(ch_var) + 1e-8)
    noisy = np.abs(z) > VAR_Z_THRESH

    corr = np.corrcoef(x)
    np.fill_diagonal(corr, 0)
    mean_corr = np.mean(np.abs(corr), axis=1)
    uncorrelated = mean_corr < CORR_THRESH

    bad = flat | noisy | uncorrelated

    # If too many are flagged the trial is probably globally noisy; skip
    if bad.sum() > MAX_BAD_CHANNELS * x.shape[0]:
        return np.zeros_like(bad, dtype=bool)

    return bad


def interpolate_bad_channels(x, bad_mask, sfreq=RAW_FS):
    """Spherical spline interpolation via MNE (standard_1005 montage)."""
    if not bad_mask.any():
        return x

    info = mne.create_info(ch_names=MNE_CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    raw = mne.io.RawArray(x, info, verbose=False)
    raw.set_montage(montage, on_missing='warn', verbose=False)

    bad_ch_names = [MNE_CHANNEL_NAMES[i] for i in np.where(bad_mask)[0]]
    raw.info['bads'] = bad_ch_names
    raw.interpolate_bads(mode='accurate', verbose=False)

    return raw.get_data()

# --- Diagnostic plots ---

def save_stage_plot(signal, title, ylabel, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def inspect_single_window(mat, key, out_dir, channel_idx=0, window_idx=0):
    """Save stage-by-stage plots for one representative window."""
    os.makedirs(out_dir, exist_ok=True)

    x_raw = mat[key]
    x_raw = x_raw if x_raw.shape[0] == 62 else x_raw.T

    raw_win_len = int(WINDOW_SEC * RAW_FS)
    save_stage_plot(x_raw[channel_idx, :raw_win_len],
                    "Raw EEG window", "µV (raw)", f"{out_dir}/01_raw_window.png")

    x = zero_phase_bandpass(x_raw, RAW_FS, HPF, LPF)
    x = notch_filter(x, RAW_FS, NOTCH_FREQ, NOTCH_Q)
    if ENABLE_BAD_CHANNEL_INTERP:
        x = interpolate_bad_channels(x, detect_bad_channels(x))
    x = resample_poly(x, TARGET_FS, RAW_FS, axis=1)

    proc_win_len = int(WINDOW_SEC * TARGET_FS)
    save_stage_plot(x[channel_idx, :proc_win_len],
                    "Post-filter + downsample (pre-norm)", "µV (approx)",
                    f"{out_dir}/02_pre_normalization.png")

    w = x[:, :proc_win_len]
    mu, sigma = w.mean(), w.std() + EPS
    w_norm = (w - mu) / sigma
    save_stage_plot(w_norm[channel_idx],
                    "Post global z-normalization", "z-scored",
                    f"{out_dir}/03_post_normalization.png")

    if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
        w_soft = SOFT_CLIP_VAL * np.tanh(w_norm / SOFT_CLIP_VAL)

        save_stage_plot(
            w_soft[channel_idx],
            "Post soft clipping (model input)",
            "Normalized units (soft clipped)",
            f"{out_dir}/04_post_soft_clipping.png"
        )

    print("Saved PrC-1 visual inspection plots.")

# --- PrC-1 pipeline ---

def prc1_preprocess_trial(x):
    """Full PrC-1: bandpass → notch → bad-ch repair → downsample → window → normalize."""
    x = zero_phase_bandpass(x, RAW_FS, HPF, LPF)
    x = notch_filter(x, RAW_FS, NOTCH_FREQ, NOTCH_Q)
    if ENABLE_BAD_CHANNEL_INTERP:
        x = interpolate_bad_channels(x, detect_bad_channels(x))
    x = resample_poly(x, TARGET_FS, RAW_FS, axis=1)
    return window_signal(x)


def prc1_preprocess_trial_with_prenorm(x):
    """Same as above but also returns pre-norm signal and bad channel mask."""
    x = zero_phase_bandpass(x, RAW_FS, HPF, LPF)
    x = notch_filter(x, RAW_FS, NOTCH_FREQ, NOTCH_Q)

    bad_mask = np.zeros(x.shape[0], dtype=bool)
    if ENABLE_BAD_CHANNEL_INTERP:
        bad_mask = detect_bad_channels(x)
        x = interpolate_bad_channels(x, bad_mask)

    x = resample_poly(x, TARGET_FS, RAW_FS, axis=1)
    pre_norm_x = x.copy()
    windows, stats = window_signal(x)
    return windows, stats, pre_norm_x, bad_mask

# --- Helper function to process a single .mat file ---
def process_mat_file(mat_file_path):
    """Process a single .mat file and save outputs."""
    print(f"\n{'='*70}")
    print(f"Processing: {mat_file_path}")
    print(f"{'='*70}")
    
    # Extract dataset name from MAT_FILE for unique output folder
    dataset_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    
    # Get session folder (1, 2, or 3)
    session_folder = os.path.basename(os.path.dirname(mat_file_path))
    
    # Create output directory structure: session/subject
    out_dir = f"{PROCESSED_DATA_DIR}/{session_folder}/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Dataset: {dataset_name}")
    print(f"Session: {session_folder}")
    print(f"Output directory: {out_dir}")
    
    # --- Load & process ---
    mat = loadmat(mat_file_path)
    all_keys = [k for k in mat.keys() if not k.startswith("__")]
    eeg_keys = [k for k in all_keys if "eeg" in k.lower()]
    
    if not eeg_keys:
        raise ValueError(f"No EEG keys found in {mat_file_path}. Available keys: {all_keys}")
    
    print(f"Found {len(eeg_keys)} EEG trials with prefix: {eeg_keys[0].rsplit('eeg', 1)[0]}eeg*")
    
    # Get trial labels for this session
    session_trial_labels = SESSION_LABELS[session_folder]
    if len(eeg_keys) != len(session_trial_labels):
        print(f"  WARNING: Number of EEG trials ({len(eeg_keys)}) doesn't match expected labels ({len(session_trial_labels)})")
    
    all_windows = []
    all_stats = []
    all_pre_norm = []
    all_bad_masks = []
    all_labels = []  # Labels for each window
    
    t_start = time.time()
    for i, key in enumerate(eeg_keys):
        print(f"  Trial {i+1}/{len(eeg_keys)}: {key}", end="\r")
        x = mat[key]
        x = x if x.shape[0] == 62 else x.T
    
        w, s, pre_norm_x, bad_mask = prc1_preprocess_trial_with_prenorm(x)
        all_windows.append(w)
        all_stats.append(s)
        all_pre_norm.append(pre_norm_x)
        all_bad_masks.append(bad_mask)
        
        # Assign label to all windows from this trial
        trial_label = session_trial_labels[i]
        trial_labels = np.full(w.shape[0], trial_label, dtype=np.int32)  # Shape: (n_windows,)
        all_labels.append(trial_labels)
        
    print(f"\n  Done in {time.time() - t_start:.1f}s")
    
    X = np.concatenate(all_windows, axis=0)
    stats = np.concatenate(all_stats, axis=0)
    labels = np.concatenate(all_labels, axis=0)  # Shape: (total_windows,)
    
    print(f"  Total windows: {X.shape[0]}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    for lbl, emotion in EMOTION_LABELS.items():
        count = np.sum(labels == lbl)
        if count > 0:
            print(f"    {emotion}: {count} windows ({count/len(labels)*100:.1f}%)")
    
    first_raw_x = mat[eeg_keys[0]]
    first_raw_x = first_raw_x if first_raw_x.shape[0] == 62 else first_raw_x.T
    pre_norm_x = all_pre_norm[0]
    
    # Downsample raw signal to TARGET_FS for fair validation comparisons
    raw_ds = resample_poly(first_raw_x, TARGET_FS, RAW_FS, axis=1)
    
    # Build full raw reference (all trials) for distribution comparisons
    raw_all_ds_list = []
    for k in eeg_keys:
        rx = mat[k]
        rx = rx if rx.shape[0] == 62 else rx.T
        raw_all_ds_list.append(resample_poly(rx, TARGET_FS, RAW_FS, axis=1))
    raw_all_ds = np.concatenate(raw_all_ds_list, axis=1)  # (62, total_samples)
    
    np.save(f"{out_dir}/X_prc1.npy", X)
    np.save(f"{out_dir}/X_prc1_norm_stats.npy", stats)
    np.save(f"{out_dir}/labels.npy", labels)
    
    # Also save trial-level labels for reference
    trial_labels_dict = {
        "trial_labels": session_trial_labels,
        "emotion_mapping": EMOTION_LABELS,
        "n_trials": len(eeg_keys),
        "windows_per_trial": [all_labels[i].shape[0] for i in range(len(all_labels))]
    }
    with open(f"{out_dir}/trial_labels.json", "w") as f:
        json.dump(trial_labels_dict, f, indent=2)
    
    # --- Reversal output (reference for pointwise evaluation) ---
    # Invert the same transformations applied during PrC-1 to produce a µV-scale
    # reference signal.  Comparing this against the SR model's reversed output
    # (from reconstruct_prc1.py) gives a fair RMSE / MAE / correlation metric
    # because BOTH signals have passed through the same forward→inverse path.
    # Do NOT compare the SR reversed output against raw — that conflates
    # preprocessing distortion with SR model error.
    X_reversed = X.copy()
    if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
        # Invert soft clip: arctanh
        limit = SOFT_CLIP_VAL * (1 - 1e-7)
        X_reversed = SOFT_CLIP_VAL * np.arctanh(np.clip(X_reversed, -limit, limit) / SOFT_CLIP_VAL)
    # Invert z-normalization per window
    for i in range(X_reversed.shape[0]):
        mu_i, sigma_i = stats[i]
        X_reversed[i] = X_reversed[i] * sigma_i + mu_i
    
    np.save(f"{out_dir}/X_prc1_reversed.npy", X_reversed)
    print(f"Saved PrC-1 reversed reference: {X_reversed.shape}")
    
    prc1_meta = {
        "target_fs": TARGET_FS,
        "window_sec": WINDOW_SEC,
        "soft_clip_applied": ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE,
        "soft_clip_val": SOFT_CLIP_VAL,
        "n_channels": X.shape[1],
        "n_windows": X.shape[0],
        "window_samples": X.shape[2],
        "dataset": dataset_name,
        "session": session_folder,
        "n_trials": len(eeg_keys),
        "emotion_labels": EMOTION_LABELS,
        "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }
    with open(f"{out_dir}/prc1_meta.json", "w") as f:
        json.dump(prc1_meta, f, indent=2)
    
    print("Saved PrC-1 output:", X.shape)
    
    inspect_single_window(mat=mat, key=eeg_keys[0], out_dir=f"{out_dir}/inspection")
    
    return out_dir, mat, eeg_keys, all_pre_norm, all_bad_masks, X, stats, labels, first_raw_x, pre_norm_x, raw_ds, raw_all_ds

# --- Main processing loop for entire dataset ---
print("\n" + "="*70)
print("SEED-IV DATASET PROCESSING - PrC-1 Pipeline")
print("="*70)

# Find all session folders (1, 2, 3)
session_folders = sorted([f for f in os.listdir(RAW_DATA_DIR) 
                          if os.path.isdir(os.path.join(RAW_DATA_DIR, f)) and f.isdigit()])

print(f"Found {len(session_folders)} sessions: {session_folders}")

total_files = 0
processed_files = 0
failed_files = []

overall_start = time.time()

for session_folder in session_folders:
    session_path = os.path.join(RAW_DATA_DIR, session_folder)
    mat_files = sorted([f for f in os.listdir(session_path) if f.endswith('.mat')])
    
    print(f"\n{'='*70}")
    print(f"SESSION {session_folder}: {len(mat_files)} files")
    print(f"{'='*70}")
    
    for mat_file in mat_files:
        mat_file_path = os.path.join(session_path, mat_file)
        total_files += 1
        
        try:
            out_dir, mat, eeg_keys, all_pre_norm, all_bad_masks, X, stats, labels, first_raw_x, pre_norm_x, raw_ds, raw_all_ds = process_mat_file(mat_file_path)
            processed_files += 1
            print(f"✓ Successfully processed {mat_file}")
        except Exception as e:
            print(f"✗ Failed to process {mat_file}: {str(e)}")
            failed_files.append((mat_file_path, str(e)))
            continue

print(f"\n{'='*70}")
print("PROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Total files: {total_files}")
print(f"Successfully processed: {processed_files}")
print(f"Failed: {len(failed_files)}")
print(f"Total time: {time.time() - overall_start:.1f}s")

if failed_files:
    print("\nFailed files:")
    for fpath, error in failed_files:
        print(f"  - {fpath}")
        print(f"    Error: {error}")

# For validation, use the last processed file
print(f"\nValidation will run on the last processed file: {mat_file_path}")

# --- Quick validation ---

if ENABLE_VALIDATION and processed_files > 0:
    
    # Use variables from last processed file (already in scope from the loop)
    OUT_DIR = out_dir  # Use the out_dir from the last processed file

    def effective_rank(x):
        eigs = np.linalg.eigvalsh(np.cov(x))
        return np.sum(eigs > 1e-6 * eigs.max())

    print("\n=== VALIDATION ===")
    print("NaNs:", np.isnan(X).any())
    print("Global mean:", X.mean())
    print("Global std:", X.std())
    print("Effective rank (first window):", effective_rank(X[0]))

    # Covariance spectrum
    eigs = np.linalg.eigvalsh(np.cov(X[0]))
    plt.figure()
    plt.semilogy(eigs[::-1])
    plt.title("Covariance spectrum (PrC-1 Final)")
    plt.xlabel("Mode")
    plt.ylabel("Eigenvalue (log)")
    plt.savefig(f"{OUT_DIR}/covariance_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Invert normalization for one window
    w_norm = X[0]
    mu, sigma = stats[0]
    w_reconstructed = w_norm * sigma + mu

    plt.figure(figsize=(10, 4))
    plt.plot(w_reconstructed[0])
    plt.title("Reconstructed amplitude scale (soft clipping)")
    plt.xlabel("Samples")
    plt.ylabel("µV (approx)")
    plt.savefig(f"{OUT_DIR}/reconstructed_amplitude.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll outputs saved to: {OUT_DIR}")

    VALIDATION_DIR = f"{OUT_DIR}/validation"
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    def savefig(name):
        plt.tight_layout()
        plt.savefig(f"{VALIDATION_DIR}/validation_{name}.png", dpi=150)
        plt.close()

    print("\n" + "="*60)
    print("PrC-1 VALIDATION")
    print("="*60)

    win_len = int(WINDOW_SEC * TARGET_FS)

    # Reconstruct bypassing soft clip: z-normalize pre_norm then invert
    w0 = pre_norm_x[:, :win_len]
    mu0, sig0 = w0.mean(), w0.std() + EPS
    linear_reconstructed = ((w0 - mu0) / sig0) * sig0 + mu0   # z-norm → invert (should be identity)
    reconstructed_first = linear_reconstructed[0]
    reconstructed_window = linear_reconstructed

    # Also store the soft-clipped reconstruction for separate checks
    softclip_reconstructed = X[0] * stats[0][1] + stats[0][0]

    # [1] Cross-correlation lag check
    print("\n[1] Cross-correlation lag")
    corr_full = np.correlate(
        pre_norm_x[0, :win_len] - pre_norm_x[0, :win_len].mean(),
        reconstructed_first - reconstructed_first.mean(), mode="full")
    lag = np.argmax(corr_full) - (win_len - 1)
    print(f"  Lag (samples): {lag}")

    # [2] Phase preservation — quantitative QC
    print("\n[2] Phase preservation (quantitative)")

    # --- QC tolerances ---
    PHASE_MAE_TOL       = 0.05   # mean |Δφ| in radians (≈ 2.9°)
    PHASE_PLV_TOL       = 0.99   # phase-locking value lower bound
    PHASE_EXCEED_TOL    = 0.01   # max fraction of samples with |Δφ| > π/8
    PHASE_CIRC_CORR_TOL = 0.99   # circular correlation lower bound
    PHASE_EXCEED_RAD    = np.pi / 8  # threshold for "large" phase error

    def phase_qc(phi_ref, phi_test, label):
        """Compute quantitative phase-preservation metrics and return pass/fail."""
        delta = np.angle(np.exp(1j * (phi_ref - phi_test)))  # wrapped difference
        mae   = np.mean(np.abs(delta))
        plv   = np.abs(np.mean(np.exp(1j * delta)))
        exceed_frac = np.mean(np.abs(delta) > PHASE_EXCEED_RAD)

        # circular correlation  (Fisher & Lee, 1983)
        phi_ref_c  = phi_ref  - np.mean(phi_ref)
        phi_test_c = phi_test - np.mean(phi_test)
        circ_corr = np.abs(
            np.sum(np.sin(phi_ref_c) * np.sin(phi_test_c))
            / np.sqrt(np.sum(np.sin(phi_ref_c)**2) * np.sum(np.sin(phi_test_c)**2) + EPS)
        )

        pass_mae    = mae          < PHASE_MAE_TOL
        pass_plv    = plv          > PHASE_PLV_TOL
        pass_exceed = exceed_frac  < PHASE_EXCEED_TOL
        pass_circ   = circ_corr    > PHASE_CIRC_CORR_TOL
        overall     = pass_mae and pass_plv and pass_exceed and pass_circ

        tag = "PASS" if overall else "FAIL"
        print(f"  [{label}]  {tag}")
        print(f"    Mean |Δφ|        : {mae:.6f} rad  (tol < {PHASE_MAE_TOL})  {'✓' if pass_mae else '✗'}")
        print(f"    PLV              : {plv:.6f}      (tol > {PHASE_PLV_TOL})  {'✓' if pass_plv else '✗'}")
        print(f"    Exceed π/8 frac  : {exceed_frac:.6f}  (tol < {PHASE_EXCEED_TOL})  {'✓' if pass_exceed else '✗'}")
        print(f"    Circ. correlation: {circ_corr:.6f}  (tol > {PHASE_CIRC_CORR_TOL})  {'✓' if pass_circ else '✗'}")

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(np.unwrap(delta))
        plt.axhline(y=0, color='k', ls='--', lw=0.5)
        plt.title(f"Phase difference: {label}  [{tag}]")
        plt.xlabel("Samples"); plt.ylabel("Radians")
        return overall

    phi_pre = np.angle(hilbert(pre_norm_x[0, :win_len]))

    # (a) Linear path (no soft clip) — should be ~0
    phi_lin = np.angle(hilbert(reconstructed_first))
    phase_qc(phi_pre, phi_lin, "Linear path (no soft clip)")
    savefig("phase_preservation_linear")

    # (b) Soft-clipped path — shows tanh distortion at extreme samples
    phi_clip = np.angle(hilbert(softclip_reconstructed[0]))
    phase_qc(phi_pre, phi_clip, "Soft-clipped path")
    savefig("phase_preservation_softclip")

    print("  Saved: validation_phase_preservation_linear.png, validation_phase_preservation_softclip.png")

    # [3] PSD comparison — quantitative QC
    print("\n[3] PSD comparison (quantitative)")

    # --- QC tolerances ---
    PSD_CORR_TOL        = 0.999   # Pearson r between PSD curves
    PSD_PCT_CHANGE_TOL  = 1.0     # max % change in total band power
    PSD_KS_ALPHA        = 0.05    # KS test significance level
    PSD_LOG_DIST_TOL    = 0.01    # max mean |log10(P_after/P_before)| (log spectral distance)

    sig_before = pre_norm_x[0, :win_len]
    sig_after  = reconstructed_first

    f_b, p_b = welch(sig_before, fs=TARGET_FS, nperseg=min(1024, len(sig_before)))
    f_a, p_a = welch(sig_after,  fs=TARGET_FS, nperseg=min(1024, len(sig_after)))

    # (i) Pearson correlation between PSD curves
    psd_corr = np.corrcoef(p_b, p_a)[0, 1]
    pass_corr = psd_corr > PSD_CORR_TOL

    # (ii) % change in total band power
    total_before = p_b.sum()
    total_after  = p_a.sum()
    pct_change   = np.abs(total_after - total_before) / (total_before + EPS) * 100
    pass_pct     = pct_change < PSD_PCT_CHANGE_TOL

    # (iii) KS test on normalised PSD (treated as probability distributions)
    p_b_pdf = p_b / (p_b.sum() + EPS)
    p_a_pdf = p_a / (p_a.sum() + EPS)
    ks_stat, ks_p = ks_2samp(
        np.random.choice(f_b, size=5000, p=p_b_pdf),
        np.random.choice(f_a, size=5000, p=p_a_pdf)
    )
    pass_ks = ks_p > PSD_KS_ALPHA

    # (iv) Log spectral distance
    log_dist = np.mean(np.abs(np.log10((p_a + EPS) / (p_b + EPS))))
    pass_log = log_dist < PSD_LOG_DIST_TOL

    overall_psd = pass_corr and pass_pct and pass_ks and pass_log
    tag = "PASS" if overall_psd else "FAIL"

    print(f"  PSD QC: {tag}")
    print(f"    Pearson r        : {psd_corr:.6f}    (tol > {PSD_CORR_TOL})    {'✓' if pass_corr else '✗'}")
    print(f"    % power change   : {pct_change:.4f}%   (tol < {PSD_PCT_CHANGE_TOL}%)  {'✓' if pass_pct else '✗'}")
    print(f"    KS stat / p-value: {ks_stat:.4f} / {ks_p:.4f}  (p > {PSD_KS_ALPHA})     {'✓' if pass_ks else '✗'}")
    print(f"    Log spectral dist: {log_dist:.6f}    (tol < {PSD_LOG_DIST_TOL})    {'✓' if pass_log else '✗'}")

    # PSD overlay plot (still useful visually)
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_b, p_b, label="Pre-norm")
    plt.semilogy(f_a, p_a, label="Reconstructed", ls='--')
    plt.legend(); plt.title(f"PSD comparison  [{tag}]")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power")
    savefig("psd")

    # % change per frequency bin plot
    pct_per_bin = np.abs(p_a - p_b) / (p_b + EPS) * 100
    plt.figure(figsize=(10, 4))
    plt.plot(f_b, pct_per_bin)
    plt.axhline(y=PSD_PCT_CHANGE_TOL, color='r', ls='--', label=f'{PSD_PCT_CHANGE_TOL}% tol')
    plt.legend(); plt.title("PSD % change per frequency bin")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("% change")
    savefig("psd_pct_change")

    # Raw vs. processed PSD overlay — visualizes filtering effect
    raw_seg = raw_ds[0, :win_len]
    f_raw, p_raw = welch(raw_seg, fs=TARGET_FS, nperseg=min(1024, len(raw_seg)))
    plt.figure(figsize=(10, 4))
    plt.semilogy(f_raw, p_raw, label="Raw (downsampled)", alpha=0.7)
    plt.semilogy(f_b, p_b, label="After PrC-1 (pre-norm)")
    plt.axvline(x=HPF, color='g', ls=':', lw=0.8, label=f'HPF = {HPF} Hz')
    plt.axvline(x=LPF, color='r', ls=':', lw=0.8, label=f'LPF = {LPF} Hz')
    plt.axvline(x=NOTCH_FREQ, color='orange', ls=':', lw=0.8, label=f'Notch = {NOTCH_FREQ} Hz')
    plt.legend(); plt.title("PSD: Raw vs. Processed")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power")
    savefig("psd_raw_vs_processed")
    print("  Saved: validation_psd.png, validation_psd_pct_change.png, validation_psd_raw_vs_processed.png")

    # Precompute per-channel PSD once (vectorized)
    _nperseg = min(1024, win_len)
    _f_psd, _psd_raw  = welch(raw_ds[:, :win_len],  fs=TARGET_FS, nperseg=_nperseg, axis=1)
    _,      _psd_proc = welch(pre_norm_x[:, :win_len], fs=TARGET_FS, nperseg=_nperseg, axis=1)

    def _bp(psd, fmin, fmax):
        """Band power from precomputed PSD."""
        mask = (_f_psd >= fmin) & (_f_psd <= fmax)
        return psd[:, mask].sum(axis=1) if psd.ndim == 2 else psd[mask].sum()

    # [4] Band-power ratios (raw vs. processed)
    print("\n[4] Band-power ratios")

    BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}
    print(f"  {'Band':<8} {'Raw':>12} {'Processed':>12} {'Ratio (P/R)':>12}")
    for bname, (fmin, fmax) in BANDS.items():
        bp_raw  = _bp(_psd_raw[0], fmin, fmax)
        bp_proc = _bp(_psd_proc[0], fmin, fmax)
        print(f"  {bname:<8} {bp_raw:12.4f} {bp_proc:12.4f} {bp_proc/(bp_raw+EPS):12.4f}")

    alpha_power = _bp(_psd_proc[0], 8, 13)
    theta_power = _bp(_psd_proc[0], 4, 8)
    alpha_theta_ratio = alpha_power / (theta_power + EPS)
    print(f"  Alpha/Theta ratio (processed): {alpha_theta_ratio:.4f}")

    # Per-channel alpha power paired t-test (raw vs processed)
    t_alpha, p_alpha = ttest_rel(_bp(_psd_raw, 8, 13), _bp(_psd_proc, 8, 13))
    print(f"  Paired t-test (alpha per channel): t={t_alpha:.3f}, p={p_alpha:.4f}")

    # [5] Topography correlation (spatial pattern preservation)
    print("\n[5] Topography correlation")
    num_t = min(100, win_len, pre_norm_x.shape[1], reconstructed_window.shape[1])
    topo_corrs_proc = [np.corrcoef(pre_norm_x[:, t], reconstructed_window[:, t])[0, 1]
                       for t in range(num_t)]
    topo_corrs_raw  = [np.corrcoef(raw_ds[:, t], pre_norm_x[:, t])[0, 1]
                       for t in range(num_t)]
    print(f"  Pre-norm  vs reconstructed (mean r): {np.mean(topo_corrs_proc):.4f}")
    print(f"  Raw       vs processed     (mean r): {np.mean(topo_corrs_raw):.4f}")

    # [6] Channel correlation matrix similarity
    print("\n[6] Channel corr-matrix similarity")
    C_raw  = np.corrcoef(raw_ds[:, :win_len])
    C_proc = np.corrcoef(pre_norm_x[:, :win_len])
    C_recon = np.corrcoef(reconstructed_window)
    raw_proc_sim  = np.corrcoef(C_raw.flatten(), C_proc.flatten())[0, 1]
    proc_recon_sim = np.corrcoef(C_proc.flatten(), C_recon.flatten())[0, 1]
    print(f"  Raw  vs processed    : {raw_proc_sim:.4f}")
    print(f"  Processed vs reconst.: {proc_recon_sim:.4f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(C_raw, cmap='RdBu_r', vmin=-1, vmax=1); plt.title('Raw'); plt.colorbar(shrink=0.6)
    plt.subplot(1, 3, 2)
    plt.imshow(C_proc, cmap='RdBu_r', vmin=-1, vmax=1); plt.title('Processed'); plt.colorbar(shrink=0.6)
    plt.subplot(1, 3, 3)
    plt.imshow(C_recon, cmap='RdBu_r', vmin=-1, vmax=1); plt.title('Reconstructed'); plt.colorbar(shrink=0.6)
    savefig("corr_matrix_comparison")

    # [7] Distribution symmetry
    print("\n[7] Distribution symmetry")
    vals = X.flatten()
    # Reconstruct all windows back to µV scale
    recon_vals = (X * stats[:, 1:2, np.newaxis] + stats[:, 0:1, np.newaxis]).flatten()
    raw_vals = raw_all_ds.flatten()  # all trials for fair comparison
    print(f"  Processed — Skewness: {skew(vals):.4f}, Kurtosis: {kurtosis(vals):.4f}")
    print(f"  Reconstr. — Skewness: {skew(recon_vals):.4f}, Kurtosis: {kurtosis(recon_vals):.4f}")
    print(f"  Raw       — Skewness: {skew(raw_vals):.4f}, Kurtosis: {kurtosis(raw_vals):.4f}")
    plt.figure(figsize=(10, 4))
    plt.hist(raw_vals, bins=200, alpha=0.5, label='Raw', density=True)
    plt.hist(recon_vals, bins=200, alpha=0.5, label='Reconstructed', density=True)
    plt.legend(); plt.title("Value distribution: Raw vs Reconstructed")
    plt.xlabel("µV"); plt.ylabel("Density")
    savefig("histogram")

    # [8] Normalization stability
    print("\n[8] μ/σ stability")
    print(f"  μ range: [{stats[:,0].min():.4f}, {stats[:,0].max():.4f}]")
    print(f"  σ range: [{stats[:,1].min():.4f}, {stats[:,1].max():.4f}]")
    plt.figure(figsize=(10, 4))
    plt.hist(stats[:,1], bins=50); plt.title("σ distribution")
    plt.xlabel("σ"); plt.ylabel("Count")
    savefig("sigma_distribution")

    # [9] SR learnability proxy (can ch0 be predicted from the rest?)
    print("\n[9] SR learnability proxy")
    Xr = pre_norm_x[1:, :win_len].T
    score = LinearRegression().fit(Xr, pre_norm_x[0, :win_len]).score(Xr, pre_norm_x[0, :win_len])
    print(f"  R² (ch0 from remaining): {score:.4f}")

    # [10] HF noise floor
    print("\n[10] HF noise floor")
    print(f"  Power 60-80 Hz: {_bp(_psd_proc[0], 60, 80):.6f}")

    # [11] Bad channel diagnosis
    print("\n[11] Bad channels")

    total_bad_per_trial = [m.sum() for m in all_bad_masks]
    all_bad_union = np.zeros(62, dtype=bool)
    for m in all_bad_masks:
        all_bad_union |= m

    print(f"  Per trial: {total_bad_per_trial}")
    print(f"  Unique bad (union): {all_bad_union.sum()}")
    if all_bad_union.any():
        print(f"  Indices: {np.where(all_bad_union)[0].tolist()}")

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(total_bad_per_trial)), total_bad_per_trial, color='salmon')
    plt.xlabel("Trial"); plt.ylabel("# Bad channels")
    plt.title("Bad channels per trial")
    plt.axhline(y=MAX_BAD_CHANNELS * 62, color='r', ls='--', label=f'{MAX_BAD_CHANNELS*100:.0f}% cap')
    plt.legend()
    savefig("bad_channel_count_per_trial")

    # [12] Statistical tests
    print("\n[12] Statistical tests")

    win_means = np.array([X[i].mean() for i in range(X.shape[0])])
    win_stds  = np.array([X[i].std()  for i in range(X.shape[0])])

    if len(win_means) >= 3:
        sw_stat, sw_p = shapiro(win_means[:min(5000, len(win_means))])
        print(f"  Shapiro-Wilk on window means: W={sw_stat:.4f}, p={sw_p:.4f}")
        print(f"    -> {'Normally distributed' if sw_p > 0.05 else 'Not normally distributed'} (a=0.05)")

    t_1s, p_1s = ttest_1samp(win_means, 0.0)
    print(f"  One-sample t-test (mu of window means = 0): t={t_1s:.4f}, p={p_1s:.4f}")
    print(f"    -> {'Cannot reject mu=0' if p_1s > 0.05 else 'Reject mu=0'} (a=0.05)")

    rms_raw  = np.sqrt(np.mean(raw_ds[:, :win_len]**2, axis=1))
    rms_proc = np.sqrt(np.mean(pre_norm_x[:, :win_len]**2, axis=1))
    t_rms, p_rms = ttest_rel(rms_raw, rms_proc)
    print(f"  Paired t-test (per-channel RMS, raw vs processed): t={t_rms:.3f}, p={p_rms:.4f}")

    d_rms = rms_raw - rms_proc
    cohen_d = d_rms.mean() / (d_rms.std() + EPS)
    print(f"  Cohen's d (RMS difference): {cohen_d:.4f}")

    try:
        w_stat, w_p = wilcoxon(rms_raw, rms_proc)
        print(f"  Wilcoxon signed-rank (per-channel RMS): W={w_stat:.1f}, p={w_p:.4f}")
    except ValueError as e:
        print(f"  Wilcoxon skipped: {e}")

    var_ratio = pre_norm_x[:, :win_len].var(axis=1) / (raw_ds[:, :win_len].var(axis=1) + EPS)
    print(f"  Variance ratio (processed/raw) — mean: {var_ratio.mean():.4f}, std: {var_ratio.std():.4f}")

    plt.figure(figsize=(10, 4))
    plt.bar(range(62), var_ratio, color='steelblue')
    plt.axhline(y=1.0, color='r', ls='--', lw=0.8, label='Unity')
    plt.xlabel('Channel'); plt.ylabel('Var ratio (proc / raw)')
    plt.title('Per-channel variance ratio (processed / raw)')
    plt.legend()
    savefig('variance_ratio_per_channel')

    ks_stat_norm, ks_p_norm = kstest(X.flatten()[:50000], 'norm')
    print(f"  KS test vs N(0,1): D={ks_stat_norm:.4f}, p={ks_p_norm:.4f}")
    print(f"    -> {'Output ~ N(0,1)' if ks_p_norm > 0.05 else 'Output != N(0,1)'} (a=0.05)")

    # [13] Signal-to-noise ratio (in-band vs out-of-band)
    print("\n[13] SNR: in-band vs out-of-band")
    _in_mask = (_f_psd >= HPF) & (_f_psd <= LPF)
    _snr_raw  = 10 * np.log10(_psd_raw[:, _in_mask].sum(axis=1)  / (_psd_raw[:, ~_in_mask].sum(axis=1) + EPS))
    _snr_proc = 10 * np.log10(_psd_proc[:, _in_mask].sum(axis=1) / (_psd_proc[:, ~_in_mask].sum(axis=1) + EPS))
    print(f"  Raw  SNR (in-band {HPF}-{LPF} Hz): {_snr_raw[0]:.2f} dB")
    print(f"  Proc SNR (in-band {HPF}-{LPF} Hz): {_snr_proc[0]:.2f} dB")
    print(f"  SNR improvement: {_snr_proc[0] - _snr_raw[0]:+.2f} dB")

    snr_gain = _snr_proc - _snr_raw
    print(f"  Mean per-channel SNR gain: {snr_gain.mean():.2f} +/- {snr_gain.std():.2f} dB")

    t_snr, p_snr = ttest_rel(_snr_raw, _snr_proc)
    print(f"  Paired t-test (SNR gain across channels): t={t_snr:.3f}, p={p_snr:.4f}")
    print(f"    -> {'Significant SNR improvement' if p_snr < 0.05 else 'No significant SNR change'} (a=0.05)")

    plt.figure(figsize=(10, 4))
    plt.bar(range(62), snr_gain, color='teal')
    plt.axhline(y=0, color='r', ls='--', lw=0.8)
    plt.xlabel('Channel'); plt.ylabel('SNR gain (dB)')
    plt.title('Per-channel SNR gain after PrC-1')
    savefig('snr_gain_per_channel')

    # Variance profile (first trial)
    first_filt = notch_filter(
        zero_phase_bandpass(first_raw_x, RAW_FS, HPF, LPF), RAW_FS, NOTCH_FREQ, NOTCH_Q)
    ch_var = first_filt.std(axis=1) ** 2
    ch_var_z = (ch_var - np.median(ch_var)) / (np.std(ch_var) + 1e-8)
    colors = ['red' if all_bad_masks[0][i] else 'steelblue' for i in range(62)]

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(62), ch_var, color=colors)
    plt.xlabel("Channel"); plt.ylabel("Variance")
    plt.title("Channel variance (trial 0, red = bad)")
    plt.subplot(1, 2, 2)
    plt.bar(range(62), ch_var_z, color=colors)
    plt.axhline(y=VAR_Z_THRESH, color='r', ls='--', label=f'±{VAR_Z_THRESH}z')
    plt.axhline(y=-VAR_Z_THRESH, color='r', ls='--')
    plt.xlabel("Channel"); plt.ylabel("z-score")
    plt.title("Variance z-score (trial 0)")
    plt.legend()
    savefig("bad_channel_variance_profile")

    # Std heatmap across trials (reuse cached pre-norm data)
    std_matrix = np.array([pn.std(axis=1) for pn in all_pre_norm])
    plt.figure(figsize=(14, 6))
    plt.imshow(std_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Std')
    plt.xlabel("Channel"); plt.ylabel("Trial")
    plt.title("Channel std across trials")
    for ch in np.where(all_bad_union)[0]:
        plt.axvline(x=ch, color='cyan', alpha=0.5, lw=0.8)
    savefig("bad_channel_std_heatmap")

    # [14] Per-channel distortion: interpolated vs non-interpolated
    print("\n[14] Per-channel distortion (interpolated vs non-interpolated)")

    # Aggregate across all trials first (used for both stats and plot)
    all_ch_corr = []
    all_ch_mse  = []
    for t_idx in range(len(eeg_keys)):
        raw_t = mat[eeg_keys[t_idx]]
        raw_t = raw_t if raw_t.shape[0] == 62 else raw_t.T
        raw_t_ds = resample_poly(raw_t, TARGET_FS, RAW_FS, axis=1)
        proc_t = all_pre_norm[t_idx]
        seg_len = min(raw_t_ds.shape[1], proc_t.shape[1])
        corrs = np.array([np.corrcoef(raw_t_ds[ch, :seg_len], proc_t[ch, :seg_len])[0, 1]
                          for ch in range(62)])
        mses  = np.mean((raw_t_ds[:, :seg_len] - proc_t[:, :seg_len])**2, axis=1)
        all_ch_corr.append(corrs)
        all_ch_mse.append(mses)

    all_ch_corr = np.array(all_ch_corr)  # (n_trials, 62)
    all_ch_mse  = np.array(all_ch_mse)

    # Per-channel mean across trials
    mean_corr = all_ch_corr.mean(axis=0)
    mean_mse  = all_ch_mse.mean(axis=0)

    # Split by ever-interpolated (union across all trials) vs never-interpolated
    never_bad = ~all_bad_union
    ever_bad  = all_bad_union
    good_idx  = np.where(never_bad)[0]
    bad_idx   = np.where(ever_bad)[0]

    print(f"  Never interpolated ({never_bad.sum()} ch):")
    print(f"    Mean corr: {mean_corr[good_idx].mean():.6f}   (min {mean_corr[good_idx].min():.6f}, max {mean_corr[good_idx].max():.6f})")
    print(f"    Mean MSE : {mean_mse[good_idx].mean():.4f}     (min {mean_mse[good_idx].min():.4f}, max {mean_mse[good_idx].max():.4f})")
    if len(bad_idx) > 0:
        print(f"  Ever  interpolated ({ever_bad.sum()} ch: {bad_idx.tolist()}):")
        print(f"    Mean corr: {mean_corr[bad_idx].mean():.6f}   (min {mean_corr[bad_idx].min():.6f}, max {mean_corr[bad_idx].max():.6f})")
        print(f"    Mean MSE : {mean_mse[bad_idx].mean():.4f}     (min {mean_mse[bad_idx].min():.4f}, max {mean_mse[bad_idx].max():.4f})")
        from scipy.stats import ttest_ind
        t_corr, p_corr = ttest_ind(mean_corr[good_idx], mean_corr[bad_idx], equal_var=False)
        t_mse, p_mse   = ttest_ind(mean_mse[good_idx], mean_mse[bad_idx], equal_var=False)
        print(f"  Welch t-test (corr): t={t_corr:.3f}, p={p_corr:.4f}")
        print(f"  Welch t-test (MSE) : t={t_mse:.3f}, p={p_mse:.4f}")
    else:
        print("  No interpolated channels across all trials.")

    # Bar plot: per-channel mean correlation & MSE across all trials, colored by union mask
    ch_colors = ['red' if all_bad_union[i] else 'steelblue' for i in range(62)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.bar(range(62), mean_corr, color=ch_colors)
    ax1.set_ylabel('Pearson r'); ax1.set_title('Per-channel correlation (raw vs processed, red = ever interpolated)')
    ax1.set_ylim([min(mean_corr.min() - 0.05, 0.8), 1.01])
    ax2.bar(range(62), mean_mse, color=ch_colors)
    ax2.set_ylabel('MSE'); ax2.set_xlabel('Channel')
    ax2.set_title('Per-channel MSE (raw vs processed, red = ever interpolated)')
    savefig("per_channel_distortion_interp_vs_noninterp")

    # [15] Extra distribution checks (reconstructed vs raw)
    print("\n[15] Extra distribution checks (reconstructed vs raw)")

    # recon_vals and raw_vals already computed in [7]
    # --- 2-sample KS test ---
    ks_d, ks_p = ks_2samp(recon_vals[:50000], raw_vals[:50000])
    print(f"  KS test (recon vs raw): D={ks_d:.6f}, p={ks_p:.4e}")
    print(f"    -> D < 0.1 is good (small distributional shift). D={ks_d:.4f}")

    # --- Wasserstein (Earth Mover's) distance ---
    n_subsample = min(50000, len(recon_vals), len(raw_vals))
    rng = np.random.default_rng(42)
    w_recon = rng.choice(recon_vals, n_subsample, replace=False)
    w_raw   = rng.choice(raw_vals, n_subsample, replace=False)
    w_dist  = wasserstein_distance(w_recon, w_raw)
    raw_range = np.percentile(raw_vals, 97.5) - np.percentile(raw_vals, 2.5)
    print(f"  Wasserstein distance: {w_dist:.4f} µV")
    print(f"    -> {w_dist / (raw_range + EPS) * 100:.2f}% of raw 95% range ({raw_range:.2f} µV)")

    # --- Percentile comparison ---
    pctiles = [1, 5, 25, 50, 75, 95, 99]
    raw_pct  = np.percentile(raw_vals, pctiles)
    rec_pct  = np.percentile(recon_vals, pctiles)
    print("  Percentile comparison (µV):")
    print(f"    {'Pct':>5s}  {'Raw':>10s}  {'Recon':>10s}  {'Diff':>10s}  {'Rel%':>8s}")
    for i, p in enumerate(pctiles):
        diff = rec_pct[i] - raw_pct[i]
        rel  = diff / (abs(raw_pct[i]) + EPS) * 100
        print(f"    {p:5d}  {raw_pct[i]:10.2f}  {rec_pct[i]:10.2f}  {diff:+10.2f}  {rel:+7.2f}%")

    # --- Kurtosis change ---
    kurt_raw  = kurtosis(raw_vals)
    kurt_rec  = kurtosis(recon_vals)
    print(f"  Kurtosis — Raw: {kurt_raw:.4f}, Recon: {kurt_rec:.4f}, Change: {kurt_rec - kurt_raw:+.4f}")
    if kurt_rec < kurt_raw:
        print("    -> Slight decrease: artifact tails reduced (good)")
    elif kurt_rec > kurt_raw + 2:
        print("    -> WARNING: Large increase in kurtosis (new heavy tails introduced)")
    else:
        print("    -> Minimal change (good)")

    # --- Variance ratio (global) ---
    var_recon = np.var(recon_vals)
    var_raw   = np.var(raw_vals)
    vr = var_recon / (var_raw + EPS)
    print(f"  Variance ratio (recon/raw): {vr:.4f}")
    if 0.8 <= vr <= 1.2:
        print("    -> Within 0.8–1.2 range (signal power preserved)")
    else:
        print(f"    -> WARNING: Outside 0.8–1.2 range (energy {'loss' if vr < 0.8 else 'gain'})")

    # --- Median shift ---
    median_raw  = np.median(raw_vals)
    median_rec  = np.median(recon_vals)
    print(f"  Median — Raw: {median_raw:.4f}, Recon: {median_rec:.4f}, Shift: {median_rec - median_raw:+.4f} µV")
    if abs(median_rec - median_raw) > 1.0:
        print("    -> WARNING: Median shifted > 1 µV (possible DC bias introduced)")
    else:
        print("    -> Minimal shift (good)")

    # --- QQ plot (reconstructed vs raw) ---
    n_qq = min(10000, len(recon_vals), len(raw_vals))
    qq_quantiles = np.linspace(0, 100, n_qq)
    qq_raw  = np.percentile(raw_vals, qq_quantiles)
    qq_rec  = np.percentile(recon_vals, qq_quantiles)
    plt.figure(figsize=(6, 6))
    plt.scatter(qq_raw, qq_rec, s=1, alpha=0.5, color='steelblue')
    lim_lo = min(qq_raw.min(), qq_rec.min())
    lim_hi = max(qq_raw.max(), qq_rec.max())
    plt.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', lw=1, label='y = x')
    plt.xlabel('Raw quantiles (µV)'); plt.ylabel('Reconstructed quantiles (µV)')
    plt.title('QQ plot: Reconstructed vs Raw')
    plt.legend(); plt.axis('equal')
    savefig("extra_distribution_checks_qq")

    # --- Summary bar: percentile differences ---
    plt.figure(figsize=(8, 4))
    plt.bar([str(p) for p in pctiles], rec_pct - raw_pct, color='teal')
    plt.axhline(y=0, color='r', ls='--', lw=0.8)
    plt.xlabel('Percentile'); plt.ylabel('Difference (µV)')
    plt.title('Percentile difference: Reconstructed − Raw')
    savefig("extra_distribution_checks_percentile_diff")

    print(f"\nValidation done -> {VALIDATION_DIR}")
