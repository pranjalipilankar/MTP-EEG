import os, json, time
import numpy as np
import mne
from scipy.signal import firwin, filtfilt, sosfiltfilt, butter, resample_poly, welch, hilbert, iirnotch
from scipy.stats import skew, kurtosis, ks_2samp, ttest_rel, ttest_1samp, wilcoxon, shapiro, kstest, wasserstein_distance
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
SPIKE_SAFE_MODE = False
ENABLE_BAD_CHANNEL_INTERP = True
ENABLE_VALIDATION = False

FLAT_STD_THRESH = 1e-6
VAR_Z_THRESH = 5.0
CORR_THRESH = 0.15
MAX_BAD_CHANNELS = 0.2

# -------- DEAP CHANNEL NAMES (32ch EEG) --------
DEAP_EEG_CHANNELS = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 
    'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 
    'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 
    'Fz', 'Cz'
]
N_CHANNELS = len(DEAP_EEG_CHANNELS)

# Paths
BASE_DIR = "/DATA/EEG-MTP/"
RAW_DATA_DIR = f"{BASE_DIR}/DEAP-RAW"
PROCESSED_DATA_DIR = f"{BASE_DIR}/DEAP-PrC_final/test"
LABEL_PATH = "/home/ab_students/EEG-MTP/Noel_preprocess_deap/DEAP_labels_only.npz"

print(f"Loading labels from {LABEL_PATH}...")
ALL_LABELS = np.load(LABEL_PATH)

# --- Signal operations ---

def zero_phase_bandpass(x, fs, l_freq, h_freq):
    sos_hp = butter(N=4, Wn=l_freq, btype='high', fs=fs, output='sos')
    x = sosfiltfilt(sos_hp, x, axis=1)
    nyq = fs / 2
    trans_bw = min(5.0, (nyq - h_freq) * 0.5)
    n = int(np.ceil(3.3 * fs / trans_bw))
    max_n = (x.shape[1] - 1) // 3
    n = min(n, max_n)
    n = n if n % 2 == 1 else n - 1
    taps = firwin(numtaps=n, cutoff=h_freq / nyq)
    return filtfilt(taps, [1.0], x, axis=1)

def notch_filter(x, fs, freq, q):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, x, axis=1)

def global_normalize(w):
    mu = w.mean(); sigma = w.std() + EPS
    return (w - mu) / sigma, mu, sigma

def soft_clip(w, clip_val):
    return clip_val * np.tanh(w / clip_val)

def window_signal(x):
    win_len = int(WINDOW_SEC * TARGET_FS)
    windows, stats = [], []
    for start in range(0, x.shape[1] - win_len + 1, win_len):
        w = x[:, start:start + win_len]
        w_norm, mu, sigma = global_normalize(w)
        if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
            w_norm = soft_clip(w_norm, SOFT_CLIP_VAL)
        windows.append(w_norm)
        stats.append((mu, sigma))
    return np.stack(windows), np.array(stats)

# --- Bad channel detection & interpolation ---

def detect_bad_channels(x):
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
    return bad if bad.sum() <= MAX_BAD_CHANNELS * x.shape[0] else np.zeros_like(bad, dtype=bool)

def interpolate_bad_channels(x, bad_mask, sfreq=RAW_FS):
    if not bad_mask.any(): return x
    info = mne.create_info(ch_names=DEAP_EEG_CHANNELS, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    raw = mne.io.RawArray(x, info, verbose=False)
    raw.set_montage(montage, on_missing='warn', verbose=False)
    raw.info['bads'] = [DEAP_EEG_CHANNELS[i] for i in np.where(bad_mask)[0]]
    raw.interpolate_bads(mode='accurate', verbose=False)
    return raw.get_data()

def prc1_preprocess_trial_with_prenorm(x):
    x = zero_phase_bandpass(x, RAW_FS, HPF, LPF)
    x = notch_filter(x, RAW_FS, NOTCH_FREQ, NOTCH_Q)
    bad_mask = detect_bad_channels(x) if ENABLE_BAD_CHANNEL_INTERP else np.zeros(x.shape[0], dtype=bool)
    if ENABLE_BAD_CHANNEL_INTERP: x = interpolate_bad_channels(x, bad_mask)
    x = resample_poly(x, TARGET_FS, RAW_FS, axis=1)
    pre_norm_x = x.copy()
    windows, stats = window_signal(x)
    return windows, stats, pre_norm_x, bad_mask

# --- Diagnostic plots ---

def save_stage_plot(signal, title, ylabel, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title); plt.xlabel("Samples"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def save_before_after_overlay(raw_sig, proc_sig, title, filename, raw_label="Before", proc_label="After"):
    """Save an overlay figure for direct before/after comparison."""
    n = min(len(raw_sig), len(proc_sig))
    x = np.arange(n)
    plt.figure(figsize=(12, 4))
    plt.plot(x, raw_sig[:n], linewidth=0.8, alpha=0.8, label=raw_label)
    plt.plot(x, proc_sig[:n], linewidth=0.8, alpha=0.8, label=proc_label)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def inspect_single_window(x_raw_trial, out_dir, channel_idx=0):
    os.makedirs(out_dir, exist_ok=True)
    raw_win_len = int(WINDOW_SEC * RAW_FS)
    save_stage_plot(x_raw_trial[channel_idx, :raw_win_len], "Raw EEG", "µV", f"{out_dir}/01_raw.png")
    
    x = zero_phase_bandpass(x_raw_trial, RAW_FS, HPF, LPF)
    x = notch_filter(x, RAW_FS, NOTCH_FREQ, NOTCH_Q)
    if ENABLE_BAD_CHANNEL_INTERP: x = interpolate_bad_channels(x, detect_bad_channels(x))
    x = resample_poly(x, TARGET_FS, RAW_FS, axis=1)
    
    proc_win_len = int(WINDOW_SEC * TARGET_FS)
    save_stage_plot(x[channel_idx, :proc_win_len], "Post-Filter", "µV", f"{out_dir}/02_filt.png")

    # Plot direct before/after overlay at same sampling rate (250 Hz)
    raw_ds = resample_poly(x_raw_trial, TARGET_FS, RAW_FS, axis=1)
    save_before_after_overlay(
        raw_sig=raw_ds[channel_idx, :proc_win_len],
        proc_sig=x[channel_idx, :proc_win_len],
        title="Before vs After Preprocessing (Filtered/Interpolated)",
        filename=f"{out_dir}/02b_before_after_overlay.png",
        raw_label="Before (Raw downsampled)",
        proc_label="After preprocessing",
    )
    
    w = x[:, :proc_win_len]
    mu, sigma = w.mean(), w.std() + EPS
    w_norm = (w - mu) / sigma
    save_stage_plot(w_norm[channel_idx], "Z-Normalized", "z", f"{out_dir}/03_norm.png")

    save_before_after_overlay(
        raw_sig=x[channel_idx, :proc_win_len],
        proc_sig=w_norm[channel_idx],
        title="After Filtering vs After Normalization",
        filename=f"{out_dir}/03b_filtered_vs_normalized_overlay.png",
        raw_label="Filtered",
        proc_label="Normalized",
    )

# --- Main Processing ---

def process_bdf_file(bdf_path):
    subject_id = os.path.splitext(os.path.basename(bdf_path))[0]
    out_dir = f"{PROCESSED_DATA_DIR}/{subject_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nProcessing {subject_id}...")
    trial_emotion_labels = ALL_LABELS[subject_id] 

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    raw.pick_channels(DEAP_EEG_CHANNELS)
    full_data = raw.get_data()
    samples_per_trial = int(60 * RAW_FS)
    n_trials = min(40, full_data.shape[1] // samples_per_trial)
    
    all_windows, all_stats, all_pre_norm, all_bad_masks, all_labels = [], [], [], [], []
    
    for i in range(n_trials):
        start = i * samples_per_trial
        x = full_data[:, start:start + samples_per_trial]
        w, s, pre_norm, bad_m = prc1_preprocess_trial_with_prenorm(x)
        all_windows.append(w); all_stats.append(s); all_pre_norm.append(pre_norm)
        all_bad_masks.append(bad_m)
        all_labels.append(np.tile(trial_emotion_labels[i], (w.shape[0], 1)))
        
    X = np.concatenate(all_windows, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    stats = np.concatenate(all_stats, axis=0)
    
    np.save(f"{out_dir}/X_prc1.npy", X)
    np.save(f"{out_dir}/Y_labels.npy", Y)
    np.save(f"{out_dir}/X_stats.npy", stats)

    # Reversal for validation
    X_rev = X.copy()
    if ENABLE_SOFT_CLIP and not SPIKE_SAFE_MODE:
        limit = SOFT_CLIP_VAL * (1 - 1e-7)
        X_rev = SOFT_CLIP_VAL * np.arctanh(np.clip(X_rev, -limit, limit) / SOFT_CLIP_VAL)
    for j in range(X_rev.shape[0]):
        X_rev[j] = X_rev[j] * stats[j, 1] + stats[j, 0]
    np.save(f"{out_dir}/X_prc1_reversed.npy", X_rev)

    inspect_single_window(full_data[:, :samples_per_trial], f"{out_dir}/inspection")
    
    return out_dir, X, Y, stats, X_rev, all_pre_norm, all_bad_masks, full_data, samples_per_trial

# --- Execution ---
bdf_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.bdf')])
for f_name in bdf_files:
    bdf_full_path = os.path.join(RAW_DATA_DIR, f_name)
    out_dir, X, Y, stats, X_rev, all_pre_norm, all_bad_masks, raw_full, samples_trial = process_bdf_file(bdf_full_path)
    
    # ---------------- VALIDATION SUITE (SEED-IV Style) ----------------
    if ENABLE_VALIDATION:
        VALIDATION_DIR = f"{out_dir}/validation"
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        def savefig(name): plt.tight_layout(); plt.savefig(f"{VALIDATION_DIR}/val_{name}.png", dpi=150); plt.close()

        print("\n" + "="*40 + "\nRUNNING VALIDATION SUITE\n" + "="*40)
        win_len = int(WINDOW_SEC * TARGET_FS)
        pre_norm_x = all_pre_norm[0]
        raw_ds = resample_poly(raw_full[:, :samples_trial], TARGET_FS, RAW_FS, axis=1)

        # [1] Phase preservation
        phi_pre = np.angle(hilbert(pre_norm_x[0, :win_len]))
        # FIX: Select channel 0 of window 0
        phi_rev = np.angle(hilbert(X_rev[0, 0, :]))
        delta = np.angle(np.exp(1j * (phi_pre - phi_rev)))
        plt.figure(); plt.plot(delta); plt.title("Phase Difference (Pre vs Reversed)"); savefig("phase")

        # [2] PSD comparison
        f_b, p_b = welch(pre_norm_x[0, :win_len], fs=TARGET_FS)
        # FIX: Select channel 0 of window 0
        f_a, p_a = welch(X_rev[0, 0, :], fs=TARGET_FS)
        plt.figure(); plt.semilogy(f_b, p_b, label="Pre-norm"); plt.semilogy(f_a, p_a, '--', label="Reversed")
        plt.legend(); plt.title("PSD Comparison"); savefig("psd")

        # [3] Band Power
        BANDS = {'Alpha': (8, 13), 'Beta': (13, 30)}
        for b, (fmin, fmax) in BANDS.items():
            mask = (f_b >= fmin) & (f_b <= fmax)
            print(f"  {b} Power Ratio: {np.sum(p_a[mask]) / np.sum(p_b[mask]):.4f}")

        # [4] Distribution symmetry
        print(f"  Skewness: {skew(X.flatten()):.4f}, Kurtosis: {kurtosis(X.flatten()):.4f}")
        plt.figure(); plt.hist(X.flatten()[:10000], bins=100); plt.title("Global Distribution"); savefig("dist")

        # [5] Bad channel summary
        all_bad_union = np.zeros(N_CHANNELS, dtype=bool)
        for m in all_bad_masks: all_bad_union |= m
        print(f"  Unique Bad Channels identified: {np.where(all_bad_union)[0].tolist()}")

        # [6] Statistical SNR Gain
        _f, _psd_raw = welch(raw_ds[0, :win_len], fs=TARGET_FS)
        in_mask = (_f >= HPF) & (_f <= LPF)
        # FIX: Explicit numpy sums
        snr_raw = 10 * np.log10(np.sum(_psd_raw[in_mask]) / (np.sum(_psd_raw[~in_mask]) + EPS))
        snr_proc = 10 * np.log10(np.sum(p_b[in_mask]) / (np.sum(p_b[~in_mask]) + EPS))
        print(f"  SNR Improvement: {snr_proc - snr_raw:+.2f} dB")

        print(f"Validation finished. Results in {VALIDATION_DIR}")