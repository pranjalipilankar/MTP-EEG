import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, sosfiltfilt, butter, resample_poly, iirnotch, welch, hilbert
from scipy.stats import skew, kurtosis
import mne

# ---------------- CONFIG ----------------
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
ENABLE_VALIDATION = True
ENABLE_BAD_CHANNEL_INTERP = True

FLAT_STD_THRESH = 1e-6
VAR_Z_THRESH = 5.0
CORR_THRESH = 0.15
MAX_BAD_CHANNELS = 0.2

BASE_DIR = "/home/ab_students/EEG-MTP/DATA"
RAW_DATA_DIR = f"{BASE_DIR}/DEAP-Raw"
PROCESSED_DATA_DIR = f"{BASE_DIR}/DEAP-Processed"

DEAP_EEG_CHANNELS = [
    'Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5',
    'P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8',
    'CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2',
    'Fz','Cz'
]

# ---------------- FILTERS ----------------
def zero_phase_bandpass(x):
    sos_hp = butter(4, HPF, btype='high', fs=RAW_FS, output='sos')
    x = sosfiltfilt(sos_hp, x, axis=1)

    nyq = RAW_FS / 2
    taps = firwin(101, LPF / nyq)
    return filtfilt(taps, [1.0], x, axis=1)

def notch_filter(x):
    b, a = iirnotch(NOTCH_FREQ / (RAW_FS / 2), NOTCH_Q)
    return filtfilt(b, a, x, axis=1)

# ---------------- WINDOWING ----------------
def global_normalize(w):
    mu = w.mean()
    sigma = w.std() + EPS
    return (w - mu) / sigma, mu, sigma

def soft_clip(w):
    return SOFT_CLIP_VAL * np.tanh(w / SOFT_CLIP_VAL)

def window_signal(x):
    win_len = int(WINDOW_SEC * TARGET_FS)
    windows, stats = [], []

    for start in range(0, x.shape[1] - win_len, win_len):
        w = x[:, start:start + win_len]
        w_norm, mu, sigma = global_normalize(w)

        if ENABLE_SOFT_CLIP:
            w_norm = soft_clip(w_norm)

        windows.append(w_norm)
        stats.append((mu, sigma))

    return np.stack(windows), np.array(stats)

# ---------------- BAD CHANNELS ----------------
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

    if bad.sum() > MAX_BAD_CHANNELS * x.shape[0]:
        return np.zeros_like(bad, dtype=bool)

    return bad

def interpolate_bad_channels(x, bad_mask):
    if not bad_mask.any():
        return x

    info = mne.create_info(DEAP_EEG_CHANNELS, RAW_FS, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')

    raw = mne.io.RawArray(x, info, verbose=False)
    raw.set_montage(montage, on_missing='warn', verbose=False)

    raw.info['bads'] = [DEAP_EEG_CHANNELS[i] for i in np.where(bad_mask)[0]]
    raw.interpolate_bads(verbose=False)

    return raw.get_data()

# ---------------- LOAD ----------------
def load_bdf(file_path):
    raw = mne.io.read_raw_bdf(file_path, preload=True, stim_channel="Status")
    raw.pick_channels(DEAP_EEG_CHANNELS)
    return raw.get_data()

# ---------------- PIPELINE ----------------
def preprocess(x):
    x = zero_phase_bandpass(x)
    x = notch_filter(x)

    bad_mask = detect_bad_channels(x)

    if ENABLE_BAD_CHANNEL_INTERP:
        x = interpolate_bad_channels(x, bad_mask)

    x = resample_poly(x, TARGET_FS, RAW_FS, axis=1)

    windows, stats = window_signal(x)

    return windows, stats, x, bad_mask

# ---------------- PROCESS FILE ----------------
def process_file(file_path):
    print(f"\nProcessing: {file_path}")

    subject = os.path.basename(file_path).replace(".bdf", "")
    out_dir = os.path.join(PROCESSED_DATA_DIR, subject)
    os.makedirs(out_dir, exist_ok=True)

    x = load_bdf(file_path)

    if x.shape[1] < RAW_FS * 10:
        print("Skipping short file")
        return None

    raw_ds = resample_poly(x, TARGET_FS, RAW_FS, axis=1)

    w, s, pre_norm_x, bad_mask = preprocess(x)

    np.savez_compressed(
        os.path.join(out_dir, "data.npz"),
        windows=w,
        stats=s,
        bad_mask=bad_mask,
        pre_norm=pre_norm_x
    )

    return {
        "windows": w,
        "stats": s,
        "pre_norm": pre_norm_x,
        "bad_mask": bad_mask,
        "raw_ds": raw_ds
    }

# ---------------- MAIN ----------------
print("\n=== DEAP PIPELINE ===")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".bdf")]

last = None
all_bad_masks = []

for f in files:
    try:
        res = process_file(os.path.join(RAW_DATA_DIR, f))
        if res:
            last = res
            all_bad_masks.append(res["bad_mask"])
    except Exception as e:
        print("Error:", e)

# ---------------- VALIDATION ----------------
if ENABLE_VALIDATION and last is not None:
    print("\n=== VALIDATION ===")

    X = last["windows"]
    stats = last["stats"]
    pre_norm_x = last["pre_norm"]
    raw_ds = last["raw_ds"]

    print("NaNs:", np.isnan(X).any())
    print("Mean:", X.mean(), "Std:", X.std())

    w_norm = X[0]
    mu, sigma = stats[0]
    recon = w_norm * sigma + mu

    f1, p1 = welch(pre_norm_x[0][:1000], fs=TARGET_FS)
    f2, p2 = welch(recon[0], fs=TARGET_FS)

    plt.figure()
    plt.semilogy(f1, p1, label="Pre")
    plt.semilogy(f2, p2, label="Recon")
    plt.legend()
    plt.title("PSD")
    plt.show()

    phi1 = np.angle(hilbert(pre_norm_x[0][:1000]))
    phi2 = np.angle(hilbert(recon[0]))

    delta = np.angle(np.exp(1j*(phi1 - phi2)))
    print("Phase error:", np.mean(np.abs(delta)))

    vals = X.flatten()
    print("Skew:", skew(vals), "Kurt:", kurtosis(vals))

    plt.hist(vals, bins=100)
    plt.title("Distribution")
    plt.show()

    union = np.zeros(len(DEAP_EEG_CHANNELS), dtype=bool)
    for m in all_bad_masks:
        union |= m

    print("Total bad channels:", union.sum())