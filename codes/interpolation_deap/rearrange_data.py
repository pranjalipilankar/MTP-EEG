import numpy as np
import os

# =====================================================
# 1. Load DATA
# =====================================================
split_file = "/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz"
split_data = np.load(split_file)

X_train = split_data["X_train"]  # original EEG for training
y_train_raw = split_data["y_train"][:, 2]  # valence
y_train = (y_train_raw >= 5).astype(int)

print("Loaded X_train:", X_train.shape)

# Load reconstructed test EEG
recon_file = "results_test_harmonic_2/reconstructions_test_denoise.npz"
recon_data = np.load(recon_file)

X_test = recon_data["recons"]            # reconstructed EEG
y_test_raw = split_data["y_test"][:, 2]  # original DEAP valence
y_test = (y_test_raw >= 5).astype(int)

print("Loaded X_test:", X_test.shape)

# =====================================================
# 2. SEGMENT EEG INTO FIXED WINDOWS (128 samples)
# =====================================================
def segment_trials(X, y, window=128, step=128):
    X_out, y_out = [], []
    for trial, label in zip(X, y):
        T = trial.shape[-1]
        for start in range(0, T - window + 1, step):
            segment = trial[:, start:start+window]  # (32,128)
            X_out.append(segment)
            y_out.append(label)
    return np.array(X_out), np.array(y_out)

print("Segmenting...")
X_train_seg, y_train_seg = segment_trials(X_train, y_train)
X_test_seg, y_test_seg = segment_trials(X_test, y_test)

print("After segmentation:")
print("X_train_seg:", X_train_seg.shape)
print("X_test_seg:", X_test_seg.shape)

# =====================================================
# 3. NORMALIZE per channel (Z-score)
# =====================================================
# Compute normalization stats from training only
mean = X_train_seg.mean(axis=(0, 2), keepdims=True)
std = X_train_seg.std(axis=(0, 2), keepdims=True) + 1e-8

X_train_norm = (X_train_seg - mean) / std
X_test_norm  = (X_test_seg  - mean) / std

print("Normalized shapes:")
print(X_train_norm.shape, X_test_norm.shape)

# =====================================================
# 4. SAVE FOR RBTRANSFORMER
# =====================================================
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

np.save("data/train/X_train.npy", X_train_norm)
np.save("data/train/y_train.npy", y_train_seg)

np.save("data/test/X_test.npy", X_test_norm)
np.save("data/test/y_test.npy", y_test_seg)

print("\nSaved RBTransformer-compatible dataset:")
print("  data/train/X_train.npy")
print("  data/train/y_train.npy")
print("  data/test/X_test.npy")
print("  data/test/y_test.npy")
