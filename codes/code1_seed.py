import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load SEED data (correct)
# -------------------------
dat_path = "/home/ab_students/EEG-MTP/DATA/SEED_processed/train/sub1.npy"

eeg_data = np.load(dat_path)   # shape: (7, 62, 104000)
print("Loaded SEED EEG shape:", eeg_data.shape)

# -------------------------
# Select block + 5 channels
# -------------------------
block = 0   # index 0 → first block
channels = [0, 1, 2, 3, 4]   # choose any 5 channel indices

block_data = eeg_data[block, channels, :]   # shape: (5, 104000)

# -------------------------
# Convert samples → time axis (200 Hz)
# -------------------------
fs = 200
n_samples = block_data.shape[1]
time = np.arange(n_samples) / fs

# -------------------------
# Plot 5 channels
# -------------------------
fig, axes = plt.subplots(len(channels), 1, figsize=(14, 10), sharex=True)

for i, ch in enumerate(channels):
    axes[i].plot(time, block_data[i], linewidth=0.7)
    axes[i].set_ylabel(f"Ch {ch}", rotation=0, labelpad=20)
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("SEED - Subject 1 - Block 1\n5 EEG Channels", fontsize=15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("seed.png")
plt.show()
