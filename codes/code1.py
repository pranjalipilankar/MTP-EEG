import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to participant file
dat_path = "/home/ab_students/EEG-MTP/DATA/DEAP/s01.dat"

# Load using pickle
with open(dat_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Extract data and labels
eeg_data = data['data'][:, :5, :]   # <-- Keep only first 32 EEG channels
labels = data['labels'][:5, :]

print("EEG data shape:", eeg_data.shape)  # Expect (40, 32, 8064)
print("Labels shape:", labels.shape)

# Select a trial to visualize (e.g., first trial)
trial = 0
trial_data = eeg_data[trial]  # shape = (32, 8064)
n_channels = trial_data.shape[0]
n_samples = trial_data.shape[1]

# Plot first 1000 samples of each EEG channel
n_samples_to_plot = 8064
fig, axes = plt.subplots(n_channels, 1, figsize=(12, 1.5 * n_channels), sharex=True)

for i in range(n_channels):
    axes[i].plot(trial_data[i, :n_samples_to_plot])
    axes[i].set_ylabel(f"Ch {i+1}", rotation=0, labelpad=20)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Samples (128 Hz sampling rate)")
fig.suptitle(f"DEAP s01 – Trial {trial+1}\n32 EEG Channels (First {n_samples_to_plot} Samples)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("deap_s01_trial1_32EEG.png", dpi=300, bbox_inches="tight")
plt.show()

n_trials, n_channels, n_samples = eeg_data.shape
print(f"EEG data shape: {eeg_data.shape}")

# Compute correlation matrices for all trials
corr_matrices = np.zeros((n_trials, n_channels, n_channels))

for i in range(n_trials):
    corr_matrices[i] = np.corrcoef(eeg_data[i])

# Compute mean correlation matrix
mean_corr = np.mean(corr_matrices, axis=0)

# Channel names (DEAP standard 32 EEG channels)
channel_names = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1',
    'Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2',
    'P4','P8','PO4','O2'
]

# Plot average correlation matrix
plt.figure(figsize=(8, 7))
im = plt.imshow(mean_corr, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im, label="Pearson Correlation")
plt.title("DEAP s01 – Mean EEG Channel Correlation Matrix (Across 40 Trials)")
plt.xticks(range(32), channel_names, rotation=90)
plt.yticks(range(32), channel_names)
plt.tight_layout()
plt.savefig("deap_s01_mean_corr_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional: Print summary stats
print("Mean correlation (excluding diagonal):", np.mean(mean_corr[np.triu_indices(n_channels, k=1)]))