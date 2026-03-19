import numpy as np
import matplotlib.pyplot as plt

# Load EEG data
data = np.load('/home/ab_students/EEG-MTP/DATA/SEED_processed/train/sub10.npy')

block = 0
fs = 200  # sampling frequency
time = np.arange(data.shape[2]) / fs

# Choose a smaller offset (in µV range)
offset = 0.0003  

plt.figure(figsize=(15, 10))

# Plot only first 10 channels for clarity
num_channels_to_plot = 62

for ch in range(num_channels_to_plot):
    plt.plot(time, data[block, ch, :] + ch * offset, linewidth=0.5)

plt.title("EEG — Block 1, 62 Channels (offset for visibility)")
plt.xlabel("Time (s)")
plt.ylabel("EEG amplitude + offset (µV)")
plt.tight_layout()

# ✅ Display and save
plt.savefig("EEG_block1_62_channels.png", dpi=300, bbox_inches='tight')
plt.show()
