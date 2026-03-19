import os
import mne
import numpy as np

file_path = "/home/ab_students/EEG-MTP/DATA/DEAP-Raw/s01.bdf"

print("\nLoading BDF...\n")

raw = mne.io.read_raw_bdf(
    file_path,
    preload=True,
    stim_channel="Status",
    verbose=True
)

# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------

print("\n==== BASIC INFO ====\n")
print(raw.info)

fs = raw.info['sfreq']
n_channels = raw.info['nchan']
duration = raw.n_times / fs

print(f"\nSampling Frequency: {fs}")
print(f"Channels: {n_channels}")
print(f"Duration: {duration:.2f} sec")

# --------------------------------------------------
# CHANNEL TYPES
# --------------------------------------------------

eeg_picks = mne.pick_types(raw.info, eeg=True)
stim_picks = mne.pick_types(raw.info, stim=True)

print("\nEEG Channels:")
print([raw.ch_names[i] for i in eeg_picks])

print("\nStim Channels:")
print([raw.ch_names[i] for i in stim_picks])

# --------------------------------------------------
# GET DATA
# --------------------------------------------------

data = raw.get_data()
print("\nData shape:", data.shape)  # (channels, time)

# --------------------------------------------------
# EVENTS (VERY IMPORTANT FOR DEAP)
# --------------------------------------------------

print("\nExtracting events...")

events = mne.find_events(raw, stim_channel="Status")

print("Events shape:", events.shape)
print("First 10 events:\n", events[:10])

# --------------------------------------------------
# SAVE FOR PIPELINE
# --------------------------------------------------

out_dir = "./bdf_extracted"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/data.npy", data)
np.save(f"{out_dir}/events.npy", events)

meta = {
    "fs": float(fs),
    "n_channels": int(n_channels),
    "duration_sec": float(duration),
    "channel_names": raw.ch_names
}

import json
with open(f"{out_dir}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved extracted data to:", out_dir)

# --------------------------------------------------
# CHECK IF FILE IS TRUNCATED
# --------------------------------------------------

expected_duration = 60 * 60  # rough expectation for DEAP full session
if duration < 100:
    print("\n🚨 WARNING: FILE LOOKS TRUNCATED!")
    print("Expected long recording but got:", duration, "seconds")