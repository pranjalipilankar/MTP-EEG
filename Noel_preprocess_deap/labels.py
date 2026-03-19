import os
import pickle
import numpy as np

# Path to preprocessed DEAP data folder containing the .dat files
data_dir = "/home/ab_students/EEG-MTP/DATA"

# Dictionary to hold the labels for easy lookup later
subject_labels = {}
all_labels_flat = []

print("Extracting labels...")

# Loop through all 32 subjects
for i in range(1, 33):
    subject_id = f"s{i:02d}"  # Creates strings like 's01', 's02'
    filename = os.path.join(data_dir, f"{subject_id}.dat")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract ONLY the labels. 
    # Shape is (40, 4) -> 40 videos, 4 emotional metrics
    # Columns are: [Valence, Arousal, Dominance, Liking]
    labels = data['labels']              

    # Store in dictionary and list
    subject_labels[subject_id] = labels
    all_labels_flat.append(labels)

# Combine all participants into one big array (if you still want the 1280x4 shape)
combined_labels = np.vstack(all_labels_flat)      
print(f"Combined labels shape: {combined_labels.shape} (1280 trials x 4 metrics)")

# Save the labels out to a lightweight .npz file
save_path = "DEAP_labels_only.npz"
np.savez(save_path, **subject_labels)
print(f"\nSuccessfully saved labels to {save_path}")

# --- Example of how to load and use this in your BDF script ---
# loaded_labels = np.load("DEAP_labels_only.npz")
# s01_labels = loaded_labels['s01']
# print("Subject 1 labels shape:", s01_labels.shape)