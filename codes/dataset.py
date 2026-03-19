import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Path to DEAP data folder
data_dir = "/home/ab_students/EEG-MTP/DATA/DEAP"

all_eeg = []
all_labels = []

# Loop through all 32 subjects
for i in range(1, 33):
    filename = os.path.join(data_dir, f"s{i:02d}.dat")
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract only the EEG channels
    eeg_data = data['data'][:, :32, :]   # (40, 32, 8064)
    labels = data['labels']              # (40, 4)

    all_eeg.append(eeg_data)
    all_labels.append(labels)

# Combine all participants into one big array
all_eeg = np.vstack(all_eeg)      # shape (32*40 = 1280, 32, 8064)
all_labels = np.vstack(all_labels)  # shape (1280, 4)

print("Combined EEG shape:", all_eeg.shape)
print("Combined labels shape:", all_labels.shape)

# First split: train (70%) vs temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    all_eeg, all_labels, test_size=0.3, random_state=42, shuffle=True
)

# Second split: validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)

np.savez("DEAP_split_dataset.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)

data = np.load("DEAP_split_dataset.npz")
X_train, y_train = data['X_train'], data['y_train']

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)