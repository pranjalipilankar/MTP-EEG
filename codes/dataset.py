import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Path to DEAP data folder
data_dir = "/DATA/EEG-MTP/DEAP"

all_eeg = []
all_labels = []
all_subject_ids = []

filename = "/DATA/EEG-MTP/DEAP/s01.dat"
with open(filename, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    print(data.size())

# Loop through all 32 subjects
for i in range(1, 33):
    filename = os.path.join(data_dir, f"s{i:02d}.dat")
    with open(filename,