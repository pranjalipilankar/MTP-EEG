import os
import pickle
import numpy as np

# --- Updated Path to your user directory ---
data_dir = "/DATA/EEG-MTP/DEAP"

subject_labels = {}
all_labels_flat = []

print(f"Extracting and Printing labels from: {data_dir}")
print("-" * 50)

# Loop through all 32 subjects
for i in range(1, 33):
    subject_id = f"s{i:02d}"  
    filename = os.path.join(data_dir, f"{subject_id}.dat")
    
    if not os.path.exists(filename):
        # Silent skip if file missing, or print warning
        continue

    with open(filename, 'rb') as f:
        # DEAP .dat files use latin1 encoding
        data = pickle.load(f, encoding='latin1')

    # Extract labels (40, 4)
    labels = data['labels']              
    
    # --- PRINT AS ARRAY ---
    print(f"\n{subject_id.upper()} Labels Array (Trial x Emotion):")
    # np.array2string makes it clean, or just print(labels)
    print(labels) 

    subject_labels[subject_id] = labels
    all_labels_flat.append(labels)

if all_labels_flat:
    # Combine into one big array (1280, 4)
    combined_labels = np.vstack(all_labels_flat)      
    
    print(f"\n{'='*60}")
    print("FINAL COMBINED LABELS ARRAY (All 32 Subjects):")
    print("-" * 60)
    print(combined_labels)
    
    print(f"\nShape: {combined_labels.shape}")

    # Save to a lightweight file for your BDF pipeline
    save_path = "DEAP_labels_only.npz"
    np.savez(save_path, **subject_labels)
    print(f"\nSuccessfully saved labels to {save_path}")
else:
    print(f"\nNo labels found. Check if .dat files are in {data_dir}")