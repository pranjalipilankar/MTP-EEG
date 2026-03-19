#!/usr/bin/env python3
"""
Prepare 31-channel HR dataset from RAW SEED-IV data for comparison
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import decimate
from tqdm import tqdm


def load_and_process_raw_data(data_root):
    """Load raw SEED-IV data and process to 31 channels"""
    data_root = Path(data_root)
    
    # SEED-IV emotion labels by session
    session_labels = {
        '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
    }
    
    all_data = []
    all_labels = []
    all_subject_ids = []
    
    # Load from all 3 sessions
    for session in ['1', '2', '3']:
        session_path = data_root / session
        if not session_path.exists():
            print(f"⚠️  Session {session} not found")
            continue
        
        # Get all .mat files
        mat_files = sorted(session_path.glob('*.mat'))
        
        for mat_file in tqdm(mat_files, desc=f"Session {session}"):
            # Extract subject ID from filename (e.g., '1_20160518.mat' -> '1')
            subject_id = mat_file.stem.split('_')[0]
            
            try:
                # Load .mat file
                mat = loadmat(mat_file)
                
                # Find EEG trial keys
                eeg_keys = sorted([k for k in mat.keys() if not k.startswith('__')])
                
                trial_labels = session_labels.get(session, [0] * 24)
                
                # Process each trial
                for trial_idx, eeg_key in enumerate(eeg_keys):
                    eeg_data = mat[eeg_key]  # (n_samples, 62) or (62, n_samples)
                    
                    # Ensure shape is (62, n_samples)
                    if eeg_data.shape[0] != 62:
                        eeg_data = eeg_data.T
                    
                    # Downsample from 1000 Hz to 250 Hz
                    eeg_downsampled = decimate(eeg_data, q=4, axis=1)  # 1000/4 = 250 Hz
                    
                    # Split into 4-second windows (1000 samples at 250 Hz)
                    n_samples = eeg_downsampled.shape[1]
                    window_size = 1000
                    
                    n_windows = n_samples // window_size
                    
                    for w in range(n_windows):
                        start_idx = w * window_size
                        end_idx = start_idx + window_size
                        
                        window = eeg_downsampled[:, start_idx:end_idx]  # (62, 1000)
                        
                        all_data.append(window)
                        all_labels.append(trial_labels[trial_idx] if trial_idx < len(trial_labels) else 0)
                        all_subject_ids.append(subject_id)
                
                print(f"   Loaded {mat_file.name}: {len(eeg_keys)} trials")
                
            except Exception as e:
                print(f"⚠️  Error loading {mat_file.name}: {e}")
                continue
    
    # Convert to arrays
    all_data = np.stack(all_data, axis=0)  # (N, 62, 1000)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)
    
    print(f"\n📊 Total data shape: {all_data.shape}")
    print(f"   Labels shape: {all_labels.shape}")
    print(f"   Unique subjects: {len(np.unique(all_subject_ids))}")
    
    return all_data, all_labels, all_subject_ids


def create_multiresolution(sr_data):
    """Create multi-resolution versions (same as preprocessed)"""
    N, C, T = sr_data.shape
    
    # SR: Keep as is
    sr = sr_data.astype(np.float32)
    
    # HR: Downsample to 31 channels (evenly distributed)
    hr_indices = np.linspace(0, C-1, 31, dtype=int)
    hr = sr_data[:, hr_indices, :].astype(np.float32)
    
    # LR: Downsample to 16 channels
    lr_indices = np.linspace(0, C-1, 16, dtype=int)
    lr = sr_data[:, lr_indices, :].astype(np.float32)
    
    print(f"\n✅ Multi-resolution created:")
    print(f"   LR: {lr.shape} (16 channels)")
    print(f"   HR: {hr.shape} (31 channels, evenly distributed)")
    print(f"   SR: {sr.shape} (62 channels)")
    
    return lr, hr, sr


def main():
    print("🚀 SEED-IV RAW Dataset Preparation")
    
    # Paths
    raw_data_root = '/home/ab_students/EEG-MTP/DATA/seed4/eeg_raw_data'
    output_path = '/home/ab_students/EEG-MTP/DATA/seed4/raw_data.npz'
    
    # Load raw data
    print("\n1️⃣  Loading RAW SEED-IV data...")
    sr_data, labels, subject_ids = load_and_process_raw_data(raw_data_root)
    
    # Create multi-resolution
    print("\n2️⃣  Creating multi-resolution versions...")
    lr, hr, sr = create_multiresolution(sr_data)
    
    # Save
    print(f"\n3️⃣  Saving to {output_path}...")
    print("   (This may take a few minutes...)")
    np.savez(
        output_path,
        LR=lr,
        HR=hr,
        SR=sr,
        labels=labels,
        subject_ids=subject_ids
    )
    
    # Verify
    print("\n✅ Saved! Verifying...")
    data = np.load(output_path)
    print(f"   Keys: {list(data.keys())}")
    print(f"   LR: {data['LR'].shape}")
    print(f"   HR: {data['HR'].shape}")
    print(f"   SR: {data['SR'].shape}")
    print(f"   Subject IDs: {len(data['subject_ids'])}")
    print(f"   Unique subjects: {len(np.unique(data['subject_ids']))}")
    
    print("\n🎉 RAW dataset preparation complete!")


if __name__ == '__main__':
    main()
