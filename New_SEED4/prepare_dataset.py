#!/usr/bin/env python3
"""
Prepare SEED-IV Dataset for STAD Training
Creates multi-resolution EEG data (LR/HR/SR) and splits into train/val/test
"""
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def load_seed4_data(data_root):
    """Load all SEED-IV processed data"""
    data_root = Path(data_root)
    
    all_data = []
    all_labels = []
    
    # Load from all 3 sessions
    for session in ['1', '2', '3']:
        session_path = data_root / session
        if not session_path.exists():
            print(f"⚠️  Session {session} not found")
            continue
        
        # Get all subject folders in this session
        subject_folders = sorted([f for f in session_path.iterdir() if f.is_dir()])
        
        for folder in tqdm(subject_folders, desc=f"Session {session}"):
            x_file = folder / 'X_prc1.npy'
            label_file = folder / 'labels.npy'
            
            if x_file.exists() and label_file.exists():
                x_data = np.load(x_file)  # (num_windows, 62, 1000)
                labels = np.load(label_file)  # (num_windows,)
                
                all_data.append(x_data)
                all_labels.append(labels)
                
                print(f"   Loaded {folder.name}: {x_data.shape}")
    
    # Concatenate all data
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\n📊 Total data shape: {all_data.shape}")
    print(f"   Labels shape: {all_labels.shape}")
    
    return all_data, all_labels

def create_multiresolution(sr_data):
    """
    Create multi-resolution versions
    SR: 62 channels (full resolution)
    HR: 31 channels (MAE training resolution - evenly distributed)
    LR: 16 channels (low-resolution input)
    """
    N, C, T = sr_data.shape
    
    # SR: Keep as is
    sr = sr_data.astype(np.float32)
    
    # HR: Downsample to 31 channels (evenly distributed across 62)
    hr_indices = np.linspace(0, C-1, 31, dtype=int)
    hr = sr_data[:, hr_indices, :].astype(np.float32)
    
    # LR: Downsample to 16 channels
    lr_indices = np.linspace(0, C-1, 16, dtype=int)
    lr = sr_data[:, lr_indices, :].astype(np.float32)
    
    print(f"✅ Multi-resolution created:")
    print(f"   LR: {lr.shape} (16 channels)")
    print(f"   HR: {hr.shape} (31 channels, evenly distributed)")
    print(f"   SR: {sr.shape} (62 channels)")
    
    return lr, hr, sr

def split_dataset(num_samples, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/val/test
    """
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * num_samples)
    val_end = int((train_ratio + val_ratio) * num_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_indices)} ({len(train_indices)/num_samples*100:.1f}%)")
    print(f"   Val: {len(val_indices)} ({len(val_indices)/num_samples*100:.1f}%)")
    print(f"   Test: {len(test_indices)} ({len(test_indices)/num_samples*100:.1f}%)")
    
    return train_indices, val_indices, test_indices

def main():
    """Main preprocessing pipeline"""
    print("🚀 SEED-IV Dataset Preparation for STAD")
    
    # Paths
    data_root = '/home/ab_students/EEG-MTP/DATA/seed4/eeg_processed_data'
    output_path = '/home/ab_students/EEG-MTP/DATA/seed4/preprocessed_data.npz'
    
    # Load data
    print("\n1️⃣  Loading SEED-IV data...")
    sr_data, labels = load_seed4_data(data_root)
    
    # Create multi-resolution
    print("\n2️⃣  Creating multi-resolution versions...")
    lr, hr, sr = create_multiresolution(sr_data)
    
    # Split dataset
    print("\n3️⃣  Splitting dataset...")
    train_idx, val_idx, test_idx = split_dataset(len(sr_data))
    
    # Save
    print(f"\n4️⃣  Saving to {output_path}...")
    print("   (This may take a few minutes...)")
    np.savez(  # Use uncompressed for speed
        output_path,
        LR=lr,
        HR=hr,
        SR=sr,
        labels=labels,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx
    )
    
    # Verify
    print("\n✅ Saved! Verifying...")
    data = np.load(output_path)
    print(f"   Keys: {list(data.keys())}")
    print(f"   LR: {data['LR'].shape}")
    print(f"   HR: {data['HR'].shape}")
    print(f"   SR: {data['SR'].shape}")
    print(f"   Train: {len(data['train_indices'])}")
    print(f"   Val: {len(data['val_indices'])}")
    print(f"   Test: {len(data['test_indices'])}")
    
    print("\n🎉 Dataset preparation complete!")

if __name__ == '__main__':
    main()
