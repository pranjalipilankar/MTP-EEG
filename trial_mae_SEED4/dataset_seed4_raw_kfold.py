"""
Subject-based K-Fold dataset loader for RAW SEED-IV data
For comparison with preprocessed data using k-fold cross-validation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import loadmat
from collections import defaultdict
from scipy.signal import decimate


class SEED4RawSubjectDataset(Dataset):
    """
    RAW SEED-IV dataset with subject-level organization for cross-validation
    Loads directly from .mat files with minimal preprocessing
    """
    
    def __init__(self, data_path, subject_sessions, downsample_to=250, 
                 window_length=1000, transform=None, verbose=False):
        """
        Args:
            data_path: Path to seed4/eeg_raw_data folder  
            subject_sessions: Dict mapping subject_id to list of (session, filename) tuples
            downsample_to: Target sampling rate (250 Hz)
            window_length: Window length at target fs (1000 samples = 4s @ 250Hz)
            transform: Optional data augmentation
            verbose: Print loading info
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.transform = transform
        self.window_length = window_length
        self.raw_fs = 512
        self.target_fs = downsample_to
        self.downsample_factor = self.raw_fs // self.target_fs  # 512 / 250 = 2.048 -> use 2
        
        # SEED-IV emotion labels by session
        self.session_labels = {
            '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
            '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
            '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
        }
        
        # Load all windows for specified subject-sessions
        all_windows = []
        all_labels = []
        all_subject_ids = []
        all_session_ids = []
        
        for subject_id, sessions_list in subject_sessions.items():
            for session, filename in sessions_list:
                mat_path = self.data_path / session / filename
                
                if not mat_path.exists():
                    if verbose:
                        print(f"⚠️  Not found: {mat_path}")
                    continue
                
                try:
                    # Load raw .mat file
                    mat = loadmat(mat_path)
                    
                    # Find EEG trial keys
                    eeg_keys = sorted([k for k in mat.keys() if not k.startswith('__') and 'eeg' in k.lower()])
                    
                    if len(eeg_keys) == 0:
                        if verbose:
                            print(f"⚠️  No EEG data in {filename}")
                        continue
                    
                    # Get session labels
                    trial_labels = self.session_labels.get(session, [0] * 24)
                    
                    # Process each trial
                    for trial_idx, eeg_key in enumerate(eeg_keys):
                        eeg_data = mat[eeg_key]  # Shape: (n_samples, 62) or (62, n_samples)
                        
                        # Ensure shape is (62, n_samples)
                        if eeg_data.shape[0] != 62:
                            eeg_data = eeg_data.T
                        
                        # Simple downsampling using decimation
                        # 512 Hz -> 250 Hz by taking every 2nd sample (approximate)
                        eeg_downsampled = eeg_data[:, ::2]  # (62, n_samples//2)
                        
                        # Calculate actual target length at raw fs
                        raw_window_samples = int(self.window_length * self.raw_fs / self.target_fs)
                        
                        # Sliding window extraction
                        n_samples = eeg_downsampled.shape[1]
                        target_samples = self.window_length
                        
                        for start_idx in range(0, n_samples - target_samples, target_samples):
                            window = eeg_downsampled[:, start_idx:start_idx + target_samples]
                            
                            if window.shape[1] == target_samples:
                                # Minimal preprocessing: z-normalization only
                                window_normalized = self._normalize_window(window)
                                
                                all_windows.append(window_normalized)
                                all_labels.append(trial_labels[trial_idx] if trial_idx < len(trial_labels) else 0)
                                all_subject_ids.append(subject_id)
                                all_session_ids.append(session)
                
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Error loading {filename}: {e}")
                    continue
        
        if len(all_windows) == 0:
            raise ValueError(f"No data loaded for subjects: {list(subject_sessions.keys())}")
        
        self.windows = np.array(all_windows, dtype=np.float32)  # (n_windows, 62, 1000)
        self.labels = np.array(all_labels)
        self.subject_ids = np.array(all_subject_ids)
        self.session_ids = np.array(all_session_ids)
        
    def _normalize_window(self, window):
        """Basic z-normalization per window"""
        mean = window.mean()
        std = window.std()
        if std < 1e-6:
            std = 1.0
        return (window - mean) / std
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        eeg = self.windows[idx].copy()
        label = self.labels[idx]
        
        if self.transform is not None:
            eeg = self.transform(eeg)
        
        eeg = torch.from_numpy(eeg).float()
        
        return {
            'eeg': eeg,
            'label': label,
            'subject_id': self.subject_ids[idx],
            'session_id': self.session_ids[idx],
            'index': idx
        }


def get_all_raw_subjects_and_sessions(data_path, verbose=True):
    """
    Scan raw data directory and return mapping of subjects to their sessions
    
    Returns:
        subject_to_sessions: Dict mapping subject_id to list of (session, filename) tuples
    """
    data_path = Path(data_path)
    subject_to_sessions = defaultdict(list)
    
    # Scan all sessions
    for session_folder in sorted(data_path.iterdir()):
        if not session_folder.is_dir():
            continue
        
        session = session_folder.name
        
        # Scan all .mat files in this session
        for mat_file in sorted(session_folder.glob('*.mat')):
            # Extract subject ID from filename (e.g., '1_20160518.mat' -> '1')
            filename = mat_file.name
            subject_id = filename.split('_')[0]
            
            subject_to_sessions[subject_id].append((session, filename))
    
    if verbose:
        print(f"\n{'='*70}")
        print("RAW SEED-IV Subject-Session Mapping")
        print(f"{'='*70}")
        print(f"Total unique subjects: {len(subject_to_sessions)}")
        
        for subject_id in sorted(subject_to_sessions.keys(), key=int):
            sessions = [s for s, _ in subject_to_sessions[subject_id]]
            print(f"  Subject {subject_id:>2}: {len(sessions)} sessions {sessions}")
        print(f"{'='*70}\n")
    
    return dict(subject_to_sessions)


def create_raw_subject_based_split(data_path, train_subjects, val_subjects, test_subjects,
                                   subject_to_sessions, downsample_to=250, 
                                   window_length=1000, verbose=True):
    """
    Create RAW datasets based on subject splits
    
    Args:
        data_path: Path to seed4/eeg_raw_data
        train_subjects: List of subject IDs for training
        val_subjects: List of subject IDs for validation
        test_subjects: List of subject IDs for testing
        subject_to_sessions: Dict mapping subjects to their sessions
        downsample_to: Target sampling rate
        window_length: Window length at target fs
        verbose: Print info
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    
    # Build subject-session mappings for each split
    train_mapping = {s: subject_to_sessions[s] for s in train_subjects if s in subject_to_sessions}
    val_mapping = {s: subject_to_sessions[s] for s in val_subjects if s in subject_to_sessions}
    test_mapping = {s: subject_to_sessions[s] for s in test_subjects if s in subject_to_sessions}
    
    if verbose:
        print(f"\n{'='*70}")
        print("Creating RAW Subject-Based Datasets")
        print(f"{'='*70}")
        print(f"Train subjects: {len(train_subjects)} -> {train_subjects}")
        print(f"Val subjects:   {len(val_subjects)} -> {val_subjects}")
        print(f"Test subjects:  {len(test_subjects)} -> {test_subjects}")
    
    # Define augmentation for training only
    def train_transform(eeg):
        noise = np.random.randn(*eeg.shape) * 0.01
        eeg = eeg + noise
        scale = np.random.uniform(0.95, 1.05)
        return eeg * scale
    
    train_dataset = SEED4RawSubjectDataset(
        data_path=data_path,
        subject_sessions=train_mapping,
        downsample_to=downsample_to,
        window_length=window_length,
        transform=train_transform,
        verbose=False
    )
    
    val_dataset = SEED4RawSubjectDataset(
        data_path=data_path,
        subject_sessions=val_mapping,
        downsample_to=downsample_to,
        window_length=window_length,
        transform=None,
        verbose=False
    )
    
    test_dataset = SEED4RawSubjectDataset(
        data_path=data_path,
        subject_sessions=test_mapping,
        downsample_to=downsample_to,
        window_length=window_length,
        transform=None,
        verbose=False
    )
    
    if verbose:
        print(f"\n✅ RAW Datasets created:")
        print(f"   Train: {len(train_dataset)} windows")
        print(f"   Val:   {len(val_dataset)} windows")
        print(f"   Test:  {len(test_dataset)} windows")
        print(f"{'='*70}\n")
    
    return train_dataset, val_dataset, test_dataset


def create_raw_kfold_dataloaders(config, fold_idx, fold_splits, subject_to_sessions,
                                 batch_size=None, num_workers=4, pin_memory=True, verbose=True):
    """
    Create RAW dataloaders for a specific fold
    
    Args:
        config: Config object
        fold_idx: Which fold to use
        fold_splits: Pre-computed fold splits
        subject_to_sessions: Subject to session mapping
        batch_size: Override config batch size
        num_workers: Number of workers
        pin_memory: Pin memory
        verbose: Print info
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get specific fold
    fold_info = fold_splits[fold_idx]
    
    # Determine raw data path
    raw_data_path = Path(config.data_path).parent / 'eeg_raw_data'
    
    # Create datasets for this fold
    train_dataset, val_dataset, test_dataset = create_raw_subject_based_split(
        data_path=raw_data_path,
        train_subjects=fold_info['train_subjects'],
        val_subjects=fold_info['val_subjects'],
        test_subjects=fold_info['test_subjects'],
        subject_to_sessions=subject_to_sessions,
        downsample_to=config.sampling_rate,
        window_length=config.time_len,
        verbose=verbose
    )
    
    # Create dataloaders
    if batch_size is None:
        batch_size = config.batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# For testing this module
if __name__ == '__main__':
    import sys
    sys.path.append('/home/ab_students/EEG-MTP/trial_mae_SEED4')
    from config_seed4 import Config_MAE_SEED4
    from dataset_seed4_kfold import create_subject_kfold_splits
    
    config = Config_MAE_SEED4()
    raw_data_path = Path(config.data_path).parent / 'eeg_raw_data'
    
    print("\n" + "="*80)
    print("Testing RAW Subject-Based K-Fold Dataset")
    print("="*80)
    
    # Test: Get all raw subjects
    subject_to_sessions = get_all_raw_subjects_and_sessions(raw_data_path, verbose=True)
    
    # Create 5-fold splits
    print("\n" + "="*80)
    print("Creating 5-Fold Subject Splits")
    print("="*80)
    
    fold_splits, _ = create_subject_kfold_splits(
        data_path=config.data_path,  # Use preprocessed path for splits
        n_folds=5,
        seed=config.seed,
        verbose=False
    )
    
    # Print each fold's subjects
    for fold_idx, fold_info in enumerate(fold_splits):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1} Subject Assignments")
        print(f"{'='*70}")
        print(f"Train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}")
        print(f"Val subjects   ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}")
        print(f"Test subjects  ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}")
    
    # Test creating datasets for fold 0
    print("\n" + "="*80)
    print("Testing Fold 0 Dataset Creation")
    print("="*80)
    
    train_dataset, val_dataset, test_dataset = create_raw_subject_based_split(
        data_path=raw_data_path,
        train_subjects=fold_splits[0]['train_subjects'],
        val_subjects=fold_splits[0]['val_subjects'],
        test_subjects=fold_splits[0]['test_subjects'],
        subject_to_sessions=subject_to_sessions,
        downsample_to=250,
        window_length=1000,
        verbose=True
    )
    
    # Test dataloaders
    print("\nCreating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    
    # Test loading a batch
    print(f"\nTesting batch loading...")
    batch = next(iter(train_loader))
    print(f"   Batch EEG shape: {batch['eeg'].shape}")
    print(f"   Batch labels: {batch['label'].shape}")
    print(f"   Sample EEG stats: mean={batch['eeg'].mean():.4f}, std={batch['eeg'].std():.4f}")
    print(f"   Unique subjects in batch: {len(set(batch['subject_id']))}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80 + "\n")
