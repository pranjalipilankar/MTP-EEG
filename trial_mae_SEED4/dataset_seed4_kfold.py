"""
Subject-based K-Fold Cross Validation for SEED-IV MAE Training

Implements leave-one-subject-out and k-fold subject splits for robust evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict


class SEED4SubjectDataset(Dataset):
    """
    SEED-IV dataset with subject-level organization for cross-validation
    """
    
    def __init__(self, data_path, subject_sessions, transform=None, verbose=False):
        """
        Args:
            data_path: Path to eeg_processed_data folder
            subject_sessions: Dict mapping subject_id to list of (session, subject_folder) tuples
                             e.g., {'1': [('1', '1_20160518'), ('2', '1_20160518')]}
            transform: Optional data augmentation
            verbose: Print loading info
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load all windows for specified subject-sessions
        all_windows = []
        all_labels = []
        all_subject_ids = []
        all_session_ids = []
        
        for subject_id, sessions_list in subject_sessions.items():
            for session, subject_folder_name in sessions_list:
                subject_path = self.data_path / session / subject_folder_name
                
                if not subject_path.exists():
                    if verbose:
                        print(f"⚠️  Not found: {subject_path}")
                    continue
                
                try:
                    X = np.load(subject_path / 'X_prc1.npy')
                    labels = np.load(subject_path / 'labels.npy')
                    
                    all_windows.append(X)
                    all_labels.append(labels)
                    all_subject_ids.extend([subject_id] * len(X))
                    all_session_ids.extend([session] * len(X))
                    
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Error loading {subject_folder_name}: {e}")
                    continue
        
        if len(all_windows) == 0:
            raise ValueError(f"No data loaded for subjects: {list(subject_sessions.keys())}")
        
        self.windows = np.concatenate(all_windows, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.subject_ids = np.array(all_subject_ids)
        self.session_ids = np.array(all_session_ids)
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        eeg = self.windows[idx].copy()
        label = self.labels[idx]
        
        if eeg.dtype != np.float32:
            eeg = eeg.astype(np.float32)
        
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


def get_all_subjects_and_sessions(data_path, verbose=True):
    """
    Scan data directory and return mapping of subjects to their sessions
    
    Returns:
        subject_to_sessions: Dict mapping subject_id to list of (session, folder_name) tuples
        e.g., {'1': [('1', '1_20160518'), ('2', '1_20160518'), ('3', '1_20160518')], ...}
    """
    data_path = Path(data_path)
    subject_to_sessions = defaultdict(list)
    
    # Scan all sessions
    for session_folder in sorted(data_path.iterdir()):
        if not session_folder.is_dir():
            continue
        
        session = session_folder.name
        
        # Scan all subject folders in this session
        for subject_folder in sorted(session_folder.iterdir()):
            if not subject_folder.is_dir():
                continue
            
            # Extract subject ID from folder name (e.g., '1_20160518' -> '1')
            folder_name = subject_folder.name
            subject_id = folder_name.split('_')[0]
            
            # Check if preprocessed data exists
            if (subject_folder / 'X_prc1.npy').exists():
                subject_to_sessions[subject_id].append((session, folder_name))
    
    if verbose:
        print(f"\n{'='*70}")
        print("SEED-IV Subject-Session Mapping")
        print(f"{'='*70}")
        print(f"Total unique subjects: {len(subject_to_sessions)}")
        
        for subject_id in sorted(subject_to_sessions.keys(), key=int):
            sessions = [s for s, _ in subject_to_sessions[subject_id]]
            print(f"  Subject {subject_id:>2}: {len(sessions)} sessions {sessions}")
        print(f"{'='*70}\n")
    
    return dict(subject_to_sessions)


def create_subject_kfold_splits(data_path, n_folds=5, seed=2024, verbose=True):
    """
    Create k-fold cross-validation splits at subject level
    
    Args:
        data_path: Path to preprocessed data
        n_folds: Number of folds
        seed: Random seed
        verbose: Print information
    
    Returns:
        List of (train_subjects, val_subjects, test_subjects) tuples for each fold
    """
    # Get all subjects
    subject_to_sessions = get_all_subjects_and_sessions(data_path, verbose=False)
    all_subjects = sorted(subject_to_sessions.keys(), key=int)
    
    if verbose:
        print(f"\n📊 Creating {n_folds}-Fold Subject-Based Cross-Validation")
        print(f"   Total subjects: {len(all_subjects)}")
        print(f"   Subjects per test fold: ~{len(all_subjects) // n_folds}")
    
    # Create k-fold splits
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_splits = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(all_subjects)):
        # Get subject IDs for this fold
        test_subjects = [all_subjects[i] for i in test_idx]
        train_val_subjects = [all_subjects[i] for i in train_val_idx]
        
        # Further split train_val into train and val (80-20 split of train_val)
        n_val = max(1, len(train_val_subjects) // 5)
        np.random.seed(seed + fold_idx)
        val_indices = np.random.choice(len(train_val_subjects), n_val, replace=False)
        
        val_subjects = [train_val_subjects[i] for i in val_indices]
        train_subjects = [s for s in train_val_subjects if s not in val_subjects]
        
        fold_splits.append({
            'fold': fold_idx,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects
        })
        
        if verbose:
            print(f"\n   Fold {fold_idx + 1}:")
            print(f"     Train: {len(train_subjects)} subjects {train_subjects}")
            print(f"     Val:   {len(val_subjects)} subjects {val_subjects}")
            print(f"     Test:  {len(test_subjects)} subjects {test_subjects}")
    
    if verbose:
        print(f"\n{'='*70}\n")
    
    return fold_splits, subject_to_sessions


def create_subject_based_split(data_path, train_subjects, val_subjects, test_subjects, 
                               subject_to_sessions, verbose=True):
    """
    Create datasets based on subject splits
    
    Args:
        data_path: Path to preprocessed data
        train_subjects: List of subject IDs for training
        val_subjects: List of subject IDs for validation
        test_subjects: List of subject IDs for testing
        subject_to_sessions: Dict mapping subjects to their sessions
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
        print("Creating Subject-Based Datasets")
        print(f"{'='*70}")
        print(f"Train subjects: {len(train_subjects)} -> {train_subjects}")
        print(f"Val subjects:   {len(val_subjects)} -> {val_subjects}")
        print(f"Test subjects:  {len(test_subjects)} -> {test_subjects}")
    
    # Create datasets (with augmentation for training only)
    def train_transform(eeg):
        noise = np.random.randn(*eeg.shape) * 0.01
        eeg = eeg + noise
        scale = np.random.uniform(0.95, 1.05)
        return eeg * scale
    
    train_dataset = SEED4SubjectDataset(
        data_path=data_path,
        subject_sessions=train_mapping,
        transform=train_transform,
        verbose=False
    )
    
    val_dataset = SEED4SubjectDataset(
        data_path=data_path,
        subject_sessions=val_mapping,
        transform=None,
        verbose=False
    )
    
    test_dataset = SEED4SubjectDataset(
        data_path=data_path,
        subject_sessions=test_mapping,
        transform=None,
        verbose=False
    )
    
    if verbose:
        print(f"\n✅ Datasets created:")
        print(f"   Train: {len(train_dataset)} windows")
        print(f"   Val:   {len(val_dataset)} windows")
        print(f"   Test:  {len(test_dataset)} windows")
        print(f"{'='*70}\n")
    
    return train_dataset, val_dataset, test_dataset


def create_kfold_dataloaders(config, fold_idx, batch_size=None, num_workers=4, 
                             pin_memory=True, verbose=True):
    """
    Create dataloaders for a specific fold
    
    Args:
        config: Config object with data_path, seed, etc.
        fold_idx: Which fold to use (0-indexed)
        batch_size: Override config batch size if provided
        num_workers: Number of workers
        pin_memory: Pin memory
        verbose: Print info
    
    Returns:
        train_loader, val_loader, test_loader, fold_info
    """
    
    # Get k-fold splits
    n_folds = getattr(config, 'n_folds', 5)
    fold_splits, subject_to_sessions = create_subject_kfold_splits(
        data_path=config.data_path,
        n_folds=n_folds,
        seed=config.seed,
        verbose=verbose
    )
    
    # Get specific fold
    fold_info = fold_splits[fold_idx]
    
    # Create datasets for this fold
    train_dataset, val_dataset, test_dataset = create_subject_based_split(
        data_path=config.data_path,
        train_subjects=fold_info['train_subjects'],
        val_subjects=fold_info['val_subjects'],
        test_subjects=fold_info['test_subjects'],
        subject_to_sessions=subject_to_sessions,
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
    
    return train_loader, val_loader, test_loader, fold_info


# For testing this module
if __name__ == '__main__':
    import sys
    sys.path.append('/home/ab_students/EEG-MTP/trial_mae_SEED4')
    from config_seed4 import Config_MAE_SEED4
    
    config = Config_MAE_SEED4()
    
    print("\n" + "="*80)
    print("Testing Subject-Based K-Fold Cross-Validation")
    print("="*80)
    
    # Test: Get all subjects
    subject_to_sessions = get_all_subjects_and_sessions(config.data_path, verbose=True)
    
    # Test: Create 5-fold splits
    fold_splits, _ = create_subject_kfold_splits(
        data_path=config.data_path,
        n_folds=5,
        seed=config.seed,
        verbose=True
    )
    
    # Test: Create dataloaders for fold 0
    print("\n" + "="*80)
    print("Testing Fold 0 Dataloaders")
    print("="*80)
    
    train_loader, val_loader, test_loader, fold_info = create_kfold_dataloaders(
        config=config,
        fold_idx=0,
        num_workers=4,
        verbose=True
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print(f"\nTesting batch loading...")
    batch = next(iter(train_loader))
    print(f"   Batch EEG shape: {batch['eeg'].shape}")
    print(f"   Batch labels: {batch['label'].shape}")
    print(f"   Unique subjects in batch: {len(set(batch['subject_id']))}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80 + "\n")
