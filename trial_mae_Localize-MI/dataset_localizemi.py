import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
from scipy import signal

class LocalizeMIPretrainDataset(Dataset):
    """Dataset for MAE pretraining on Localize-MI HD-EEG dataset"""
    
    def __init__(self, data_path, subjects='all', num_channels=256, time_len=2080, transform=None, 
                 orig_fs=8000, target_fs=500):
        """
        Args:
            data_path: Path to Localize-MI derivatives/epochs folder
            subjects: List of subject IDs (e.g., ['sub-01', 'sub-02']) or 'all'
            num_channels: Number of EEG channels (256 for Localize-MI)
            time_len: Fixed time length to use AFTER resampling (at target_fs)
            transform: Optional transform to apply
            orig_fs: Original sampling rate (8000 Hz for Localize-MI)
            target_fs: Target sampling rate for downsampling (e.g., 500 Hz)
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.time_len = time_len
        self.transform = transform
        self.orig_fs = orig_fs
        self.target_fs = target_fs
        self.resample_factor = orig_fs / target_fs
        
        # Determine which subjects to load
        if subjects == 'all':
            subject_dirs = sorted(Path(data_path).glob('sub-*/eeg'))
        else:
            subject_dirs = [Path(data_path) / subj / 'eeg' for subj in subjects]
        
        # Load all epoch files
        print(f"Loading Localize-MI data...")
        self.epochs = []
        self.metadata = []
        
        for subj_dir in subject_dirs:
            if not subj_dir.exists():
                continue
            
            subject_id = subj_dir.parent.name
            epoch_files = sorted(subj_dir.glob('*_epochs.npy'))
            
            for epoch_file in epoch_files:
                run_id = epoch_file.stem  # e.g., 'sub-01_task-seegstim_run-01_epochs'
                
                # Load epochs
                data = np.load(epoch_file)  # Shape: (n_epochs, 256, 2081)
                
                # Process each epoch
                for epoch_idx in range(data.shape[0]):
                    epoch = data[epoch_idx]  # (256, 2081)
                    
                    # Step 1: Resample from orig_fs to target_fs with anti-aliasing
                    if self.resample_factor != 1.0:
                        # resample_poly: up=target_fs, down=orig_fs (simplifies to 1:16 for 500:8000)
                        # This includes anti-aliasing lowpass filter automatically
                        epoch_resampled = signal.resample_poly(epoch, up=self.target_fs, down=self.orig_fs, axis=1)
                    else:
                        epoch_resampled = epoch
                    
                    # Step 2: Truncate or pad to fixed length (at target_fs)
                    if epoch_resampled.shape[1] >= time_len:
                        epoch_final = epoch_resampled[:, :time_len]
                    else:
                        # Pad if shorter
                        pad_width = ((0, 0), (0, time_len - epoch_resampled.shape[1]))
                        epoch_final = np.pad(epoch_resampled, pad_width, mode='edge')
                    
                    self.epochs.append(epoch_final)
                    self.metadata.append({
                        'subject': subject_id,
                        'run': run_id,
                        'epoch_idx': epoch_idx
                    })
        
        self.epochs = np.array(self.epochs)  # (total_epochs, 256, time_len)
        print(f"Loaded {len(self.epochs)} epochs from {len(subject_dirs)} subjects")
        print(f"Resampled from {self.orig_fs}Hz to {self.target_fs}Hz (factor: {self.resample_factor:.1f}x)")
        print(f"Data shape per epoch: {self.epochs[0].shape}")
        print(f"Duration per epoch: {self.time_len/self.target_fs*1000:.1f}ms @ {self.target_fs}Hz")
        print(f"Value range: [{self.epochs.min():.6f}, {self.epochs.max():.6f}]")
    
    def __len__(self):
        return len(self.epochs)
    
    def __getitem__(self, idx):
        # Get EEG epoch: (256, 2080)
        eeg = self.epochs[idx].copy()
        
        # Apply transform FIRST (if using)
        if self.transform is not None:
            eeg = self.transform(eeg)
        
        # Then normalize (captures actual data distribution)
        eeg = self.normalize(eeg)
        
        # Convert to tensor
        eeg = torch.from_numpy(eeg).float()
        
        return {'eeg': eeg}
    
    def normalize(self, x):
        """Normalize each channel independently"""
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test
    
    Args:
        dataset: LocalizeMIPretrainDataset
        train_ratio: Proportion for training (0.7 = 70%)
        val_ratio: Proportion for validation (0.15 = 15%)
        seed: Random seed
    
    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(seed)
    n_total = len(dataset)
    indices = np.random.permutation(n_total)
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def localizemi_transform(x, sparse_rate=0.2):
    """
    Augmentation: randomly zero out some time points
    x: (num_channels, time_len)
    """
    x_aug = x.copy()
    num_timepoints = x.shape[1]
    idx = np.random.choice(num_timepoints, int(num_timepoints * sparse_rate), replace=False)
    x_aug[:, idx] = 0
    return x_aug


def load_saved_split(split_file_path):
    """
    Load train/val/test split indices saved during MAE training
    
    Args:
        split_file_path: Path to 'dataset_split_indices.npz'
        
    Returns:
        train_indices, val_indices, test_indices (numpy arrays)
        
    Example:
        >>> split_file = 'trial_mae_Localize-MI/results_128ch/dataset_split_indices.npz'
        >>> train_idx, val_idx, test_idx = load_saved_split(split_file)
        >>> # Use these indices with Subset() for consistent splits
    """
    data = np.load(split_file_path)
    train_indices = data['train_indices']
    val_indices = data['val_indices']
    test_indices = data['test_indices']
    
    print(f"Loaded saved split from {split_file_path}")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")
    print(f"  Seed:  {data['seed']}")
    
    return train_indices, val_indices, test_indices
