import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

class DEAPPretrainDataset(Dataset):
    """Dataset for MAE pretraining on DEAP"""
    
    def __init__(self, data_path, split='train', time_len=8064, num_channels=32, transform=None):
        """
        Args:
            data_path: Path to DEAP_split_dataset.npz
            split: 'train', 'val', or 'test'
            time_len: Fixed time length (default 8064 for DEAP)
            num_channels: Number of EEG channels (32 for DEAP)
            transform: Optional transform to apply
        """
        super().__init__()
        
        # Load data
        data = np.load(data_path)
        if split == 'train':
            self.eeg_data = data['X_train']  # (N, 32, 8064)
        elif split == 'val':
            self.eeg_data = data['X_val']
        elif split == 'test':
            self.eeg_data = data['X_test']
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        
        self.time_len = time_len
        self.num_channels = num_channels
        self.transform = transform
        
        print(f"Loaded {split} data: {self.eeg_data.shape}")
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        
        # Apply transform FIRST (if using)
        if self.transform is not None:
            eeg = self.transform(eeg)
        
        # Then normalize (captures actual data distribution)
        eeg = self.normalize(eeg)
        
        return {'eeg': torch.from_numpy(eeg).float()}
    
    def normalize(self, x):
        """Normalize each channel independently"""
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)


def deap_transform(x, sparse_rate=0.2):
    """
    Augmentation: randomly zero out some time points
    x: (num_channels, time_len)
    """
    x_aug = x.copy()
    num_timepoints = x.shape[1]
    idx = np.random.choice(num_timepoints, int(num_timepoints * sparse_rate), replace=False)
    x_aug[:, idx] = 0
    return x_aug