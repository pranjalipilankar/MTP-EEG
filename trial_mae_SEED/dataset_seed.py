import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob

class SEEDPretrainDataset(Dataset):
    """Dataset for MAE pretraining on SEED dataset"""
    
    def __init__(self, data_path, split='train', num_channels=31, segment_length=4000, 
                 segment_overlap=0.5, transform=None, channel_indices=None):
        """
        Args:
            data_path: Path to SEED_processed folder
            split: 'train', 'val', or 'test'
            num_channels: Number of EEG channels to use (31 for selected channels)
            segment_length: Length of each segment in samples (e.g., 4000 = 20 sec @ 200Hz)
            segment_overlap: Overlap ratio between segments (0.5 = 50%)
            transform: Optional transform to apply
            channel_indices: List of channel indices to select from 62 channels.
                           If None, uses default 31-channel selection.
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.segment_length = segment_length
        self.transform = transform
        
        # Default 31-channel selection (customize based on your EEG montage)
        if channel_indices is None:
            # Select even-numbered channels: 0, 2, 4, ..., 60 (31 channels total)
            self.channel_indices = list(range(0, 62, 2))
        else:
            self.channel_indices = channel_indices
        
        assert len(self.channel_indices) == num_channels, \
            f"Number of channel indices ({len(self.channel_indices)}) must match num_channels ({num_channels})"
        
        # Load all subject files for the split
        split_path = Path(data_path) / split
        subject_files = sorted(split_path.glob('*.npy'))
        
        if len(subject_files) == 0:
            raise ValueError(f"No data files found in {split_path}")
        
        print(f"Loading {split} data from {len(subject_files)} subjects...")
        print(f"Using {num_channels} channels from indices: {self.channel_indices}")
        
        # Load and segment all data
        self.segments = []
        for subj_file in subject_files:
            data = np.load(subj_file)  # Shape: (num_trials, 62, 104000)
            
            # Select only the desired channels
            data = data[:, self.channel_indices, :]  # (num_trials, 31, 104000)
            
            # Extract segments from each trial
            for trial_idx in range(data.shape[0]):
                trial_data = data[trial_idx]  # (31, 104000)
                segments = self._create_segments(trial_data, segment_length, segment_overlap)
                self.segments.extend(segments)
        
        self.segments = np.array(self.segments)  # (num_segments, 31, segment_length)
        print(f"Loaded {split} data: {len(subject_files)} subjects, {self.segments.shape[0]} segments")
        print(f"Segment shape: {self.segments[0].shape}")
    
    def _create_segments(self, trial_data, segment_length, overlap):
        """
        Create overlapping segments from a single trial
        
        Args:
            trial_data: (31, 104000) - single trial
            segment_length: Length of each segment
            overlap: Overlap ratio (0.5 = 50%)
        
        Returns:
            List of segments, each of shape (31, segment_length)
        """
        num_channels, total_length = trial_data.shape
        step_size = int(segment_length * (1 - overlap))
        
        segments = []
        start = 0
        while start + segment_length <= total_length:
            segment = trial_data[:, start:start + segment_length]
            segments.append(segment)
            start += step_size
        
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        # Get EEG segment: (31, segment_length)
        eeg = self.segments[idx].copy()
        
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


def seed_transform(x, sparse_rate=0.2):
    """
    Augmentation: randomly zero out some time points
    x: (num_channels, time_len)
    """
    x_aug = x.copy()
    num_timepoints = x.shape[1]
    idx = np.random.choice(num_timepoints, int(num_timepoints * sparse_rate), replace=False)
    x_aug[:, idx] = 0
    return x_aug
