import torch
import torch.nn.functional as F
import numpy as np

def downsample_channels_average(x, target_channels=31):
    """Downsample EEG from 62 to 31 channels using linear interpolation along channel dim."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1, 62, T)
    x = x.to(torch.float32)
    # interpolate along channel dimension, not time
    x_swapped = x.transpose(1, 2)  # (B, T, C)
    x_down = F.interpolate(x_swapped, size=target_channels, mode='linear', align_corners=True)
    x_down = x_down.transpose(1, 2)  # back to (B, C, T)
    return x_down.squeeze(0) if x_down.shape[0] == 1 else x_down


def upsample_channels_linear(x, target_channels=62):
    """Upsample EEG from 31 to 62 channels using linear interpolation along channel dim."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1, 31, T)
    x = x.to(torch.float32)
    # interpolate along channel dimension (2nd axis)
    x_swapped = x.transpose(1, 2)  # (B, T, C)
    x_up = F.interpolate(x_swapped, size=target_channels, mode='linear', align_corners=True)
    x_up = x_up.transpose(1, 2)  # back to (B, C, T)
    return x_up.squeeze(0) if x_up.shape[0] == 1 else x_up