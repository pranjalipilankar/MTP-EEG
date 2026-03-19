import torch
import torch.nn.functional as F
import numpy as np

def downsample_channels_keep_odd(x: torch.Tensor) -> torch.Tensor:
    """Keep channels 1,3,5,... (index 1,3,..) i.e. odd-numbered channels."""
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0); squeeze = True
    out = x[:, 1::2, :]
    return out.squeeze(0) if squeeze else out

def upsample_channels_linear_even(x: torch.Tensor, target_channels: int = 32) -> torch.Tensor:
    """
    Take 16 odd channels (placed at indices 1,3,...) and interpolate even channels.
    Returns full 32-channel signal.
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0); squeeze = True
    B, C, T = x.shape
    if C != 16:
        raise ValueError("upsample_channels_linear_even expects 16 channels input")
    out = torch.zeros(B, target_channels, T, device=x.device, dtype=x.dtype)
    out[:, 1::2, :] = x  # place odd channels
    # fill even channels by neighbors' average
    # channel 0 (first) copy from channel 1
    out[:, 0, :] = out[:, 1, :]
    # even channels 2..30
    out[:, 2:-1:2, :] = 0.5 * (out[:, 1:-2:2, :] + out[:, 3::2, :])
    # near end: fill second-last even if missing
    out[:, -2, :] = 0.5 * (out[:, -3, :] + out[:, -1, :])
    # fallback for last index
    out[:, -1, :] = out[:, -2, :]
    return out.squeeze(0) if squeeze else out
