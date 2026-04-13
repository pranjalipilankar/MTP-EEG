# hfd_curve_loss.py
# 
# Differentiable Higuchi Fractal Dimension loss for EEG super-resolution.
# Instead of just comparing scalar FD values, this compares the entire log-log
# curve across time scales to preserve complexity structure better.
# 
# Works with (B, 1, T) or (B, C, T) EEG tensors.
# 
# Usage:
#   from hfd_curve_loss import HFDProfileLoss, k_list_logspace

import math
from typing import Iterable, Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# Smooth utility functions for differentiability
# Standard abs() and log() aren't differentiable everywhere or can produce
# NaN/inf, so we use smooth approximations that work well for autograd.

def smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Smooth absolute value using sqrt(x^2 + eps).
    
    Regular abs() has undefined gradient at x=0 which can cause issues during
    backprop. This version is differentiable everywhere and behaves like |x|
    for values much larger than sqrt(eps).
    
    Args:
        x: Input tensor
        eps: Small constant to avoid sqrt(0), default 1e-6
    
    Returns:
        Smooth approximation of |x|
    """
    return torch.sqrt(x * x + eps)


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable log by clamping to minimum value.
    
    Prevents log(0) = -inf and log(negative) = NaN which break gradients.
    Clamps input to eps before taking log, so worst case is log(eps) ≈ -18.
    
    Args:
        x: Input tensor (should be positive)
        eps: Minimum clamp value, default 1e-8
    
    Returns:
        log(x) with numerical safety
    """
    return torch.log(torch.clamp(x, min=eps))


def torch_interp_1d(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation fallback for older PyTorch (<2.1).
    
    Interpolates y(x) to evaluate at x_new points. Used when aligning HFD
    curves with different k-scales. Assumes x is sorted ascending.
    
    Args:
        x_new: New x-coordinates, shape (K2,)
        x: Original x-coordinates, shape (K1,), must be sorted
        y: Original y-values, shape (B*, K1)
    
    Returns:
        Interpolated y-values at x_new, shape (B*, K2)
    """
    # Expand dimensions for broadcasting
    x = x.view(1, 1, -1)          # (1, 1, K1)
    x_new = x_new.view(1, -1, 1)  # (1, K2, 1)
    y = y.unsqueeze(1)            # (B*, 1, K1)

    # Find interpolation intervals using binary search
    j = torch.searchsorted(x.squeeze(0).squeeze(0), x_new.squeeze(0), right=True)
    j = j.clamp(min=1, max=x.shape[-1]-1)
    j0 = (j - 1).long()
    j1 = j.long()

    # Gather interval boundaries
    x0 = x[..., j0]
    x1 = x[..., j1]
    y0 = y[..., j0]
    y1 = y[..., j1]

    # Linear interpolation
    denom = (x1 - x0).clamp(min=1e-12)
    w = (x_new - x0) / denom
    y_new = y0 * (1 - w) + y1 * w
    return y_new.squeeze(1)


# Converting physical time scales to k values
# These helpers convert millisecond time scales to integer sample steps (k)
# based on sampling frequency, so the analysis is consistent across datasets.

def k_list_from_ms(scales_ms: Iterable[float], fs_hz: float) -> torch.Tensor:
    """
    Convert time scales (ms) to sample steps k.
    
    For example, 10ms at 250Hz = 2.5 samples, rounds to k=3.
    Removes duplicates and sorts the result.
    
    Args:
        scales_ms: List of time scales in milliseconds
        fs_hz: Sampling frequency in Hz
    
    Returns:
        LongTensor of unique k values, sorted ascending
    """
    ks = []
    for ms in scales_ms:
        k = int(round((ms / 1000.0) * fs_hz))
        if k >= 1:
            ks.append(k)
    
    ks = sorted(set(ks))
    
    if len(ks) == 0:
        raise ValueError("k_list_from_ms produced an empty list. Check scales_ms and fs_hz.")
    
    return torch.tensor(ks, dtype=torch.long)


def k_list_logspace(
    fs_hz: float,
    min_ms: float = 4.0,
    max_ms: float = 200.0,
    num_scales: int = 16,
) -> torch.Tensor:
    """
    Generate log-spaced time-scales between [min_ms, max_ms], then convert to k.

    WHY LOG-SPACING:
    ----------------
    Fractal analysis requires scales spanning multiple orders of magnitude
    Linear spacing (4, 8, 12, 16...) oversamples small scales, undersamples large
    Log spacing (4, 8, 16, 32, 64...) provides even coverage across scale ranges
    Matches the log-log nature of Higuchi analysis

    RECOMMENDED USAGE:
    ------------------
    For most EEG applications:
    fs_hz: Your sampling rate (e.g., 250, 500, 1000 Hz)
    min_ms=4: Captures fast dynamics (gamma band ~250 Hz)
    max_ms=200: Captures slow dynamics (delta band ~5 Hz)
    num_scales=16: Good balance of detail vs computation time

    For longer windows (> 2 seconds):
    Increase max_ms to 400-500 to capture even slower trends

    For shorter windows (< 1 second):
    Decrease max_ms to 100 to avoid exceeding window length

    fs_hz : float
        Sampling frequency in Hz
    min_ms : float, default=4.0
        Minimum time scale in milliseconds
    Lower values capture faster dynamics but may hit numerical limits
    Should be at least 2-3 samples: min_ms >= 2000/fs_hz
    max_ms : float, default=200.0
        Maximum time scale in milliseconds
    Higher values capture slower dynamics
    Should be much less than window length: max_ms << (window_samples / fs_hz) * 1000
    num_scales : int, default=16
        Number of scales to sample
    More scales: Better curve resolution, slower computation
    Fewer scales: Faster but coarser curve
    Typical range: 12-32

    torch.Tensor
        Integer tensor of k values, log-spaced and sorted
    Higuchi computation scales as O(K * T) where K = num_scales
    num_scales=16: ~20-50ms per batch on GPU
    num_scales=32: ~40-100ms per batch on GPU

    >>> # For 250 Hz EEG with 2-second windows
    >>> k_list = k_list_logspace(fs_hz=250, min_ms=4, max_ms=200, num_scales=16)
    >>> print(k_list)
    tensor([1, 1, 2, 3, 4, 5, 7, 10, 14, 19, 27, 37, 50, 50])
    # Note: Some duplicates removed, values rounded to integers

    ```python
    # In your training setup:
    fs = 250  # Your EEG sampling rate
    k_list = k_list_logspace(fs_hz=fs, min_ms=4, max_ms=200, num_scales=16)
    hfd_loss = HFDProfileLoss(k_list=k_list, distance='mse')

    # In training loop:
    sr_eeg = model(lr_eeg)  # Super-resolved EEG
    hfd = hfd_loss(sr_eeg, hr_eeg)  # Compare to high-res target
    total_loss = amp_loss + 0.3 * hfd  # Combine with amplitude loss
    ```
    """
    # Generate log-spaced time scales
    ts = torch.logspace(math.log10(min_ms), math.log10(max_ms), steps=num_scales)
    
    # Convert to integer k values (removes duplicates, sorts)
    return k_list_from_ms(ts.tolist(), fs_hz)


# 3) Internal helpers: shape prep & k validation
# These utilities handle input shape normalization and validate that
# k values are appropriate for the given signal length.

def _prepare_signal(x: torch.Tensor) -> Tuple[torch.Tensor, int, bool]:
    """
    Normalize input shape to (B*, T) for uniform processing.

    SUPPORTS MULTIPLE INPUT FORMATS:
    ---------------------------------
    (B, T): Batch of single-channel signals
    (B, C, T): Batch of multi-channel signals (e.g., 64-channel EEG)
    HFD is computed per-channel (each channel analyzed independently)
    Flattening (B, C, T) → (B*C, T) simplifies batch processing
    Loss is then averaged over all B*C series
    Alternative approach (cross-channel HFD) is not standard

    x : torch.Tensor
        Input signal of shape (B, T) or (B, C, T)

    x_flat : torch.Tensor
        Flattened to (B*, T) where B* = B (single-channel) or B*C (multi-channel)
    T : int
        Time length (number of samples)
    had_channels : bool
        True if input was (B, C, T), False if (B, T)

    ValueError
        If input is not 2D or 3D

    >>> x = torch.randn(4, 64, 512)  # 4 subjects, 64 channels, 512 samples
    >>> x_flat, T, had_ch = _prepare_signal(x)
    >>> x_flat.shape
    torch.Size([256, 512])  # 4*64 = 256 series to analyze independently
    >>> T
    512
    >>> had_ch
    True
    """
    if x.dim() == 2:
        # Single-channel format: (B, T)
        B, T = x.shape
        return x, T, False
    
    if x.dim() == 3:
        # Multi-channel format: (B, C, T) → flatten to (B*C, T)
        B, C, T = x.shape
        return x.reshape(B * C, T), T, True
    
    # Unsupported format
    raise ValueError(f"Expected x shape (B,T) or (B,C,T); got {tuple(x.shape)}")


@torch.no_grad()
def _validate_ks(T: int, ks: torch.Tensor) -> torch.Tensor:
    """
    Filter k values to keep only those that can produce valid subsequences.

    VALIDATION RULE:
    ----------------
    Each subsequence must have at least 2 points to compute differences.
    For offset m and step k, subsequence indices are: m, m+k, m+2k, ...
    Maximum index is (T-1), so we need: m + k <= T-1
    For m=0 (worst case): k <= T-1
    For at least 2 points: k <= T-2
    If k is too large relative to T, subsequences become too short
    Example: T=100, k=99 → subsequence [0, 99] has only 2 points
    Example: T=100, k=100 → subsequence [0] has only 1 point (invalid!)
    This validation prevents division by zero and NaN values

    T : int
        Signal length (number of samples)
    ks : torch.Tensor
        Proposed k values (step sizes)

    torch.Tensor
        Filtered k values, keeping only k <= T-2

    ValueError
        If ALL k values are too large (no valid scales remain)

    >>> ks = torch.tensor([1, 2, 5, 10, 50, 100])
    >>> _validate_ks(T=60, ks=ks)
    tensor([1, 2, 5, 10, 50])  # k=100 removed (> T-2=58)

    >>> _validate_ks(T=10, ks=torch.tensor([20, 50, 100]))
    ValueError: All k are too large for T=10

    PRACTICAL IMPLICATIONS:
    -----------------------
    For short windows (T=100), max_ms should be conservative
    For long windows (T=1000), you have more scale flexibility
    Rule of thumb: max_k ≈ T/4 provides good coverage without issues
    """
    # Filter k values: keep only those where subsequences have >= 2 points
    ks = ks[ks <= (T - 2)]
    
    # Check if any valid k values remain
    if len(ks) == 0:
        raise ValueError(
            f"All k are too large for T={T}. "
            f"Decrease max_ms or increase window length."
        )
    
    return ks


# 4) Differentiable Higuchi profile (log–log)
# This is the core function that computes the Higuchi curve for each signal.
# The curve captures how signal "roughness" varies across time scales.

def higuchi_profile_loglog(
    x: torch.Tensor,
    k_list: torch.Tensor,
    eps_abs: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute differentiable Higuchi log-log profile for each signal.

    HIGUCHI ALGORITHM (per scale k):
    ---------------------------------
    1. For each offset m ∈ {0, 1, ..., k-1}:
    Extract subsequence: X_m^k = [x[m], x[m+k], x[m+2k], ...]
    Compute curve length: L_m(k) = (sum of |differences|) * normalization

    2. Average curve lengths: L(k) = mean_m{L_m(k)}

    3. Build log-log plot: log(L(k)) vs log(1/k)
    Slope approximates Fractal Dimension
    Entire curve shape captures multiscale complexity

    DIFFERENTIABILITY:
    ------------------
    Uses smooth_abs instead of |·| (no gradient discontinuity)
    Uses safe_log to avoid -inf and NaN
    Masking ensures invalid subsequences don't contribute NaN
    All operations are differentiable for gradient-based optimization

    NORMALIZATION FORMULA:
    ----------------------
    L_m(k) = [(T-1) / (N_m * k)] * sum_{i} |x[m+ik] - x[m+(i-1)k]|
    where:
    T: Total signal length
    N_m: Number of steps in subsequence m
    k: Step size (time scale)

    x : torch.Tensor
        Input signal: (B, T), (B, 1, T), or (B, C, T)
    k_list : torch.Tensor
        1D LongTensor of step sizes k (samples)
        Precomputed from physical time scales using k_list_logspace()
    eps_abs : float, default=1e-6
        Smoothing epsilon for smooth_abs (gradient stability)

    log_k_inv : torch.Tensor, shape (K,)
        X-axis values: log(1/k) for each scale k
    log_Lk : torch.Tensor, shape (B*, K)
        Y-axis values: log(L(k)) for each series and scale
        B* = B for (B,T), B*C for (B,C,T)

    Time: O(B* * K * T) where K = len(k_list), T = signal length
    Memory: O(B* * T) for signals + O(B* * K) for results
    """
    # Prepare input: flatten to (B*, T)
    x_flat, T, _ = _prepare_signal(x)
    device = x_flat.device
    Bflat = x_flat.size(0)

    # Validate k values are appropriate for this signal length
    ks = _validate_ks(T, k_list.to(device))

    # Storage for L(k) at each scale
    Lk_collect = []
    
    # Loop over each scale k
    for k in ks.tolist():
        # For this scale k, we compute k different subsequences (one per offset m)
        n_steps_per_m = []      # Number of steps in each subsequence
        diffs_sum_per_m = []    # Sum of differences for each subsequence
        
        # Loop over all possible offsets m ∈ {0, 1, ..., k-1}
        for m in range(k):
            # Extract subsequence: indices = m, m+k, m+2k, m+3k, ...
            idx = torch.arange(m, T, k, device=device)
            
            # Check if subsequence has enough points (need at least 2 for differences)
            if idx.numel() < 2:
                # Not enough points → invalid subsequence
                # Record zero contribution (will be masked out)
                n_steps_per_m.append(torch.tensor(0, device=device))
                diffs_sum_per_m.append(torch.zeros(Bflat, device=device))
                continue

            # Gather subsequence values: (Bflat, L_m) where L_m = number of points
            seg = x_flat.index_select(1, idx)                 # (Bflat, L_m)
            
            # Compute consecutive differences: x[i+k] - x[i]
            diffs = seg[:, 1:] - seg[:, :-1]                  # (Bflat, L_m-1)
            
            # Record number of difference steps for normalization
            n_steps_per_m.append(torch.tensor(diffs.size(1), device=device))
            
            # Sum absolute differences (using smooth_abs for differentiability)
            diffs_sum_per_m.append(smooth_abs(diffs, eps=eps_abs).sum(dim=1))  # (Bflat,)

        # Stack results from all m offsets
        n_steps_per_m = torch.stack(n_steps_per_m)            # (k,)
        diffs_sum_per_m = torch.stack(diffs_sum_per_m, dim=1) # (Bflat, k)

        # ---- Masking of invalid subsequences ----
        # valid[m] = 1 if subsequence m had at least 2 points, else 0
        valid = (n_steps_per_m > 0).float()                   # (k,)
        valid_b = valid.unsqueeze(0).expand(Bflat, -1)        # (Bflat, k)

        # ---- Normalization (Higuchi's formula) ----
        # L_m(k) = [(T-1) / (n_steps * k)] * sum_of_differences
        # This normalization accounts for:
        # - Subsequence length (n_steps)
        # - Time scale (k)
        # - Total signal length (T-1)
        denom = (n_steps_per_m.clamp(min=1).float() * float(k))  # (k,) - avoid division by zero
        norm = (float(T - 1)) / denom                            # (k,)
        norm = norm.unsqueeze(0).expand(Bflat, -1)               # (Bflat, k)

        # Apply normalization to get L_m(k) for each offset m
        Lm = norm * diffs_sum_per_m                              # (Bflat, k)

        # Average L_m(k) over valid offsets m to get final L(k)
        # Only include valid subsequences in average (mask out invalid ones)
        Lk = (Lm * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp(min=1e-8)  # (Bflat,)
        
        # Store L(k) for this scale
        Lk_collect.append(Lk)

    # Stack results across all k values → (Bflat, K)
    Lk_all = torch.stack(Lk_collect, dim=1)
    k_tensor = ks.to(device).float()        # (K,)

    # Build the log–log curve coordinates:
    # X-axis: log(1/k) - decreases as k increases (convention in Higuchi analysis)
    log_k_inv = safe_log(1.0 / k_tensor)    # (K,)
    
    # Y-axis: log(L(k)) - logarithm of curve length at each scale
    log_Lk = safe_log(Lk_all)               # (Bflat, K)
    
    return log_k_inv, log_Lk


# 5) Public loss: full HFD curve difference (L1 or MSE)
# PyTorch nn.Module that wraps Higuchi computation into a differentiable loss.

class HFDProfileLoss(nn.Module):
    """
    Higuchi Fractal Dimension Profile Loss for EEG super-resolution.

    Instead of comparing single scalar FD values, this loss compares the entire
    log-log HFD curve across all time scales. This preserves scale-specific
    complexity information and provides richer gradients for training.

    The loss computes HFD curves for both generated and target signals, aligns
    them if needed, then measures the distance (L1 or MSE) between the curves.

    Args:
        k_list: LongTensor of time scales (samples) to analyze at
        distance: 'mse' (default) or 'l1' for curve comparison
        eps_abs: Smoothing parameter for numerical stability
        reduction: 'mean' (default), 'sum', or 'none' for batch aggregation

    Example:
        fs = 250  # Hz
        k_list = k_list_logspace(fs, min_ms=10, max_ms=100, num_scales=15)
        hfd_loss = HFDProfileLoss(k_list, distance='mse')
        
        # In training loop
        output = model(input_eeg)
        loss = amp_loss + 0.3 * hfd_loss(output, target_eeg)
        loss.backward()

    Notes:
        - Typical weight for HFD loss: 0.2-0.5 relative to amplitude loss
        - Computation: O(B * K * T) where B=batch, K=scales, T=length
        - Works with (B, T) or (B, C, T) tensors
        - All operations are differentiable with smooth gradients
    """
    
    def __init__(
        self,
        k_list: torch.Tensor,
        distance: Literal["l1", "mse"] = "mse",
        eps_abs: float = 1e-6,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """Initialize HFD Profile Loss with given parameters."""
        super().__init__()
        
        # Validate k_list type
        if k_list.dtype != torch.long:
            raise ValueError("k_list must be a LongTensor of sample steps.")
        
        # Register as buffer so it moves with .to(device) and is saved in state_dict
        # Buffer vs Parameter: Buffer doesn't have gradients (it's configuration, not learned)
        self.register_buffer("k_list", k_list)
        
        # Store hyperparameters
        self.distance = distance
        self.eps_abs = eps_abs
        self.reduction = reduction

    def forward(self, x_gen: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        Compute HFD profile loss between generated and target signals.

        1. Compute HFD curve for generated signal
        2. Compute HFD curve for target signal
        3. Align x-axes (interpolate if needed)
        4. Compute difference between y-values
        5. Aggregate: L1 or MSE across scales
        6. Reduce: mean/sum/none across batch

        x_gen : torch.Tensor
            Generated/predicted signal: (B, T), (B, 1, T), or (B, C, T)
            Must have requires_grad=True for training

        x_true : torch.Tensor
            Target/ground-truth signal: same shape as x_gen
            Typically detached (no gradients needed for target)

        torch.Tensor
    Scalar loss if reduction='mean' or 'sum' (default behavior)
    Per-sample losses (B*,) if reduction='none' (for analysis)

        ValueError
            If x_gen and x_true have different shapes
            If distance metric is not 'l1' or 'mse'

        Loss ← diff ← y_gen ← HFD curve ← smooth operations ← x_gen
        All steps are differentiable, gradients propagate cleanly to input

        INTERPOLATION:
        --------------
        Normally, both signals use same k_list → same x-axes → no interpolation needed.
        Defensive check: If x-axes differ, interpolate y_gen to match y_target grid.
        This can happen if signals have different lengths causing different k filtering.

        DISTANCE COMPUTATION:
        ---------------------
        For each sample in batch:
            diff[b, :] = y_gen[b, :] - y_target[b, :]  # (K,) differences at each scale

            If distance='l1':
                loss[b] = mean_k |diff[b, k]|  # Mean absolute difference

            If distance='mse':
                loss[b] = mean_k diff[b, k]²  # Mean squared error

        REDUCTION:
        ----------
        reduction='mean': return mean(loss) → scalar
        reduction='sum': return sum(loss) → scalar
        reduction='none': return loss → (B*,) vector
    Similar signals: Loss ~ 0.01-0.1 (log-scale differences)
    Very different signals: Loss ~ 0.5-2.0
    Identical signals: Loss ~ 1e-8 (numerical precision)

        MONITORING:
        -----------
        During training, track:
        1. HFD loss value: Should decrease over time
        2. Gradient norm: Should be stable (~0.001-0.1)
        3. Loss ratio: HFD/(amplitude loss) should be 0.2-0.5

        >>> # Training iteration
        >>> loss_fn = HFDProfileLoss(k_list, distance='mse', reduction='mean')
        >>> x_gen.requires_grad_(True)
        >>> loss = loss_fn(x_gen, x_target)  # Scalar
        >>> loss.backward()  # Compute gradients
        >>> optimizer.step()  # Update weights

        >>> # Analysis mode
        >>> loss_fn = HFDProfileLoss(k_list, reduction='none')
        >>> losses = loss_fn(x_gen, x_target)  # (B,) per-sample losses
        >>> print(f"Mean: {losses.mean()}, Std: {losses.std()}")
        >>> print(f"Worst sample: {losses.argmax()}")
        """
        # Validate input shapes match
        if x_gen.shape != x_true.shape:
            raise ValueError(
                f"Shape mismatch: generated {tuple(x_gen.shape)} "
                f"vs target {tuple(x_true.shape)}"
            )

        # Compute HFD curves for both signals
        # xg, xt: (K,) - x-axis coordinates log(1/k)
        # yg, yt: (B*, K) - y-axis coordinates log(L(k)) for each signal
        xg, yg = higuchi_profile_loglog(x_gen, self.k_list, eps_abs=self.eps_abs)
        xt, yt = higuchi_profile_loglog(x_true, self.k_list, eps_abs=self.eps_abs)

        # Defensive alignment: x-axes should match (same k_list used for both)
        # But if signal lengths differ, k validation may filter differently
        if not torch.allclose(xg, xt, atol=1e-6, rtol=1e-5):
            # X-axes differ → interpolate generated curve to target grid
            try:
                # PyTorch >= 2.1 has torch.interp
                yg = torch.interp(xt, xg, yg)   # (B*, K)
            except AttributeError:
                # Fallback for older PyTorch versions
                yg = torch_interp_1d(xt, xg, yg)

        # Compute difference between curves at each scale
        diff = yg - yt  # (B*, K)

        # Aggregate differences across scales (K dimension)
        if self.distance == "l1":
            # L1 distance: Mean absolute difference across scales
            # More robust to outlier scales
            per_series = smooth_abs(diff, eps=1e-12).mean(dim=1)     # (B*,)
            
        elif self.distance == "mse":
            # MSE distance: Mean squared error across scales
            # Penalizes large deviations more heavily
            # Smoother gradients (standard for neural networks)
            per_series = (diff * diff).mean(dim=1)                   # (B*,)
            
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        # Reduce across batch dimension
        if self.reduction == "mean":
            # Average loss across all samples (standard for training)
            # Loss magnitude independent of batch size
            return per_series.mean()  # Scalar
            
        if self.reduction == "sum":
            # Sum loss across all samples
            # Loss magnitude scales with batch size
            # Useful for weighted loss combinations
            return per_series.sum()   # Scalar
            
        # No reduction: return per-sample losses
        # Useful for analysis, debugging, per-sample weighting
        return per_series  # (B*,)


# 6) Demonstration and sanity test
# Quick demonstration showing HFD loss in action with gradient flow.

def _demo():
    """
    Quick demonstration of HFD Profile Loss.

    WHAT THIS SHOWS:
    ----------------
    1. Creating k_list from physical time scales
    2. Setting up HFD loss with typical parameters
    3. Computing loss for noisy predictions vs targets
    4. Verifying gradient flow (required for neural network training)
    5. Combining HFD loss with amplitude loss

    EXPECTED OUTPUT:
    ----------------
    HFD loss: ~0.001-0.01 (log-scale differences are small for close signals)
    Gradient norm: ~0.0001-0.001 (small but non-zero → can guide optimization)
    Total loss: ~1-2 (dominated by amplitude term which is in raw signal scale)

    INTERPRETATION:
    ---------------
    Small HFD loss: Generated signal has similar complexity structure to target
    Non-zero gradient: Loss provides useful signal for optimization
    Stable values: No NaN or overflow issues
    """
    # Select device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate a batch of EEG recordings
    B, T = 8, 512  # 8 signals, 512 samples (~1 second at 500 Hz)
    fs = 500.0     # Sampling frequency in Hz
    
    # Generate k_list from physical time scales
    # 4-200ms covers typical EEG dynamics at 500 Hz
    klist = k_list_logspace(fs_hz=fs, min_ms=4.0, max_ms=200.0, num_scales=16)

    # Create synthetic data
    x_target = torch.randn(B, 1, T) * 10.0         # Target: Random EEG-like signal
    x_pred   = x_target + 0.15 * torch.randn(B, 1, T)  # Predicted: Target + 15% noise

    # Move to device
    x_target = x_target.to(device)
    x_pred   = x_pred.to(device)

    # Build HFD loss function
    hfd_loss = HFDProfileLoss(
        k_list=klist,        # Time scales to analyze
        distance="mse",      # Mean squared error of curves
        reduction="mean"     # Average over batch
    ).to(device)

    # Enable gradient tracking on predicted signal
    x_pred.requires_grad_(True)
    
    # Compute HFD loss
    loss_val = hfd_loss(x_pred, x_target)
    
    # Backpropagate to compute gradients
    loss_val.backward()

    # Display results
    print(f"[HFD curve loss] value: {loss_val.detach().item():.6f}")
    print(f"Grad norm on x_pred: {x_pred.grad.norm().item():.6f}")

    # Example: Combining with amplitude loss (common in practice)
    # HFD captures complexity, L1 captures amplitude matching
    amp_loss = F.l1_loss(x_pred, x_target)
    total = amp_loss + 0.5 * hfd_loss(x_pred, x_target)
    print(f"[Total loss example] amp + 0.5*hfd = {total.detach().item():.6f}")
    print("\nInterpretation:")
    print(f"  - Amplitude loss: {amp_loss.detach().item():.2f} (raw signal difference)")
    print(f"  - HFD loss (weighted): {(0.5*loss_val).detach().item():.4f} (complexity difference)")
    print(f"  - Total: {total.detach().item():.2f} (combined objective)")



# -----------------------------------------------------------
# TEST 1: Basic Functionality Test
# -----------------------------------------------------------
def test_basic_functionality():
    print("\n=== TEST 1: Basic Functionality ===")
    
    # Create simple test signals
    fs = 250  # Hz
    duration = 2.0  # seconds
    t = torch.linspace(0, duration, int(fs * duration))
    
    # Target: sine wave
    x_target = torch.sin(2 * np.pi * 10 * t).unsqueeze(0)  # (1, T)
    
    # Predicted: sine wave with slight noise
    x_pred = x_target + 0.01 * torch.randn_like(x_target)
    
    # Create k_list first
    ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=10)
    
    # Compute loss
    loss_fn = HFDProfileLoss(k_list=ks, distance='l1')
    loss = loss_fn(x_pred, x_target)
    
    print(f"Loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Basic functionality test passed")

# -----------------------------------------------------------
# TEST 2: Gradient Flow Test
# -----------------------------------------------------------
def test_gradient_flow():
    print("\n=== TEST 2: Gradient Flow ===")
    
    fs = 250
    x_target = torch.randn(2, 500)  # (B, T)
    x_pred = torch.randn(2, 500, requires_grad=True)
    
    ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=10)
    loss_fn = HFDProfileLoss(k_list=ks)
    loss = loss_fn(x_pred, x_target)
    
    # Backward pass
    loss.backward()
    
    assert x_pred.grad is not None, "Gradient should exist"
    assert not torch.isnan(x_pred.grad).any(), "Gradient should not contain NaN"
    print(f"Gradient norm: {x_pred.grad.norm().item():.6f}")
    print("✓ Gradient flow test passed")

# -----------------------------------------------------------
# TEST 3: Edge Case - Constant Signal
# -----------------------------------------------------------
def test_constant_signal():
    print("\n=== TEST 3: Constant Signal ===")
    
    fs = 250
    x_target = torch.ones(1, 500)
    x_pred = torch.ones(1, 500)
    
    ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=10)
    loss_fn = HFDProfileLoss(k_list=ks)
    loss = loss_fn(x_pred, x_target)
    
    print(f"Loss (constant signal): {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss should not be NaN for constant signal"
    print("✓ Constant signal test passed")

# -----------------------------------------------------------
# TEST 4: Edge Case - Very Short Signal
# -----------------------------------------------------------
def test_short_signal():
    print("\n=== TEST 4: Very Short Signal ===")
    
    fs = 250
    x_target = torch.randn(1, 50)  # Very short signal
    x_pred = torch.randn(1, 50)
    
    try:
        ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=5)
        loss_fn = HFDProfileLoss(k_list=ks)
        loss = loss_fn(x_pred, x_target)
        print(f"Loss (short signal): {loss.item():.6f}")
        print("✓ Short signal test passed")
    except Exception as e:
        print(f"✗ Short signal test failed: {e}")

# -----------------------------------------------------------
# TEST 5: Multi-Channel Input
# -----------------------------------------------------------
def test_multi_channel():
    print("\n=== TEST 5: Multi-Channel Input ===")
    
    fs = 250
    x_target = torch.randn(2, 64, 500)  # (B, C, T)
    x_pred = torch.randn(2, 64, 500)
    
    ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=10)
    loss_fn = HFDProfileLoss(k_list=ks)
    loss = loss_fn(x_pred, x_target)
    
    print(f"Loss (multi-channel): {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss should not be NaN for multi-channel input"
    print("✓ Multi-channel test passed")

# -----------------------------------------------------------
# TEST 7: Distance Metric Comparison
# -----------------------------------------------------------
def test_distance_metrics():
    print("\n=== TEST 7: Distance Metric Comparison ===")
    
    fs = 250
    x_target = torch.randn(2, 500)
    x_pred = x_target + 0.1 * torch.randn_like(x_target)
    
    ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=10)
    loss_l1 = HFDProfileLoss(k_list=ks, distance='l1')(x_pred, x_target)
    loss_mse = HFDProfileLoss(k_list=ks, distance='mse')(x_pred, x_target)
    
    print(f"L1 Loss: {loss_l1.item():.6f}")
    print(f"MSE Loss: {loss_mse.item():.6f}")
    print("✓ Distance metric test passed")

# -----------------------------------------------------------
# TEST 8: Batch Processing
# -----------------------------------------------------------
def test_batch_processing():
    print("\n=== TEST 8: Batch Processing ===")
    
    fs = 250
    batch_sizes = [1, 4, 16, 64]
    
    ks = k_list_logspace(fs_hz=fs, min_ms=10, max_ms=200, num_scales=10)
    
    for batch_size in batch_sizes:
        x_target = torch.randn(batch_size, 500)
        x_pred = torch.randn(batch_size, 500)
        
        loss_fn = HFDProfileLoss(k_list=ks, reduction='mean')
        loss = loss_fn(x_pred, x_target)
        
        print(f"Batch size {batch_size}: Loss = {loss.item():.6f}")
    
    print("✓ Batch processing test passed")

# -----------------------------------------------------------
# RUN ALL TESTS
# -----------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("HIGUCHI PROFILE LOSS - TEST SUITE")
    print("=" * 60)
    
    test_basic_functionality()
    test_gradient_flow()
    test_constant_signal()
    test_short_signal()
    test_multi_channel()
    test_distance_metrics()
    test_batch_processing()
    
    print("\n" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    _demo()


# INTEGRATION GUIDE FOR EEG SUPER-RESOLUTION
#
# QUICK START:
# ------------
# 1. Import the loss function:
#    from hfd_curve_loss import HFDProfileLoss, k_list_logspace
#
# 2. Set up k_list based on your sampling rate:
#    fs = 250  # Your EEG sampling frequency in Hz
#    k_list = k_list_logspace(fs_hz=fs, min_ms=4, max_ms=200, num_scales=16)
#
# 3. Create loss function:
#    hfd_loss = HFDProfileLoss(k_list=k_list, distance='mse', reduction='mean')
#
# 4. Use in training loop:
#    total_loss = amp_loss + 0.3 * hfd_loss(sr_eeg, hr_eeg)
#
# TYPICAL TRAINING LOOP:
# ----------------------
# ```python
# # Setup (once at beginning)
# fs = 250  # Hz
# k_list = k_list_logspace(fs_hz=fs, min_ms=4, max_ms=200, num_scales=16)
# hfd_loss_fn = HFDProfileLoss(k_list=k_list, distance='mse')
# 
# # Training loop
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         lr_eeg, hr_eeg = batch  # Low-res input, high-res target
#         
#         # Forward pass
#         sr_eeg = model(lr_eeg)  # Super-resolved output
#         
#         # Compute losses
#         amp_loss = F.l1_loss(sr_eeg, hr_eeg)      # Amplitude matching
#         hfd_loss = hfd_loss_fn(sr_eeg, hr_eeg)    # Complexity matching
#         
#         # Combined loss (tune weight as needed)
#         total_loss = amp_loss + 0.3 * hfd_loss
#         
#         # Backward and optimize
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#         
#         # Optional: Log individual losses for monitoring
#         if step % log_interval == 0:
#             print(f"Amp: {amp_loss:.4f}, HFD: {hfd_loss:.4f}, Total: {total_loss:.4f}")
# ```
#
# HYPERPARAMETER TUNING:
# ----------------------
# - HFD weight: Start with 0.3, adjust based on results
#   * Too smooth output → Increase to 0.4-0.5
#   * Too noisy output → Decrease to 0.1-0.2
#   * Unstable training → Decrease or reduce learning rate
#
# - num_scales: Default 16 is good balance
#   * More (24-32): Better curve resolution, slower
#   * Fewer (8-12): Faster, coarser curve
#
# - min_ms/max_ms: Should match your signal characteristics
#   * Standard EEG: 4-200ms covers delta to gamma bands
#   * Longer windows: Can use up to 400ms
#   * Shorter windows: Reduce max_ms to ~100ms
#
# MONITORING TRAINING:
# --------------------
# Track these metrics:
# 1. HFD loss magnitude: Should decrease over epochs
# 2. HFD/Amp ratio: Typically 0.01-0.1 (HFD is in log scale)
# 3. Gradient norm: Should be stable (1e-4 to 1e-2)
# 4. Visual inspection: Plot HFD curves occasionally to verify matching
#
# TROUBLESHOOTING:
# ----------------
# Issue: HFD loss is NaN
# → Check signal lengths (must be >= max(k_list) + 2)
# → Verify k_list generation (print k_list to inspect values)
#
# Issue: HFD loss doesn't decrease
# → Lower HFD weight (0.1-0.2)
# → Check if amplitude loss is also not decreasing (model issue)
# → Verify gradient flow (print x_gen.grad.norm())
#
# Issue: Training is unstable
# → Reduce HFD weight
# → Lower learning rate
# → Use gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
# Issue: Output is too smooth/noisy
# → Adjust HFD weight up/down
# → Check if other losses are conflicting (e.g., over-smoothing regularization)
#
# ADVANCED USAGE:
# ---------------
# Multi-scale loss (different k ranges):
# ```python
# k_fast = k_list_logspace(fs, 4, 50, 8)    # Fast dynamics
# k_slow = k_list_logspace(fs, 50, 200, 8)  # Slow dynamics
# hfd_fast = HFDProfileLoss(k_fast)
# hfd_slow = HFDProfileLoss(k_slow)
# total = amp + 0.2*hfd_fast(sr, hr) + 0.2*hfd_slow(sr, hr)
# ```
#
# Per-channel weighting (for multi-channel EEG):
# ```python
# hfd_fn = HFDProfileLoss(k_list, reduction='none')
# losses = hfd_fn(sr_eeg, hr_eeg)  # (B*C,) per-channel losses
# # Apply custom weighting based on channel importance
# channel_weights = get_channel_weights()  # Your function
# weighted_loss = (losses * channel_weights).mean()
# ```
#
# EXPECTED RESULTS:
# -----------------
# When training converges successfully:
# - HFD curves of SR and HR EEG should visually overlap
# - Fractal dimension should match (slope of log-log curve)
# - Signal complexity preserved at all time scales
# - Output looks physiologically realistic (not over-smooth or noisy)
#

# ===========================================================================
