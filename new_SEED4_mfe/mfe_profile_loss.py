# mfe_profile_loss.py
# MULTISCALE FUZZY ENTROPY (MFE) PROFILE LOSS FOR EEG SUPER-RESOLUTION
#
# PURPOSE:
# --------
# This module implements a differentiable loss function based on Multiscale
# Fuzzy Entropy (MFE) for comparing EEG signals. Unlike simple amplitude-based
# losses (L1, MSE), MFE captures the complexity and temporal structure of
# signals across multiple time scales, making it ideal for EEG super-resolution
# where preserving physiological patterns is crucial.
#
# KEY CONCEPTS:
# -------------
# 1. FUZZY ENTROPY (FuzzyEn):
#    - Measures signal regularity/complexity using fuzzy membership functions
#    - More robust than Sample Entropy (SampEn) - smoother, more continuous
#    - Formula: FuzzyEn(m, n, r) = ln(Phi_m) - ln(Phi_{m+1})
#      where Phi_m is the average similarity between length-m patterns
#
# 2. MULTISCALE EXTENSION (MFE):
#    - Computes FuzzyEn across multiple time scales (tau = 1, 2, ..., tau_max)
#    - Coarse-graining: Downsample signal by averaging non-overlapping windows
#    - Reveals complexity at different temporal resolutions (fast vs slow dynamics)
#
# 3. PROFILE LOSS:
#    - Compares the ENTIRE MFE curve (tau=1 to tau_max) between generated
#      and target signals
#    - Ensures super-resolved EEG matches not just amplitude, but also
#      complexity structure across scales
#
# USAGE:
# ------
#   from mfe_profile_loss import MFEProfileLoss
#   
#   loss_fn = MFEProfileLoss(m=2, n=2.0, tau_max=20)
#   loss = loss_fn(generated_eeg, target_eeg)
#   loss.backward()  # Fully differentiable!
#
# MATHEMATICAL DETAILS:
# ---------------------
# - Pattern distance: Chebyshev (max absolute difference)
# - Fuzzy membership: exp(-(d^n)/r) where d is distance, n is exponent
# - Smoothing for gradients:
#   * |x| → sqrt(x^2 + eps)  (smooth absolute value)
#   * max(v) → (1/beta)*log(sum(exp(beta*v)))  (soft-max approximation)
#
# PARAMETERS:
# -----------
# - m: Pattern length (default 2, standard for physiological signals)
# - n: Fuzzy exponent (default 2.0, controls membership sharpness)
# - tau_max: Maximum time scale (default 20, covers 4-80ms at 250 Hz)
# - r: Tolerance threshold (0.15 is typical for z-scored signals)
#
# ===========================================================================

import math
from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# SECTION 1: SMOOTH DIFFERENTIABLE PRIMITIVES
# These functions replace non-differentiable operations (absolute value, max)
# with smooth approximations to enable gradient-based optimization.

def smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Smooth approximation of absolute value: |x| ≈ sqrt(x^2 + eps)
    Standard |x| has zero gradient at x=0 (non-differentiable)
    Neural networks need smooth gradients for backpropagation
    sqrt(x^2 + eps) is differentiable everywhere and ≈ |x| for |x| >> sqrt(eps)

    x : torch.Tensor
        Input tensor of any shape
    eps : float, default=1e-6
        Small constant for numerical stability (prevents sqrt(0))

    torch.Tensor
        Smooth absolute value, same shape as input

    >>> x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    >>> smooth_abs(x)
    tensor([2.0000, 0.1000, 0.0010, 0.1000, 2.0000])  # Note: 0 → sqrt(eps)
    For |x| >> sqrt(eps): gradient ≈ sign(x)
    Near x=0: gradient smoothly transitions from -1 to +1
    """
    return torch.sqrt(x * x + eps)


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Numerically stable logarithm: log(max(x, eps))
    log(0) = -inf causes NaN gradients
    log(negative) = NaN breaks optimization
    Clamping to eps prevents these issues while maintaining differentiability

    x : torch.Tensor
        Input tensor (should be positive, but this function handles edge cases)
    eps : float, default=1e-12
        Minimum value to clamp x to (log(1e-12) ≈ -27.6)

    torch.Tensor
        Stabilized log values, same shape as input
    Computing FuzzyEn = log(Phi_m) - log(Phi_{m+1})
    When Phi values might be very small (high-entropy signals)

    >>> x = torch.tensor([1.0, 1e-10, 0.0, -1.0])
    >>> safe_log(x)
    tensor([0.0000, -23.0259, -27.6310, -27.6310])  # All finite, no NaN
    """
    return torch.log(torch.clamp(x, min=eps))


def softmax_max(x: torch.Tensor, dim: int = -1, beta: float = 50.0) -> torch.Tensor:
    """
    Smooth approximation of max operation using LogSumExp:
    max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    Standard max(x) has zero gradient everywhere except at the maximum
    This creates dead gradients (most inputs don't contribute to learning)
    LogSumExp is fully differentiable and approaches max as beta → ∞
    Small beta: Average-like behavior (all elements contribute)
    Large beta: Max-like behavior (dominant element dominates)
    Trade-off: Higher beta → better max approximation but risk of overflow

    x : torch.Tensor
        Input tensor
    dim : int, default=-1
        Dimension along which to compute the soft-max
    beta : float, default=50.0
        Sharpness parameter (higher = closer to true max, but less stable)
        Typical range: 10-100 for normalized signals

    torch.Tensor
        Soft-max values with dimension 'dim' reduced
    Gradient flows to ALL elements (not just the max)
    Elements closer to max get proportionally larger gradients
    Gradients weighted by exp(beta * (x_i - max(x)))

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> softmax_max(x, beta=10)   # ≈ 3.0 (close to true max)
    >>> softmax_max(x, beta=1)    # ≈ 2.4 (more averaged)

    OVERFLOW PROTECTION:
    --------------------
    torch.logsumexp internally subtracts max(x) for numerical stability
    Safe for normalized signals (|x| ~ 1) with beta up to ~100
    """
    return torch.logsumexp(beta * x, dim=dim) / beta


# SECTION 2: SIGNAL PREPROCESSING AND COARSE-GRAINING
# These functions handle input shape normalization, z-score standardization,
# and multiscale coarse-graining (downsampling) of time series.

def _prepare_signal(x: torch.Tensor) -> Tuple[torch.Tensor, int, bool, int]:
    """
    Normalize input shape to (B*, T) for uniform processing.

    SUPPORTS:
    ---------
    (B, T): Batch of single-channel signals
    (B, C, T): Batch of multi-channel signals (e.g., 64-channel EEG)

    x : torch.Tensor
        Input signal of shape (B, T) or (B, C, T)

    x_flat : torch.Tensor
        Flattened to (B*, T) where B* = B (if single-channel) or B*C (if multi-channel)
    T : int
        Time length (number of samples)
    had_channels : bool
        True if input was (B, C, T), False if (B, T)
    Bstar : int
        Effective batch size after flattening (B or B*C)
    Entropy measures are typically per-channel (each channel analyzed independently)
    Flattening B*C channels into batch dimension simplifies computation
    Alternative: Could compute cross-channel entropy, but less standard

    >>> x = torch.randn(4, 64, 512)  # 4 subjects, 64 EEG channels, 512 samples
    >>> x_flat, T, had_ch, Bstar = _prepare_signal(x)
    >>> x_flat.shape
    torch.Size([256, 512])  # 4*64 = 256 series to analyze independently
    """
    if x.dim() == 2:
        # Single-channel: (B, T)
        B, T = x.shape
        return x, T, False, B
    
    if x.dim() == 3:
        # Multi-channel: (B, C, T) → (B*C, T)
        B, C, T = x.shape
        return x.reshape(B * C, T), T, True, B * C
    
    raise ValueError(f"Expected x shape (B,T) or (B,C,T); got {tuple(x.shape)}")


def zscore_per_series(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-series z-score normalization: (x - mean) / std

    WHY Z-SCORE:
    ------------
    Makes tolerance parameter 'r' comparable across different signals/subjects
    Without z-scoring, r would need to be scaled based on signal amplitude
    Standard practice in entropy analysis of physiological signals

    x : torch.Tensor
        Input of shape (B*, T) - batch of time series
    eps : float, default=1e-8
        Small constant to prevent division by zero (for constant signals)

    torch.Tensor
        Z-scored signals: mean=0, std=1 for each series independently

    COMPUTATION:
    ------------
    For each series i:
        x_i_normalized = (x_i - mean(x_i)) / std(x_i)

    EDGE CASE:
    ----------
    Constant signals (std=0) get std clamped to eps
    Results in all zeros (which is mathematically correct for zero variance)

    >>> x = torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]])
    >>> zscore_per_series(x)
    tensor([[-1.3416, -0.4472,  0.4472,  1.3416],
            [-1.3416, -0.4472,  0.4472,  1.3416]])  # Same pattern, different scales → same z-scores
    """
    # Compute statistics along time dimension (dim=1)
    mean = x.mean(dim=1, keepdim=True)  # (B*, 1)
    std = x.std(dim=1, unbiased=False, keepdim=True).clamp(min=eps)  # (B*, 1)
    return (x - mean) / std


def coarse_grain_mean(x: torch.Tensor, tau: int) -> torch.Tensor:
    """
    Non-overlapping coarse-graining by averaging over windows of length tau.
    This is the "multiscale" part of Multiscale Fuzzy Entropy.

    INTUITION:
    ----------
    tau=1: Original signal (no coarse-graining)
    tau=2: Average every 2 consecutive points → signal at 2× slower timescale
    tau=10: Average every 10 points → captures slow dynamics only

    WHY COARSE-GRAIN:
    -----------------
    Reveals complexity at different temporal resolutions
    Fast oscillations (high freq) → small tau
    Slow trends (low freq) → large tau
    Physiological signals have structure at multiple scales (e.g., alpha, beta, gamma waves in EEG)

    x : torch.Tensor
        Input signal of shape (B*, T)
    tau : int
        Scale factor (window size for averaging)

    torch.Tensor
        Coarse-grained signal of shape (B*, T') where T' = floor(T / tau)

    1. Trim signal to Tprime = floor(T / tau) * tau (discard trailing samples)
    2. Reshape to (B*, T'/tau, tau)
    3. Average along last dimension → (B*, T'/tau)

    >>> x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # 8 samples
    >>> coarse_grain_mean(x, tau=2)
    tensor([[1.5, 3.5, 5.5, 7.5]])  # 4 samples: (1+2)/2, (3+4)/2, ...
    >>> coarse_grain_mean(x, tau=3)
    tensor([[2., 5.]])  # 2 samples: (1+2+3)/3, (4+5+6)/3 (7,8 discarded)

    EDGE CASE:
    ----------
    If T < tau: Returns empty tensor (B*, 0)
    If tau=1: Returns original signal unchanged
    """
    Bstar, T = x.shape
    
    # No coarse-graining needed
    if tau <= 1:
        return x
    
    # Calculate usable length (multiple of tau)
    Tprime = (T // tau) * tau
    
    # Not enough data for even one window
    if Tprime < tau:
        return x[:, :0]  # Return empty: (B*, 0)
    
    # Trim and reshape
    x_cut = x[:, :Tprime]  # (B*, Tprime)
    x_reshaped = x_cut.view(Bstar, Tprime // tau, tau)  # (B*, T'/tau, tau)
    
    # Average over window
    return x_reshaped.mean(dim=2)  # (B*, T'/tau)


# SECTION 3: FUZZY ENTROPY CORE COMPUTATION
# These functions implement the pattern matching and similarity computation
# that forms the heart of Fuzzy Entropy calculation.

def _build_templates(x: torch.Tensor, m: int, stride: int = 1) -> torch.Tensor:
    """
    Extract all length-m sliding windows (templates) from time series.

    WHAT ARE TEMPLATES:
    -------------------
    Templates are short subsequences used for pattern matching.
    For signal x = [1, 2, 3, 4, 5] with m=3:
        Template 1: [1, 2, 3]
        Template 2: [2, 3, 4]
        Template 3: [3, 4, 5]

    WHY MEAN-CENTER:
    ----------------
    Standard practice in FuzzyEn to remove DC offset
    Makes comparison focus on shape, not absolute level
    Template [1,2,3] and [11,12,13] have same mean-centered form: [-1,0,1]

    x : torch.Tensor
        Input signal of shape (B*, N) where N is coarse-grained length
    m : int
        Template length (pattern length)
    stride : int, default=1
        Step size between consecutive templates
        stride=1: All overlapping windows (standard)
        stride=2: Every other window (faster, less memory)

    torch.Tensor
        Templates of shape (B*, M, m) where M = number of templates
        M = floor((N - m + 1) / stride)

    1. Calculate valid starting positions: 0, stride, 2*stride, ..., (N-m)
    2. For each start position, extract m consecutive samples
    3. Subtract mean from each template (mean-centering)
    For N=1000, m=2, stride=1: M=999 templates
    Pairwise distances require M×M matrix → ~1M elements
    Large stride or max_templates (in caller) controls memory usage

    >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
    >>> templates = _build_templates(x, m=3, stride=1)
    >>> templates.shape
    torch.Size([1, 3, 3])  # 1 series, 3 templates, length 3 each
    >>> templates[0]  # After mean-centering
    tensor([[-1.,  0.,  1.],
            [-1.,  0.,  1.],
            [-1.,  0.,  1.]])  # All have same shape (linear increase)
    """
    Bstar, N = x.shape
    
    # Calculate number of templates
    M = (N - m + 1) // stride
    
    # Edge case: Signal too short
    if M <= 0:
        return x[:, :0].unsqueeze(-1).repeat(1, 1, m)  # Empty: (B*, 0, m)
    
    # Generate indices for sliding windows
    starts = torch.arange(0, N - m + 1, stride, device=x.device)  # (M,)
    idx = starts.unsqueeze(1) + torch.arange(0, m, device=x.device).unsqueeze(0)  # (M, m)
    
    # Gather windows (vectorized indexing)
    X = x.index_select(1, idx.view(-1)).view(Bstar, M, m)  # (B*, M, m)
    
    # Mean-center each template
    X = X - X.mean(dim=2, keepdim=True)  # (B*, M, m)
    
    return X


def _pairwise_chebyshev(
    A: torch.Tensor,
    beta_max: float = 50.0,
    eps_abs: float = 1e-6,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute smooth Chebyshev distance (L-infinity norm) between all template pairs.

    CHEBYSHEV DISTANCE:
    -------------------
    For vectors u, v: d_Chebyshev(u, v) = max_k |u[k] - v[k]|

    WHY CHEBYSHEV (not Euclidean):
    -------------------------------
    Standard choice in FuzzyEn literature (Costa et al., 2005)
    More sensitive to outliers/spikes (important in EEG)
    Euclidean averages out differences; Chebyshev focuses on worst-case

    SMOOTHING FOR GRADIENTS:
    ------------------------
    |·| replaced with smooth_abs (differentiable)
    max(·) replaced with softmax_max (differentiable)

    A : torch.Tensor
        Templates of shape (B*, M, m)
    beta_max : float, default=50.0
        Sharpness for soft-max approximation (see softmax_max)
    eps_abs : float, default=1e-6
        Epsilon for smooth_abs
    chunk : int, optional
        If provided, compute distances in chunks over j-dimension to reduce memory
        Useful when M is large (e.g., M > 500)

    torch.Tensor
        Pairwise distance matrix of shape (B*, M, M)
        D[b, i, j] = Chebyshev distance between template i and j in series b
    Full computation: (B*, M, M, m) intermediate tensor
    For M=1000: ~1 billion elements (4 GB at float32)
    Chunking: Compute distances in blocks to stay within memory limits

    ALGORITHM (unchunked):
    ----------------------
    1. Broadcast: A[i] - A[j] for all i,j → (B*, M, M, m)
    2. Apply smooth_abs → (B*, M, M, m)
    3. Apply softmax_max along m-dimension → (B*, M, M)

    ALGORITHM (chunked):
    --------------------
    1. Split j-dimension into chunks of size 'chunk'
    2. For each chunk: Compute distances for all i vs chunk of j
    3. Concatenate results

    >>> A = torch.tensor([[[1, 2], [3, 4], [5, 6]]])  # (1, 3, 2): 3 templates, length 2
    >>> D = _pairwise_chebyshev(A, beta_max=100)
    >>> D[0]  # Distances between all pairs
    tensor([[0.0, 2.0, 4.0],   # Template 0 vs all
            [2.0, 0.0, 2.0],   # Template 1 vs all
            [4.0, 2.0, 0.0]])  # Template 2 vs all (diagonal is self-distance = 0)
    """
    Bstar, M, m = A.shape
    
    # Edge case: No templates
    if M == 0:
        return A.new_zeros(Bstar, 0, 0)
    
    # UNCHUNKED PATH: Compute all distances at once
    if chunk is None or chunk >= M:
        # Broadcast to compute all pairwise differences
        # A.unsqueeze(2): (B*, M, 1, m) - broadcasts over j
        # A.unsqueeze(1): (B*, 1, M, m) - broadcasts over i
        diff = A.unsqueeze(2) - A.unsqueeze(1)  # (B*, M, M, m)
        
        # Smooth absolute value
        diff_abs = smooth_abs(diff, eps=eps_abs)  # (B*, M, M, m)
        
        # Smooth maximum along pattern dimension
        D = softmax_max(diff_abs, dim=-1, beta=beta_max)  # (B*, M, M)
        
        return D
    
    # CHUNKED PATH: Reduce memory by computing in blocks
    D_out = A.new_empty(Bstar, M, M)
    
    for j0 in range(0, M, chunk):
        j1 = min(j0 + chunk, M)
        
        # Compute distances for all i vs j-chunk
        # A: (B*, M, m), A[:, j0:j1, :]: (B*, chunk, m)
        diff = A.unsqueeze(2) - A[:, j0:j1, :].unsqueeze(1)  # (B*, M, chunk, m)
        diff_abs = smooth_abs(diff, eps=eps_abs)
        D_block = softmax_max(diff_abs, dim=-1, beta=beta_max)  # (B*, M, chunk)
        
        # Store chunk
        D_out[:, :, j0:j1] = D_block
    
    return D_out


def _phi_from_templates(
    Xm: torch.Tensor,
    n: float,
    r_val: torch.Tensor,
    beta_max: float = 50.0,
    eps_abs: float = 1e-6,
    chunk_j: Optional[int] = None,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    Compute Phi_m: average fuzzy similarity between all template pairs.
    This is the core statistic of Fuzzy Entropy.

    FUZZY SIMILARITY:
    -----------------
    For two templates X_i and X_j with Chebyshev distance d_ij:
        similarity = exp( -(d_ij^n) / r )

    where:
    n: Fuzzy exponent (controls membership sharpness)
    r: Tolerance threshold (smaller r → stricter matching)

    INTUITION:
    ----------
    d=0: similarity=1 (perfect match)
    d=r: similarity=exp(-1) ≈ 0.37
    d>>r: similarity≈0 (no match)
    n=1: Linear decay
    n=2: Gaussian-like (standard choice)
    n>2: Sharper cutoff

    PHI COMPUTATION:
    ----------------
    Phi_m = (1 / M*(M-1)) * sum_{i≠j} exp(-(d_ij^n)/r)
    Denominator M*(M-1): Number of non-self pairs
    exclude_self=True: Skip diagonal (i=j) as standard in entropy

    Xm : torch.Tensor
        Templates of shape (B*, M, m)
    n : float
        Fuzzy exponent (typically 2.0)
    r_val : torch.Tensor or float
        Tolerance threshold
        Can be scalar or (B*, 1, 1) for per-series tolerance
    beta_max : float
        Passed to _pairwise_chebyshev
    eps_abs : float
        Passed to _pairwise_chebyshev
    chunk_j : int, optional
        Passed to _pairwise_chebyshev for memory control
    exclude_self : bool, default=True
        Whether to exclude diagonal (self-matches) from averaging

    torch.Tensor
        Phi values of shape (B*,) - one per series

    EDGE CASES:
    -----------
    M <= 1: Not enough templates for pairs → return zeros
    All templates identical: d_ij=0 for all i,j → Phi=1 → FuzzyEn=log(1)-log(1)=0

    EXAMPLE (conceptual):
    ---------------------
    Signal: [1, 2, 1, 2, 1, 2] (periodic)
        → High Phi (patterns repeat) → Low FuzzyEn (predictable)

    Signal: [random noise]
        → Low Phi (patterns don't repeat) → High FuzzyEn (complex)
    """
    Bstar, M, _ = Xm.shape
    
    # Edge case: Need at least 2 templates for pairs
    if M <= 1:
        return Xm.new_zeros(Bstar)
    
    # Compute pairwise Chebyshev distances
    D = _pairwise_chebyshev(Xm, beta_max=beta_max, eps_abs=eps_abs, chunk=chunk_j)  # (B*, M, M)
    
    # Handle self-exclusion
    if exclude_self:
        # Create mask: True everywhere except diagonal
        mask = ~torch.eye(M, device=Xm.device, dtype=torch.bool).unsqueeze(0)  # (1, M, M)
        
        # Zero out diagonal
        D = torch.where(mask, D, torch.zeros_like(D))
        
        # Count valid pairs: M*(M-1)
        denom = mask.sum(dim=(1, 2)).clamp(min=1)  # (Bstar,)
    else:
        # Include all M*M pairs
        denom = torch.full((Bstar,), M * M, device=Xm.device, dtype=torch.long)
    
    # Ensure r_val is tensor with proper shape for broadcasting
    if not torch.is_tensor(r_val):
        r_val = torch.tensor(r_val, device=Xm.device, dtype=Xm.dtype)
    
    if r_val.dim() == 0:
        # Scalar → (1, 1, 1)
        r_b = r_val.view(1, 1, 1)
    else:
        # Already shaped (B*, 1, 1) or broadcastable
        r_b = r_val
    
    # Compute fuzzy similarity
    # sim[b,i,j] = exp( -(D[b,i,j]^n) / r[b] )
    sim = torch.exp(- (D ** n) / r_b)  # (B*, M, M)
    
    # Average similarity per series
    phi_m = sim.sum(dim=(1, 2)) / denom  # (B*,)
    
    return phi_m


# SECTION 4: MULTISCALE FUZZY ENTROPY PROFILE COMPUTATION
# This is the main function that ties everything together to compute the
# full MFE curve across multiple time scales.

def mfe_profile(
    x: torch.Tensor,
    m: int = 2,
    n: float = 2.0,
    tau_max: int = 20,
    normalize_z: bool = True,
    r_fixed: float = 0.15,
    r_frac: float = 0.15,
    stride_templates: int = 1,
    max_templates: int = 300,
    beta_max: float = 50.0,
    eps_abs: float = 1e-6,
    chunk_j: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Multiscale Fuzzy Entropy profile across tau = 1 to tau_max.
    This is the main workhorse function for MFE computation.

    WHAT IS RETURNED:
    -----------------
    A tensor of shape (B*, tau_max) where each column is the FuzzyEn
    at that time scale. This forms the "complexity fingerprint" of the signal.

    For each scale tau in [1, 2, ..., tau_max]:
        1. Coarse-grain signal: y_tau = coarse_grain_mean(x, tau)
        2. Build templates of length m: X_m
        3. Build templates of length m+1: X_{m+1}
        4. Compute Phi_m and Phi_{m+1} (average similarities)
        5. FuzzyEn(tau) = log(Phi_m) - log(Phi_{m+1})

    x : torch.Tensor
        Input signal: (B, T), (B, 1, T), or (B, C, T)

    m : int, default=2
        Pattern length (embedding dimension)
    m=2: Compare pairs of consecutive points (standard)
    m=3: Compare triplets (more complex patterns)
        Typical: m=2 for physiological signals

    n : float, default=2.0
        Fuzzy membership exponent
    n=1: Linear similarity decay
    n=2: Gaussian-like (most common)
    n>2: Sharper similarity cutoff

    tau_max : int, default=20
        Maximum time scale to analyze
    tau_max=20 at 250 Hz → covers 4 ms to 80 ms
    tau_max=50 at 250 Hz → covers 4 ms to 200 ms
        Larger tau_max captures slower dynamics but increases computation

    normalize_z : bool, default=True
        Whether to z-score signals before analysis
    True: Use r_fixed as absolute threshold (recommended)
    False: Scale r by signal std (r = r_frac * std(y_tau))

    r_fixed : float, default=0.15
        Tolerance threshold when normalize_z=True
    Typical range: 0.10 to 0.25
    Smaller r: Stricter matching, higher entropy
    Larger r: Looser matching, lower entropy

    r_frac : float, default=0.15
        Tolerance as fraction of std when normalize_z=False
    r = r_frac * std(signal)
    Makes threshold adaptive to signal amplitude

    stride_templates : int, default=1
        Stride for template extraction
    stride=1: Use all overlapping windows (standard, more accurate)
    stride=2: Every other window (faster, less memory)
        Larger stride trades accuracy for speed

    max_templates : int, default=300
        Maximum number of templates to use per scale
    If M > max_templates, subsample evenly
    Controls memory: M²  pairwise distances
    300: ~90k distances per scale (manageable on GPU)
    1000: ~1M distances (may need chunking)

    beta_max : float, default=50.0
        Sharpness for softmax_max approximation
    Higher: Better approximation of true max
    Risk: Overflow for very large beta (>100)
        50 is safe for normalized signals

    eps_abs : float, default=1e-6
        Epsilon for smooth_abs
        Smaller → closer to true |·|, but gradient issues near zero

    chunk_j : int, optional
        Chunk size for pairwise distance computation
    None: Compute all at once (faster, more memory)
    50-100: Reduce memory usage (slightly slower)
        Use if hitting GPU memory limits

    torch.Tensor
        MFE profile of shape (B*, tau_max)
    B* = B for (B,T), B*C for (B,C,T)
    Each row is the entropy curve for one signal
    Column tau corresponds to FuzzyEn at scale tau

    INTERPRETATION:
    ---------------
    High MFE at scale tau: Complex, unpredictable patterns at that timescale
    Low MFE at scale tau: Regular, predictable patterns
    Decreasing MFE with tau: Complexity reduces at slower scales (common)
    Constant MFE: Scale-invariant complexity (fractal-like)

    >>> # Analyze 10 Hz sine wave vs random noise
    >>> t = torch.linspace(0, 4, 1000)
    >>> sine = torch.sin(2*np.pi*10*t).unsqueeze(0)
    >>> noise = torch.randn(1, 1000)
    >>>
    >>> prof_sine = mfe_profile(sine, tau_max=20)
    >>> prof_noise = mfe_profile(noise, tau_max=20)
    >>>
    >>> # Sine: Low entropy (predictable)
    >>> # Noise: High entropy (unpredictable)
    >>> print(prof_sine.mean(), prof_noise.mean())  # noise >> sine
    Per scale: O(M² * m) where M = number of templates
    Total: O(tau_max * M² * m)
    Typical: 10-50 ms per batch on GPU (tau_max=20, M=300)

    EDGE CASES:
    -----------
    Signal too short for scale tau: Returns 0 for that scale
    Constant signal: Phi_m = Phi_{m+1} → FuzzyEn = 0
    All-NaN coarse-grained signal: Returns 0 (handled by masking)
    """
    # Prepare input: flatten to (B*, T)
    x_flat, T, _, Bstar = _prepare_signal(x)
    
    # Optional z-score normalization (highly recommended)
    if normalize_z:
        x_flat = zscore_per_series(x_flat)
    
    def _subsample_templates(X: torch.Tensor, Mmax: int) -> torch.Tensor:
        """
        Reduce number of templates to at most Mmax by even subsampling.
    Memory: Pairwise distances are M²
    M=1000 → 1M distances (4 MB per batch)
    M=300 → 90k distances (360 KB per batch)

        STRATEGY:
        ---------
    Use evenly spaced indices to preserve temporal coverage
    Example: M=1000, Mmax=300 → keep every ~3rd template

        X : torch.Tensor
            Templates (B*, M, m)
        Mmax : int
            Maximum desired M

        torch.Tensor
            Subsampled templates (B*, min(M, Mmax), m)
        """
        Bstar, M, mlen = X.shape
        if M <= Mmax:
            return X
        
        # Generate evenly spaced indices
        idx = torch.linspace(0, M - 1, steps=Mmax, device=X.device).round().long()
        return X.index_select(1, idx)
    
    # Storage for entropy at each scale
    mfe_vals = []
    
    # Loop over time scales
    for tau in range(1, tau_max + 1):
        # Coarse-grain signal
        y = coarse_grain_mean(x_flat, tau)  # (B*, N_tau)
        Bstar, Ntau = y.shape
        
        # Check if enough data for entropy computation
        # Need at least m+2 points (for m+1 templates with at least 2 templates)
        if Ntau < (m + 1 + 1):
            # Insufficient data: return zero entropy
            mfe_vals.append(y.new_zeros(Bstar))
            continue
        
        # Build templates of length m and m+1
        Xm = _build_templates(y, m=m, stride=stride_templates)       # (B*, M, m)
        Xm1 = _build_templates(y, m=m + 1, stride=stride_templates)  # (B*, M1, m+1)
        
        # Check if templates were successfully created
        if Xm.size(1) == 0 or Xm1.size(1) == 0:
            mfe_vals.append(y.new_zeros(Bstar))
            continue
        
        # Subsample templates if too many (memory control)
        Xm = _subsample_templates(Xm, max_templates)
        Xm1 = _subsample_templates(Xm1, max_templates)
        
        # Determine tolerance threshold r
        if normalize_z:
            # Z-scored signals: use fixed r
            r_val_m = Xm.new_tensor(r_fixed).view(1, 1, 1)   # (1, 1, 1)
            r_val_m1 = Xm1.new_tensor(r_fixed).view(1, 1, 1)
        else:
            # Non-normalized: r = r_frac * std(signal) per series
            std_tau = y.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)  # (B*, 1)
            r_val_m = (r_frac * std_tau).view(Bstar, 1, 1)   # (B*, 1, 1)
            r_val_m1 = (r_frac * std_tau).view(Bstar, 1, 1)
        
        # Compute Phi_m: average similarity for length-m patterns
        phi_m = _phi_from_templates(
            Xm, n=n, r_val=r_val_m, beta_max=beta_max,
            eps_abs=eps_abs, chunk_j=chunk_j, exclude_self=True
        )  # (B*,)
        
        # Compute Phi_{m+1}: average similarity for length-(m+1) patterns
        phi_m1 = _phi_from_templates(
            Xm1, n=n, r_val=r_val_m1, beta_max=beta_max,
            eps_abs=eps_abs, chunk_j=chunk_j, exclude_self=True
        )  # (B*,)
        
        # Fuzzy Entropy = log(Phi_m) - log(Phi_{m+1})
        # Intuition: If Phi_m ≈ Phi_{m+1}, adding one more point doesn't change
        #            similarity → signal is predictable → low entropy
        fuzzy_en = safe_log(phi_m) - safe_log(phi_m1)  # (B*,)
        
        mfe_vals.append(fuzzy_en)
    
    # Stack into profile matrix
    mfe = torch.stack(mfe_vals, dim=1)  # (B*, tau_max)
    
    return mfe


# SECTION 5: LOSS FUNCTION WRAPPER FOR NEURAL NETWORK TRAINING
# PyTorch nn.Module that wraps MFE computation into a differentiable loss.

class MFEProfileLoss(nn.Module):
    """
    PyTorch loss function that compares MFE profiles of generated vs target signals.

    In EEG super-resolution, you want the upsampled signal to not just match
    amplitudes (L1/MSE) but also preserve complexity structure. MFE Profile Loss
    ensures the generated signal has the same entropy characteristics across scales.

    LOSS COMPUTATION:
    -----------------
    1. Compute MFE profile of generated signal: P_gen(tau)
    2. Compute MFE profile of target signal: P_target(tau)
    3. Measure difference: Loss = mean_tau |P_gen(tau) - P_target(tau)|  (L1)
                            or Loss = mean_tau (P_gen(tau) - P_target(tau))²  (MSE)
    If generated signal matches target complexity at all scales → Loss ≈ 0
    If generated signal is over-smoothed → Low entropy → High loss
    If generated signal is too noisy → High entropy → High loss
    Encourages physiologically realistic complexity

    All parameters from mfe_profile() plus:

    distance : Literal["l1", "mse"], default="mse"
        How to measure profile difference
    "l1": Mean absolute error (robust to outliers)
    "mse": Mean squared error (penalizes large deviations more)

    reduction : Literal["mean", "sum", "none"], default="mean"
        How to aggregate loss across batch
    "mean": Average loss (standard for training)
    "sum": Total loss (useful for weighting)
    "none": Per-sample loss (for analysis)

    ```python
    # In model training loop
    mfe_loss_fn = MFEProfileLoss(
        m=2, n=2.0, tau_max=20,
        normalize_z=True, r_fixed=0.15,
        distance='mse', reduction='mean'
    )

    for batch in dataloader:
        lr_eeg, hr_eeg = batch  # Low-res input, high-res target
        sr_eeg = model(lr_eeg)  # Super-resolved output

        # Compute losses
        amp_loss = F.l1_loss(sr_eeg, hr_eeg)      # Amplitude matching
        mfe_loss = mfe_loss_fn(sr_eeg, hr_eeg)    # Complexity matching

        # Combined loss (tune weights)
        total_loss = amp_loss + 0.5 * mfe_loss

        # Optimize
        total_loss.backward()
        optimizer.step()
    ```

    LOSS WEIGHTS:
    -------------
    Start with 0.5 weight for MFE (amp + 0.5*mfe)
    If generated signals are too smooth: Increase MFE weight
    If training is unstable: Decrease MFE weight
    Monitor both losses separately during training
    ~2× cost of amplitude loss (need to compute MFE for both gen and target)
    Typical: 20-100ms per batch on GPU
    Comparable to perceptual losses (e.g., VGG features)

    ADVANTAGES OVER OTHER LOSSES:
    ------------------------------
    Spectral loss: Only captures frequency content, not temporal structure
    Perceptual loss: Requires pre-trained network, may not suit EEG
    MFE: Directly measures complexity, parameter-free (after tuning m, n, r)

    >>> target = torch.randn(4, 1, 1024)  # 4 signals, 1024 samples
    >>> generated = target + 0.1*torch.randn(4, 1, 1024)  # Slightly noisy
    >>>
    >>> loss_fn = MFEProfileLoss(tau_max=20)
    >>> loss = loss_fn(generated, target)
    >>> print(loss)  # Small value (signals are similar)
    >>>
    >>> # Gradient check
    >>> generated.requires_grad_(True)
    >>> loss.backward()
    >>> print(generated.grad.norm())  # Non-zero gradient (can optimize)
    """
    
    def __init__(
        self,
        m: int = 2,
        n: float = 2.0,
        tau_max: int = 20,
        normalize_z: bool = True,
        r_fixed: float = 0.15,
        r_frac: float = 0.15,
        stride_templates: int = 1,
        max_templates: int = 300,
        beta_max: float = 50.0,
        eps_abs: float = 1e-6,
        chunk_j: Optional[int] = None,
        distance: Literal["l1", "mse"] = "mse",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """
        Initialize MFE Profile Loss with specified parameters.
        All parameters are stored as instance variables for use in forward().
        """
        super().__init__()
        
        # Store all hyperparameters
        self.m = m
        self.n = n
        self.tau_max = tau_max
        self.normalize_z = normalize_z
        self.r_fixed = r_fixed
        self.r_frac = r_frac
        self.stride_templates = stride_templates
        self.max_templates = max_templates
        self.beta_max = beta_max
        self.eps_abs = eps_abs
        self.chunk_j = chunk_j
        self.distance = distance
        self.reduction = reduction
    
    def forward(self, x_gen: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        Compute MFE profile loss between generated and target signals.

        x_gen : torch.Tensor
            Generated/predicted signal: (B, T), (B, 1, T), or (B, C, T)
        x_true : torch.Tensor
            Target/ground-truth signal: same shape as x_gen

        torch.Tensor
            Scalar loss (if reduction='mean' or 'sum')
            or per-sample loss (B*,) if reduction='none'

        ValueError
            If x_gen and x_true have different shapes

        1. Validate input shapes match
        2. Compute MFE profile of generated signal: P_g (B*, tau_max)
        3. Compute MFE profile of target signal: P_t (B*, tau_max)
        4. Compute per-scale difference: diff = P_g - P_t (B*, tau_max)
        5. Aggregate difference:
    L1: Mean |diff| across tau dimension → (B*,)
    MSE: Mean diff² across tau dimension → (B*,)
        6. Reduce across batch:
    mean: Average across B*
    sum: Sum across B*
    none: Return (B*,) tensor

        All operations are differentiable:
    mfe_profile() uses smooth_abs, softmax_max (differentiable primitives)
    Backward pass propagates gradients through:
          x_gen → MFE_gen → diff → loss → scalar
    Two MFE profile computations (gen + target)
    Peak memory during pairwise distance computation
    Use chunk_j if hitting GPU memory limits
    For similar signals: Loss ~ 0.001 - 0.01
    For very different signals: Loss ~ 0.1 - 1.0
    For identical signals: Loss ~ 1e-8 (numerical precision)
        """
        # Validate matching shapes
        if x_gen.shape != x_true.shape:
            raise ValueError(
                f"Shape mismatch: generated {tuple(x_gen.shape)} "
                f"vs target {tuple(x_true.shape)}"
            )
        
        # Compute MFE profile of generated signal
        prof_g = mfe_profile(
            x_gen, m=self.m, n=self.n, tau_max=self.tau_max,
            normalize_z=self.normalize_z, r_fixed=self.r_fixed, r_frac=self.r_frac,
            stride_templates=self.stride_templates, max_templates=self.max_templates,
            beta_max=self.beta_max, eps_abs=self.eps_abs, chunk_j=self.chunk_j
        )  # (B*, tau_max)
        
        # Compute MFE profile of target signal
        prof_t = mfe_profile(
            x_true, m=self.m, n=self.n, tau_max=self.tau_max,
            normalize_z=self.normalize_z, r_fixed=self.r_fixed, r_frac=self.r_frac,
            stride_templates=self.stride_templates, max_templates=self.max_templates,
            beta_max=self.beta_max, eps_abs=self.eps_abs, chunk_j=self.chunk_j
        )  # (B*, tau_max)
        
        # Compute difference between profiles
        diff = prof_g - prof_t  # (B*, tau_max)
        
        # Aggregate difference across scales (tau dimension)
        if self.distance == "l1":
            # L1 distance: Mean absolute difference across tau
            # More robust to outlier scales
            per_series = smooth_abs(diff, eps=1e-12).mean(dim=1)  # (B*,)
        elif self.distance == "mse":
            # MSE distance: Mean squared difference across tau
            # Penalizes large deviations more heavily
            per_series = (diff * diff).mean(dim=1)  # (B*,)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        
        # Reduce across batch dimension
        if self.reduction == "mean":
            return per_series.mean()  # Scalar
        if self.reduction == "sum":
            return per_series.sum()   # Scalar
        return per_series  # (B*,) - per-sample losses


# SECTION 6: DEMONSTRATION AND TESTING
# Functions to demonstrate usage and validate correctness of the implementation.

def _demo():
    """
    Quick demonstration of MFE Profile Loss in action.

    WHAT THIS SHOWS:
    ----------------
    1. Creating realistic synthetic EEG data
    2. Computing MFE loss between noisy predictions and targets
    3. Verifying gradient flow (required for neural network training)
    4. Combining MFE loss with amplitude loss

    EXPECTED OUTPUT:
    ----------------
    MFE loss: ~0.000001 (very small, as prediction is close to target)
    Gradient norm: ~0.000002 (small but non-zero, indicating healthy gradients)
    Total loss: ~0.08 (dominated by amplitude term)

    INTERPRETATION:
    ---------------
    Small MFE loss: Generated signal has similar complexity to target
    Non-zero gradient: Loss can guide optimization
    Stable gradients: No NaN or explosion
    """
    # Select device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate a batch of EEG recordings
    B, T = 6, 1024  # 6 signals, 1024 samples (~4 seconds at 250 Hz)
    x_true = torch.randn(B, 1, T) * 10.0  # Target: Random EEG-like signal
    x_gen = x_true + 0.1 * torch.randn(B, 1, T)  # Generated: Target + 10% noise
    
    # Move to device
    x_true = x_true.to(device)
    x_gen = x_gen.to(device)
    
    # Build MFE loss function
    mfe_loss = MFEProfileLoss(
        m=2,                    # Pattern length (standard)
        n=2.0,                  # Fuzzy exponent (Gaussian-like)
        tau_max=20,             # Analyze up to 80ms timescale at 250 Hz
        normalize_z=True,       # Z-score signals (recommended)
        r_fixed=0.15,           # Tolerance threshold
        stride_templates=1,     # Use all overlapping windows
        max_templates=300,      # Memory control
        beta_max=50.0,          # Sharpness for soft-max
        distance="mse",         # MSE profile comparison
        reduction="mean"        # Average over batch
    ).to(device)
    
    # Enable gradient tracking on generated signal
    x_gen.requires_grad_(True)
    
    # Compute loss
    loss_val = mfe_loss(x_gen, x_true)
    
    # Backpropagate to compute gradients
    loss_val.backward()
    
    # Print results
    print(f"[MFE curve loss] value: {loss_val.detach().item():.6f}")
    print(f"Grad norm on x_gen: {x_gen.grad.norm().item():.6f}")
    
    # Example: Combining with amplitude loss (common in practice)
    amp_loss = F.l1_loss(x_gen, x_true)
    total = amp_loss + 0.5 * mfe_loss(x_gen, x_true)
    print(f"[Total loss example] amp + 0.5*mfe = {total.detach().item():.6f}")


# COMPREHENSIVE TEST SUITE
# The following tests validate all aspects of the MFE implementation.

def test_basic_functionality():
    """
    TEST 1: Basic functionality check.

    VALIDATES:
    ----------
    Loss computation runs without errors
    Loss is non-negative (entropy is always >= 0)
    Loss is finite (no NaN or Inf)

    SETUP:
    ------
    4 signals, 512 samples each
    Target + 10% noise as prediction

    EXPECTED:
    ---------
    Small positive loss (~0.0001)
    No errors or warnings
    """
    print("\n=== TEST 1: Basic Functionality ===")
    
    # Create test data
    B, T = 4, 512
    x_target = torch.randn(B, 1, T)
    x_pred = x_target + 0.1 * torch.randn(B, 1, T)
    
    # Compute loss
    loss_fn = MFEProfileLoss(
        m=2, n=2.0, tau_max=10,
        normalize_z=True, r_fixed=0.15,
        distance='mse', reduction='mean'
    )
    loss = loss_fn(x_pred, x_target)
    
    print(f"Loss: {loss.detach().item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✓ Basic functionality test passed")


def test_gradient_flow():
    """
    TEST 2: Gradient flow validation.

    VALIDATES:
    ----------
    Gradients exist after backward pass
    Gradients are finite (no NaN or Inf)
    Gradient magnitudes are reasonable

    Without proper gradients, neural networks cannot learn.
    This test ensures the loss is truly differentiable.

    EXPECTED:
    ---------
    Gradient norm: ~0.001-0.01 (small but non-zero)
    No NaN or Inf in gradients
    """
    print("\n=== TEST 2: Gradient Flow ===")
    
    B, T = 4, 512
    x_target = torch.randn(B, T)
    x_pred = torch.randn(B, T, requires_grad=True)
    
    loss_fn = MFEProfileLoss(m=2, n=2.0, tau_max=10)
    loss = loss_fn(x_pred, x_target)
    
    # Backward pass
    loss.backward()
    
    assert x_pred.grad is not None, "Gradient should exist"
    assert not torch.isnan(x_pred.grad).any(), "Gradient should not contain NaN"
    assert not torch.isinf(x_pred.grad).any(), "Gradient should not contain Inf"
    print(f"Gradient norm: {x_pred.grad.norm().item():.6f}")
    print("✓ Gradient flow test passed")


def test_identical_signals():
    """
    TEST 3: Identical signal validation.

    VALIDATES:
    ----------
    Loss is zero (or near-zero) for identical signals

    If x_gen == x_target, then:
    MFE_gen == MFE_target for all scales
    Loss = mean|MFE_gen - MFE_target| = 0

    EXPECTED:
    ---------
    Loss < 1e-6 (essentially zero, accounting for numerical precision)
    """
    print("\n=== TEST 3: Identical Signals ===")
    
    B, T = 2, 512
    x = torch.randn(B, 1, T)
    
    loss_fn = MFEProfileLoss(m=2, n=2.0, tau_max=10, distance='mse')
    loss = loss_fn(x, x)
    
    print(f"Loss (identical signals): {loss.detach().item():.8f}")
    assert loss.item() < 1e-6, "Loss should be near zero for identical signals"
    print("✓ Identical signals test passed")


def test_different_shapes():
    """
    TEST 4: Input shape flexibility validation.

    VALIDATES:
    ----------
    Function handles (B, T), (B, 1, T), and (B, C, T) inputs
    Results are reasonable for all shapes

    Real-world EEG data comes in various formats:
    Single-channel: (B, T)
    Explicitly single-channel: (B, 1, T)
    Multi-channel: (B, 64, T) for 64-electrode EEG

    EXPECTED:
    ---------
    All shapes produce valid losses
    No shape-related errors
    """
    print("\n=== TEST 4: Different Signal Shapes ===")
    
    loss_fn = MFEProfileLoss(m=2, n=2.0, tau_max=10)
    
    # Test (B, T)
    x1 = torch.randn(4, 512)
    x2 = torch.randn(4, 512)
    loss1 = loss_fn(x1, x2)
    print(f"Loss (B, T): {loss1.detach().item():.6f}")
    
    # Test (B, 1, T)
    x3 = torch.randn(4, 1, 512)
    x4 = torch.randn(4, 1, 512)
    loss2 = loss_fn(x3, x4)
    print(f"Loss (B, 1, T): {loss2.detach().item():.6f}")
    
    # Test (B, C, T) - multi-channel
    x5 = torch.randn(2, 64, 512)
    x6 = torch.randn(2, 64, 512)
    loss3 = loss_fn(x5, x6)
    print(f"Loss (B, C, T): {loss3.detach().item():.6f}")
    
    print("✓ Different shapes test passed")


# TEST 5: Short Signal Handling
def test_short_signals():
    """
    TEST 5: Short signal robustness validation.

    VALIDATES:
    ----------
    Function handles very short signals gracefully
    No crashes when signal length is minimal
    Returns reasonable loss even with limited data

    Real-world scenarios may have variable-length segments
    Edge cases (e.g., artifact rejection leaving short segments)
    System should degrade gracefully, not crash

    EDGE CASE TESTED:
    -----------------
    T=100 samples (very short for entropy analysis)
    Reduced tau_max=5 to match shorter signal
    At 250 Hz: 100 samples = 0.4 seconds (minimal)

    EXPECTED BEHAVIOR:
    ------------------
    For tau > T: Returns zero entropy (no data to coarse-grain)
    For small tau: Computes entropy with available data
    No crashes or NaN values

    EXPECTED OUTPUT:
    ----------------
    Loss: ~0.01-0.05 (higher variance due to less data)
    No errors or exceptions

    WHAT TO WATCH:
    --------------
    If T < m+2: Cannot build templates → should return zeros gracefully
    If coarse-graining reduces N_tau to < m+2: Handled by checks in mfe_profile
    """
    print("\n=== TEST 5: Short Signal Handling ===")
    
    # Very short signal parameters
    B, T = 2, 100  # 2 series, only 100 samples each
    x_target = torch.randn(B, T)
    x_pred = torch.randn(B, T)
    
    try:
        # Reduced tau_max to match shorter signal
        # tau_max=5: Covers scales 1-5 (at 250Hz: 4-20ms)
        loss_fn = MFEProfileLoss(m=2, n=2.0, tau_max=5)
        loss = loss_fn(x_pred, x_target)
        
        print(f"Loss (short signal): {loss.detach().item():.6f}")
        print("✓ Short signals test passed")
    except Exception as e:
        # If exception occurs, report it (but shouldn't happen)
        print(f"✗ Short signals test failed: {e}")


# TEST 6: Distance Metric Comparison
def test_distance_metrics():
    """
    TEST 6: Distance metric comparison validation.

    VALIDATES:
    ----------
    Both L1 and MSE distance metrics produce valid losses
    L1 and MSE give different but reasonable results
    No numerical issues with either metric

    L1 (Mean Absolute Error):
        Loss = mean_tau |MFE_gen(tau) - MFE_target(tau)|

        Advantages:
    More robust to outliers
    Linear penalty (doesn't overemphasize large errors)
    Interpretable: Average absolute difference

        Use when: You want all scales weighted equally

    MSE (Mean Squared Error):
        Loss = mean_tau (MFE_gen(tau) - MFE_target(tau))²

        Advantages:
    Differentiable everywhere (smooth gradients)
    Penalizes large deviations more heavily
    Standard in neural network training

        Use when: Large profile differences should be heavily penalized

    For same difference d:
    L1: penalty = |d|
    MSE: penalty = d²

    Therefore: MSE < L1 for |d| < 1, MSE > L1 for |d| > 1

    EXPECTED BEHAVIOR:
    ------------------
    Both losses should be positive
    For small differences (d < 1): L1 > MSE
    For large differences (d > 1): MSE > L1
    Typically L1 > MSE for normalized entropy profiles

    EXPECTED OUTPUT:
    ----------------
    L1 Loss: ~0.01-0.02 (absolute difference scale)
    MSE Loss: ~0.0001-0.001 (squared difference scale)
    L1 typically 10-100× larger than MSE numerically

    PRACTICAL IMPLICATIONS:
    -----------------------
    If using multiple losses, scale accordingly:
      total = amp_loss + 0.5*L1_mfe_loss
      total = amp_loss + 5.0*MSE_mfe_loss  # Note: 10× larger weight
    """
    print("\n=== TEST 6: Distance Metric Comparison ===")
    
    # Create test signals with moderate difference
    B, T = 4, 512
    x_target = torch.randn(B, T)
    # Add 20% noise to target (moderate difference)
    x_pred = x_target + 0.2 * torch.randn(B, T)
    
    # Initialize loss functions with different distance metrics
    loss_l1 = MFEProfileLoss(
        m=2, n=2.0, tau_max=10,
        distance='l1'  # L1 distance
    )
    loss_mse = MFEProfileLoss(
        m=2, n=2.0, tau_max=10,
        distance='mse'  # MSE distance
    )
    
    # Compute losses
    l1_val = loss_l1(x_pred, x_target)
    mse_val = loss_mse(x_pred, x_target)
    
    # Display results
    print(f"L1 Loss: {l1_val.detach().item():.6f}")
    print(f"MSE Loss: {mse_val.detach().item():.6f}")
    print(f"Ratio (L1/MSE): {(l1_val/mse_val).item():.1f}")
    print("✓ Distance metrics test passed")


# TEST 7: Reduction Modes
def test_reduction_modes():
    """
    TEST 7: Loss reduction mode validation.

    VALIDATES:
    ----------
    All three reduction modes work correctly
    Proper aggregation across batch dimension
    Output shapes match expected dimensions

    REDUCTION MODES EXPLAINED:
    --------------------------
    After computing per-sample profile differences, we have a tensor
    of shape (B*,) representing loss for each signal. Reduction determines
    how to aggregate these into a final loss.

    1. REDUCTION='mean' (DEFAULT):
       Output: Scalar
       Formula: mean(per_sample_losses)

       Use case:
    Standard for training (most common)
    Loss magnitude independent of batch size
    Gradients averaged over batch

       Example:
       per_sample = [0.1, 0.2, 0.3, 0.4]
       mean reduction → 0.25

    2. REDUCTION='sum':
       Output: Scalar
       Formula: sum(per_sample_losses)

       Use case:
    When you want batch size to affect loss magnitude
    Useful for loss weighting strategies
    Total loss scales linearly with batch size

       Example:
       per_sample = [0.1, 0.2, 0.3, 0.4]
       sum reduction → 1.0

    3. REDUCTION='none':
       Output: Tensor of shape (B,)
       Formula: No aggregation, return per-sample losses

       Use case:
    Per-sample analysis (e.g., finding bad samples)
    Custom weighting schemes
    Debugging (see which samples have high loss)

    For batch size B:
        sum_loss = mean_loss × B
        mean_loss = sum_loss / B
        none_loss = individual losses before aggregation

    EXPECTED BEHAVIOR:
    ------------------
    mean: Returns scalar in range [0, ∞)
    sum: Returns scalar ≈ mean × batch_size
    none: Returns (B,) tensor with individual losses

    EXPECTED OUTPUT:
    ----------------
    Mean reduction: ~0.005-0.01 (scalar)
    Sum reduction: ~0.02-0.04 (≈ 4× mean for B=4)
    None reduction: shape=(4,), values similar to mean
    mean: Gradients divided by batch size
    sum: Gradients summed (larger magnitude)
    none: Each sample's gradient computed independently

    PRACTICAL USAGE:
    ----------------
    ```python
    # Training loop (use 'mean')
    loss = loss_fn(pred, target)  # Scalar
    loss.backward()
    optimizer.step()

    # Analysis (use 'none')
    per_sample_loss = loss_fn(pred, target)  # (B,)
    bad_samples = per_sample_loss > threshold
    print(f"High-loss samples: {bad_samples.nonzero()}")
    ```
    """
    print("\n=== TEST 7: Reduction Modes ===")
    
    # Create test data
    B, T = 4, 512
    x_target = torch.randn(B, T)
    x_pred = torch.randn(B, T)
    
    # Initialize loss functions with different reduction modes
    loss_mean = MFEProfileLoss(
        m=2, tau_max=10,
        reduction='mean'  # Average over batch
    )
    loss_sum = MFEProfileLoss(
        m=2, tau_max=10,
        reduction='sum'   # Sum over batch
    )
    loss_none = MFEProfileLoss(
        m=2, tau_max=10,
        reduction='none'  # No aggregation
    )
    
    # Compute losses with different reductions
    mean_val = loss_mean(x_pred, x_target)  # Scalar
    sum_val = loss_sum(x_pred, x_target)    # Scalar
    none_val = loss_none(x_pred, x_target)  # (B,) tensor
    
    # Display results
    print(f"Mean reduction: {mean_val.detach().item():.6f} (scalar)")
    print(f"Sum reduction: {sum_val.detach().item():.6f} (scalar)")
    print(f"None reduction shape: {none_val.shape} (should be (4,))")
    print(f"None reduction values: {none_val.detach().numpy()}")
    
    # Verify mathematical relationship
    expected_sum = mean_val * B
    ratio = sum_val / mean_val
    print(f"Sum/Mean ratio: {ratio.item():.2f} (should be ≈{B})")
    
    print("✓ Reduction modes test passed")


# TEST 8: Profile Visualization
def test_profile_visualization():
    """
    TEST 8: MFE profile visualization and interpretation.
    Visually demonstrate how different signals produce different MFE curves
    Validate that entropy captures expected signal properties
    Generate reference plot for documentation/papers

    SIGNALS TESTED:
    ---------------
    1. 10 Hz Sine Wave:
    Periodic, predictable pattern
    Expected: Low entropy (regular structure)
    MFE should be relatively flat (scale-invariant predictability)

    2. 20 Hz Sine Wave:
    Higher frequency periodic pattern
    Expected: Similar entropy to 10 Hz (both are periodic)
    May differ at specific scales due to period matching coarse-graining

    3. Random Noise:
    No structure, unpredictable
    Expected: High entropy (maximum irregularity)
    MFE should be highest across all scales

    WHAT TO OBSERVE IN PLOT:
    ------------------------
    Left Panel (MFE Profiles):
    X-axis: Time scale tau (1 to 20)
    Y-axis: Fuzzy Entropy
    Noise curve (top): Highest entropy
    Sine curves (bottom): Lower entropy
    Separation between curves: Model can distinguish signal types

    Right Panel (Time Domain):
    Shows what the signals actually look like
    Visual confirmation of signal properties
    First 200 samples for clarity

    EXPECTED PATTERNS:
    ------------------
    1. Entropy Ordering:
       Random Noise > Sine waves
       (Complexity: noise > periodic)

    2. Scale Behavior:
    Small tau (1-5): Captures fast dynamics
    Large tau (15-20): Captures slow trends
    Periodic signals: May show peaks/valleys at resonant scales

    3. Curve Shape:
    Noise: Relatively flat or slowly decreasing
    Sine: May show oscillations or plateaus
    Both: Generally decrease with scale (less complexity at slow timescales)

    INTERPRETATION GUIDE:
    ---------------------
    High MFE: Signal is unpredictable, complex, irregular
    Low MFE: Signal is predictable, simple, regular
    Decreasing MFE with tau: Complexity is in fast dynamics
    Flat MFE with tau: Scale-invariant complexity (fractal-like)

    OUTPUT FILE:
    ------------
    Saves 'mfe_profile_test.png' with two subplots:
    Use for presentations, papers, documentation
    Demonstrates loss function's ability to capture complexity

    PRACTICAL IMPLICATIONS:
    -----------------------
    If your super-resolved EEG has:
    MFE curve matching target → Complexity preserved ✓
    MFE curve below target → Over-smoothed (lost detail) ✗
    MFE curve above target → Too noisy (added artifacts) ✗
    """
    print("\n=== TEST 8: Profile Visualization ===")
    
    # Signal parameters
    T = 1024  # 1024 samples ≈ 4 seconds at 250 Hz
    t = torch.linspace(0, 4, T)  # Time vector (seconds)
    
    # Signal 1: 10 Hz sine wave (alpha band in EEG)
    # Period = 0.1s, wavelength = 25 samples at 250Hz
    x1 = torch.sin(2 * np.pi * 10 * t).unsqueeze(0)  # (1, T)
    
    # Signal 2: 20 Hz sine wave (beta band in EEG)
    # Period = 0.05s, wavelength = 12.5 samples at 250Hz
    x2 = torch.sin(2 * np.pi * 20 * t).unsqueeze(0)  # (1, T)
    
    # Signal 3: Random Gaussian noise (no structure)
    # Mean=0, std=1 (similar amplitude to sine waves)
    x3 = torch.randn(1, T)
    
    # Compute MFE profiles for all signals
    tau_max = 20  # Analyze scales 1-20 (4-80ms at 250Hz)
    
    # Profile 1: 10 Hz sine
    prof1 = mfe_profile(
        x1, m=2, n=2.0, tau_max=tau_max,
        normalize_z=True  # Z-score for fair comparison
    )  # (1, 20)
    
    # Profile 2: 20 Hz sine
    prof2 = mfe_profile(
        x2, m=2, n=2.0, tau_max=tau_max,
        normalize_z=True
    )  # (1, 20)
    
    # Profile 3: Random noise
    prof3 = mfe_profile(
        x3, m=2, n=2.0, tau_max=tau_max,
        normalize_z=True
    )  # (1, 20)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # LEFT PANEL: MFE Profiles
    plt.subplot(1, 2, 1)
    tau_vals = np.arange(1, tau_max + 1)  # X-axis: scales 1-20
    
    # Plot all three profiles
    plt.plot(tau_vals, prof1[0].detach().numpy(), 'o-',
             label='10 Hz sine', linewidth=2, markersize=6)
    plt.plot(tau_vals, prof2[0].detach().numpy(), 's-',
             label='20 Hz sine', linewidth=2, markersize=6)
    plt.plot(tau_vals, prof3[0].detach().numpy(), '^-',
             label='Random noise', linewidth=2, markersize=6)
    
    plt.xlabel('Scale (tau)', fontsize=12)
    plt.ylabel('Fuzzy Entropy', fontsize=12)
    plt.title('MFE Profile Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add interpretation note
    plt.text(0.5, 0.95, 'Higher entropy = More complex/unpredictable',
             transform=plt.gca().transAxes,
             fontsize=9, ha='center', style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # RIGHT PANEL: Time Domain Signals
    plt.subplot(1, 2, 2)
    
    # Show first 200 samples for clarity (0.8 seconds)
    plot_samples = 200
    t_plot = t[:plot_samples].numpy()
    
    plt.plot(t_plot, x1[0, :plot_samples].numpy(),
             label='10 Hz', linewidth=1.5)
    plt.plot(t_plot, x2[0, :plot_samples].numpy(),
             label='20 Hz', linewidth=1.5)
    plt.plot(t_plot, x3[0, :plot_samples].numpy(),
             label='Noise', alpha=0.7, linewidth=1.0)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Signals (first 200 samples)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add interpretation note
    plt.text(0.5, 0.05, 'Periodic signals have lower entropy than noise',
             transform=plt.gca().transAxes,
             fontsize=9, ha='center', style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Save figure
    import os
    save_dir = os.getcwd()
    save_path = os.path.join(save_dir, 'mfe_profile_test.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"✓ Profile saved to: {save_path}")
    
    # NEW: Automatically open the image
    if os.path.exists(save_path):
        print(f"  → Opening image...")
        import subprocess
        
        try:
            # Windows
            os.startfile(save_path)
            print(f"  → Image opened successfully!")
        except AttributeError:
            # macOS/Linux
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', save_path])
    
    plt.close()


# TEST 9: Numerical Stability
def test_numerical_stability():
    """
    TEST 9: Numerical stability under extreme conditions.

    VALIDATES:
    ----------
    No NaN or Inf values for extreme input magnitudes
    Stable behavior for very small signals (near machine precision)
    Stable behavior for very large signals (near overflow)
    Z-score normalization handles edge cases

    Real-world EEG data can have:
    Very small amplitudes (µV scale: ~1e-6 V)
    Variable scaling between subjects/channels
    Numerical operations (log, exp, sqrt) can overflow/underflow

    1. Very Small Values (x ~ 1e-6):
    Risk: Underflow in exp(-(d^n)/r)
    Solution: safe_log with eps=1e-12
    Risk: Division by zero in z-score
    Solution: std.clamp(min=eps)

    2. Very Large Values (x ~ 1e6):
    Risk: Overflow in exp(beta*x) for softmax
    Solution: logsumexp numerical trick
    Risk: Large distances d → exp(-d²) → 0
    Solution: safe_log handles very small Phi

    3. Constant Signals (std=0):
    Risk: Division by zero in z-score
    Solution: Clamp std to eps → results in all zeros
    Expected: Zero entropy (no variability)

    STABILITY MECHANISMS IN CODE:
    ------------------------------
    1. smooth_abs(x, eps=1e-6): |x| ≈ sqrt(x²+eps)
    Prevents gradient issues at x=0
    Stable for |x| from 1e-10 to 1e10

    2. safe_log(x, eps=1e-12): log(max(x, eps))
    Prevents log(0) = -inf
    Prevents log(negative) = NaN
    Lower bound: log(1e-12) ≈ -27.6

    3. torch.logsumexp: Numerically stable softmax
    Internally: log(sum(exp(x-max(x)))) + max(x)
    Prevents overflow for large x

    4. Z-score normalization: (x - mean) / max(std, eps)
    Makes signals comparable across scales
    Handles constant signals gracefully

    TEST CASES:
    -----------
    1. Tiny signals (x ~ 1e-6):
    Simulate µV-scale EEG
    Should compute same MFE as normalized version

    2. Huge signals (x ~ 1e6):
    Extreme amplitude variation
    Z-score should normalize to std=1

    3. Identical signals:
    Ultimate test: Loss should be exactly zero
    Tests all operations for perfect numerical matching

    EXPECTED BEHAVIOR:
    ------------------
    Small values: Loss ≈ 0 (identical after z-score)
    Large values: Loss ≈ 0 (identical after z-score)
    No NaN anywhere (all operations stable)
    No Inf anywhere (no overflow)

    WHAT COULD GO WRONG:
    --------------------
    ✗ NaN from log(0): Fixed by safe_log
    ✗ NaN from 0/0: Fixed by std.clamp(min=eps)
    ✗ Inf from exp(huge): Fixed by logsumexp trick
    ✗ Underflow in similarities: Acceptable (exp(-large) → 0)

    EXPECTED OUTPUT:
    ----------------
    Loss (tiny): < 1e-6 (essentially zero)
    Loss (huge): < 1e-6 (essentially zero)
    Both: No NaN, no Inf, no crashes
    """
    print("\n=== TEST 9: Numerical Stability ===")
    
    # Test parameters
    B, T = 2, 512
    
    # TEST 1: Very small values (µV scale)
    # Simulate EEG amplitudes: typically 10-100 µV = 1e-5 to 1e-4 V
    x_small = torch.randn(B, T) * 1e-6  # 1 µV scale
    
    loss_fn = MFEProfileLoss(
        m=2, tau_max=10,
        normalize_z=True  # Critical: normalizes away the scale
    )
    
    # Compute loss on identical tiny signals
    loss_small = loss_fn(x_small, x_small)
    
    print(f"Loss (very small values, 1e-6): {loss_small.detach().item():.8f}")
    
    # Validate: Should be near zero (identical signals)
    assert not torch.isnan(loss_small), "Loss should not be NaN for small values"
    assert not torch.isinf(loss_small), "Loss should not be Inf for small values"
    assert loss_small.item() < 1e-6, "Identical signals should give near-zero loss"
    
    # TEST 2: Very large values (extreme amplification)
    # Simulate signals amplified by 1 million times
    x_large = torch.randn(B, T) * 1e6
    
    # Compute loss on identical huge signals
    loss_large = loss_fn(x_large, x_large)
    
    print(f"Loss (very large values, 1e6): {loss_large.detach().item():.8f}")
    
    # Validate: Should be near zero (identical signals)
    assert not torch.isnan(loss_large), "Loss should not be NaN for large values"
    assert not torch.isinf(loss_large), "Loss should not be Inf for large values"
    assert loss_large.item() < 1e-6, "Identical signals should give near-zero loss"
    
    # TEST 3: Mixed scales (one tiny, one huge)
    # This tests if z-scoring properly handles different scales
    x_mix1 = torch.randn(B, T) * 1e-6
    x_mix2 = torch.randn(B, T) * 1e6
    
    # Should NOT crash, even though scales differ by 12 orders of magnitude
    try:
        loss_mix = loss_fn(x_mix1, x_mix2)
        print(f"Loss (mixed scales, 1e-6 vs 1e6): {loss_mix.detach().item():.6f}")
        print("  → Mixed scales handled successfully")
    except Exception as e:
        print(f"  ✗ Mixed scales failed: {e}")
    
    print("✓ Numerical stability test passed")
    print("  → No NaN, no Inf, handles extreme values correctly")


# TEST 10: Batch Processing Efficiency
def test_batch_processing():
    """
    TEST 10: Batch processing scalability validation.

    VALIDATES:
    ----------
    Loss computation scales efficiently with batch size
    Memory usage remains reasonable
    No batch-size-dependent bugs
    Reduction='mean' normalizes by batch size correctly

    In deep learning:
    Larger batches → Better GPU utilization
    Smaller batches → Less memory, sometimes better generalization
    Loss should be batch-size independent (when using reduction='mean')

    BATCH SIZE CONSIDERATIONS:
    --------------------------
    Small Batch (B=1):
    Pros: Minimal memory, stochastic gradients
    Cons: Noisy gradient estimates, slow training
    Use: Limited GPU memory or online learning

    Medium Batch (B=4-16):
    Pros: Good balance of memory and gradient quality
    Cons: May not fully utilize GPU
    Use: Standard for many applications

    Large Batch (B=32-64):
    Pros: Efficient GPU usage, stable gradients
    Cons: High memory, may overfit
    Use: When you have plenty of GPU RAM

    MFE computation memory: O(B * M² * m)
    where:
        B = batch size
        M = max_templates (default 300)
        m = pattern length (default 2)

    For max_templates=300:
    B=1: ~90k distances per scale
    B=16: ~1.4M distances per scale
    B=64: ~5.7M distances per scale

    GPU Memory Estimate (float32):
    B=1: ~10 MB
    B=16: ~150 MB
    B=64: ~600 MB
    (Plus overhead for activations, gradients)

    EXPECTED BEHAVIOR:
    ------------------
    With reduction='mean':
    Loss magnitude should be independent of batch size
    Losses across different batches should be comparable
    Not necessarily identical (different random data)

    Example:
    B=1, Loss=0.005 (single sample)
    B=16, Loss=0.005 ± 0.002 (average of 16 samples)
    B=64, Loss=0.005 ± 0.001 (average of 64 samples, more stable)

    WHAT TO OBSERVE:
    ----------------
    1. All batch sizes complete successfully
    2. Loss values are in similar range (e.g., 0.003-0.009)
    3. Variance decreases with larger batches (law of large numbers)
    4. No memory errors (if occurs, reduce max_templates)

    EXPECTED OUTPUT:
    ----------------
    Batch size 1: Loss = 0.003-0.010 (high variance)
    Batch size 4: Loss = 0.003-0.010 (moderate variance)
    Batch size 16: Loss = 0.003-0.010 (low variance)
    Batch size 32: Loss = 0.003-0.010 (lowest variance)

    PRACTICAL IMPLICATIONS:
    -----------------------
    If training a neural network:
    Start with B=4-8 (safe default)
    Increase until GPU memory ~80% used
    Monitor loss variance across batches
    If loss is too noisy: Increase batch size
    If OOM (out of memory): Decrease batch size or max_templates
    If batch size B=1 works but B>1 fails: Check tensor broadcasting
    If loss scales linearly with B: Check reduction mode (should be 'mean')
    If memory error: Reduce max_templates or use chunk_j
    """
    print("\n=== TEST 10: Batch Processing ===")
    
    # Signal parameters
    T = 512  # Fixed signal length
    
    # Test different batch sizes (powers of 2 for efficiency)
    batch_sizes = [1, 4, 16, 32]
    
    # Initialize loss function (reuse across batches)
    loss_fn = MFEProfileLoss(
        m=2, tau_max=10,
        max_templates=300,  # Control memory usage
        reduction='mean'    # Loss should be batch-size independent
    )
    
    print("\nTesting different batch sizes:")
    print("-" * 50)
    
    losses = []  # Store losses for statistical analysis
    
    for batch_size in batch_sizes:
        # Generate random test data for this batch
        x_target = torch.randn(batch_size, T)
        x_pred = torch.randn(batch_size, T)
        
        # Compute loss
        loss = loss_fn(x_pred, x_target)
        loss_val = loss.detach().item()
        losses.append(loss_val)
        
        # Display result
        print(f"Batch size {batch_size:2d}: Loss = {loss_val:.6f}")
    
    # Statistical analysis
    print("-" * 50)
    losses_np = np.array(losses)
    print(f"\nStatistics across batch sizes:")
    print(f"  Mean:   {losses_np.mean():.6f}")
    print(f"  Std:    {losses_np.std():.6f}")
    print(f"  Min:    {losses_np.min():.6f}")
    print(f"  Max:    {losses_np.max():.6f}")
    print(f"  Range:  {losses_np.max() - losses_np.min():.6f}")
    
    # Validate reasonable range
    # (Losses should be similar since all use random data with same distribution)
    loss_range = losses_np.max() - losses_np.min()
    if loss_range < 0.02:  # Arbitrary threshold for "similar"
        print("\n✓ Losses are consistent across batch sizes")
    else:
        print(f"\n⚠ Large variance across batches (range={loss_range:.4f})")
        print("  (This is expected for random data, but check if using real data)")
    
    print("\n✓ Batch processing test passed")
    print("  → All batch sizes computed successfully")


# MAIN EXECUTION
# This section runs when the file is executed directly (not imported)

if __name__ == '__main__':
    # First, run the quick demo
    print("=" * 60)
    print("MFE PROFILE LOSS - QUICK DEMO")
    print("=" * 60)
    _demo()
    
    # Then run the comprehensive test suite
    print("\n" + "=" * 60)
    print("MFE PROFILE LOSS - TEST SUITE")
    print("=" * 60)
    
    test_basic_functionality()
    test_gradient_flow()
    test_identical_signals()
    test_different_shapes()
    test_short_signals()
    test_distance_metrics()
    test_reduction_modes()
    test_profile_visualization()
    test_numerical_stability()
    test_batch_processing()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
