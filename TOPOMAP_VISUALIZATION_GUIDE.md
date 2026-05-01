# EEG Topomap Visualization - Workspace Summary

## Overview
This document provides a comprehensive guide to topomap visualization code in the EEG-MTP workspace. Topomaps are 2D scalp surface plots that display EEG activity or power distribution across electrode positions.

---

## Primary Topomap Visualization File

### [New_LocalizeMI/evaluate_topomap_stad.py](New_LocalizeMI/evaluate_topomap_stad.py)
**Main file for creating multi-resolution topomap comparisons**

This is the **primary topomap visualization script** in the workspace. It generates topographic maps across frequency bands (δ, θ, α, β, γ) for different channel resolutions.

#### Key Features:
- Multi-resolution comparison (64-ch LR → 128-ch HR → 256-ch SR)
- Frequency band analysis (5 bands)
- MNE-based topomap plotting with brain structure visualization
- Additional visualizations: time-series comparisons, channel detail analysis

#### Output Directory:
- [New_LocalizeMI/topomap_results/](New_LocalizeMI/topomap_results/) - Generated topomaps and visualizations

---

## Key Utility Functions for Topomap Generation

### 1. **PSD Computation (Power Spectral Density)**

```python
def compute_psd_multitaper(eeg_data, fs=8000, fmin=0.5, fmax=45):
    """Compute PSD using Multitaper method"""
    psds, freqs = psd_array_multitaper(
        eeg_data, sfreq=fs, fmin=fmin, fmax=fmax,
        adaptive=True, normalization='full', verbose=False
    )
    return psds, freqs
```
- Uses MNE's multitaper method for robust PSD estimation
- Input: EEG data (C, T), sampling frequency
- Output: Power matrix (C, F) and frequency vector

### 2. **Frequency Band Power Extraction**

```python
def extract_band_power(psds, freqs, fmin, fmax):
    """Extract average power in frequency band"""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.mean(psds[:, mask], axis=1)
```
- Extracts spatially-resolved band power
- Returns power vector (C,) for each electrode
- Used to color topomaps

### 3. **Multi-Resolution Topomap Creation**

```python
def create_multireso_topomap(eeg_dict, sr_eeg, fs=8000, save_path='topomap_multireso.png'):
    """Create multi-resolution topomap comparison with proper brain structure"""
```

**Input:**
- `eeg_dict`: Dictionary with keys '64ch', '128ch', '256ch_GT' containing EEG (C, T)
- `sr_eeg`: Super-resolution EEG (256, T)
- `fs`: Sampling frequency (default 8000 Hz)

**Process:**
1. Compute PSD for each resolution using multitaper method
2. Extract band powers for 5 frequency bands
3. Create 3×5 subplot grid (3 resolutions × 5 frequency bands)
4. For each subplot:
   - Get/create montage for channel layout
   - Extract band power
   - Plot topomap using `mne.viz.plot_topomap()`

**Key Parameters for `mne.viz.plot_topomap()`:**
```python
mne.viz.plot_topomap(
    band_power,              # (C,) power vector
    info,                    # MNE Info object with montage
    axes=ax,                 # Matplotlib axes
    show=False,              # Don't display
    cmap='RdBu_r',          # Red-Blue colormap
    contours=8,             # Contour levels
    outlines='head',        # Draw head outline
    sphere='auto',          # Sphere for projection
    sensors=True,           # Show electrode positions
    res=128,                # Resolution (128x128)
    extrapolate='head',     # Extrapolate to head edges
    border='mean',          # Border handling
    size=4                  # Electrode size
)
```

---

## Electrode Positioning & Montage Setup

### 1. **Montage Selection Strategy**

The code uses standard MNE montages based on channel count:

```python
if n_channels == 256:
    montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
elif n_channels == 128:
    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
else:  # 64 channels
    montage = mne.channels.make_standard_montage('biosemi64')

ch_names = montage.ch_names[:n_channels]
info = mne.create_info(ch_names, fs, ch_types='eeg')
info.set_montage(montage)
```

### 2. **SEED-IV Channel Coordinates (Custom Positioning)**

Located in: [new_Seed4_hfd_mfe/stad_model_CORRECT.py](new_Seed4_hfd_mfe/stad_model_CORRECT.py)

```python
def _create_seed4_positions(self, n_channels, channel_indices=None):
    """
    Build SEED-IV channel coordinates from MNE standard_1005 montage.
    Uses SEED-IV channel ordering and maps CB1/CB2 -> I1/I2.
    """
    seed4_mne_names = [
        'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
        'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
        'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
        'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'I1', 'O1', 'Oz',
        'O2', 'I2'
    ]

    # Load MNE standard 10-05 montage
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_pos = montage.get_positions()['ch_pos']

    # Extract (x, y) coordinates for each channel
    full_positions = []
    for name in seed4_mne_names:
        xyz = ch_pos[name]
        full_positions.append([xyz[0], xyz[1]])  # Take x, y only

    # Normalize positions to [-1, 1] range
    full_positions = np.array(full_positions, dtype=np.float32)
    scale = np.max(np.abs(full_positions))
    if scale > 0:
        full_positions = full_positions / scale

    # Downsample if needed
    if channel_indices is not None:
        return full_positions[channel_indices]
    
    if n_channels < len(full_positions):
        indices = np.linspace(0, len(full_positions) - 1, n_channels, dtype=int)
        return full_positions[indices]
    
    return full_positions
```

**Key Notes:**
- Uses `standard_1005` montage (10-05 system with ~340 channels)
- Extracts (x, y) coordinates from 3D positions
- Normalizes to [-1, 1] range for cortical projection
- Maps CB1→I1 and CB2→I2 for SEED-IV compatibility

### 3. **Synthetic Position Fallback**

If MNE montage is unavailable:

```python
# Create synthetic circular arrangement
angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
radius = 0.5
pos_x = radius * np.cos(angles)
pos_y = radius * np.sin(angles)
pos_z = np.zeros(n_channels)

pos_dict = {ch: np.array([x, y, z]) for ch, x, y, z in zip(ch_names, pos_x, pos_y, pos_z)}
montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')
info.set_montage(montage)
```

---

## MNE-Based Channel Interpolation

Located in: [Noel-Preprocessing/FinalPrC-1.py](Noel-Preprocessing/FinalPrC-1.py)

```python
def interpolate_bad_channels(x, bad_mask, sfreq=RAW_FS):
    """Spherical spline interpolation via MNE (standard_1005 montage)."""
    if not bad_mask.any():
        return x

    info = mne.create_info(ch_names=MNE_CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    raw = mne.io.RawArray(x, info, verbose=False)
    raw.set_montage(montage, on_missing='warn', verbose=False)

    bad_ch_names = [MNE_CHANNEL_NAMES[i] for i in np.where(bad_mask)[0]]
    raw.info['bads'] = bad_ch_names
    raw.interpolate_bads(mode='accurate', verbose=False)

    return raw.get_data()
```

**Purpose:**
- Interpolates "bad" (noisy/artifact) channels using spherical spline method
- Preserves spatial structure through montage
- Uses standard_1005 montage for realistic electrode geometry

---

## Graph Harmonic Spatial Embeddings

Located in: [New_SEED/spatio_temporal_condition.py](New_SEED/spatio_temporal_condition.py)

Advanced technique using spectral graph theory instead of direct position embeddings:

```python
def compute_graph_harmonics(chan_pos, k=8, n_neighbors=4):
    """
    Compute graph Laplacian eigenvectors for spatial embeddings.
    Uses spectral graph theory to capture topological structure of electrode layout.
    
    Returns: (C, k) graph harmonic basis functions
    """
    # Build k-NN graph from electrode positions
    A = kneighbors_graph(
        chan_pos, 
        n_neighbors=min(n_neighbors, C-1), 
        mode='connectivity', 
        include_self=False
    ).toarray()

    # Compute normalized graph Laplacian: L = D^(-1/2) * (D - A) * D^(-1/2)
    L = csgraph.laplacian(A, normed=True)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Take first k+1 eigenvectors (skip constant first one)
    # These are the "harmonic" basis functions on the electrode graph
    basis = eigenvectors[:, 1:k+1]  # (C, k)
    
    return torch.tensor(basis, dtype=torch.float32)
```

**Innovation:**
- Captures **topological structure** of electrode arrangement, not just geometric positions
- Uses graph Laplacian eigenvectors as basis functions
- More robust to electrode layout variations

---

## Frequency Bands Definition

```python
FREQ_BANDS = {
    'δ (0.5-4 Hz)': (0.5, 4),
    'θ (4-8 Hz)': (4, 8),
    'α (8-13 Hz)': (8, 13),
    'β (13-30 Hz)': (13, 30),
    'γ (30-45 Hz)': (30, 45)
}
```

---

## Supported Montages in MNE

### Standard Clinical Montages:
- `'standard_1005'` - 10-05 system (most channels)
- `'standard_1020'` - 10-20 system (classic)
- `'biosemi64'` - BioSemi 64-channel cap

### High-Density Arrays:
- `'GSN-HydroCel-256'` - Geodesic Sensor Network 256-ch
- `'GSN-HydroCel-128'` - Geodesic Sensor Network 128-ch
- `'GSN-HydroCel-64'` - Geodesic Sensor Network 64-ch

---

## Additional Visualization Functions

### Time-Series Comparison
```python
def visualize_random_channels(eeg_dict, sr_eeg, fs=8000, n_channels=10, 
                              duration_sec=0.5, save_path='timeseries_comparison.png')
```
- Plots random channels across different resolutions
- Vertical offset for clarity
- Output: 3-row subplot (LR, HR, SR)

### Channel Detail View
```python
def visualize_channel_detail_comparison(eeg_dict, sr_eeg, fs=8000, 
                                        save_path='channel_detail_comparison.png')
```
- Zoomed view of single channel across resolutions
- 200ms window for temporal detail
- Shows improvement from LR → SR

### Metrics Computation
```python
def compute_metrics(pred, target):
    """Compute reconstruction metrics"""
    return {
        'pcc': float(pcc),      # Pearson Correlation Coefficient
        'nmse': float(nmse),    # Normalized Mean Squared Error
        'snr': float(snr),      # Signal-to-Noise Ratio (dB)
        'mae': float(mae)       # Mean Absolute Error
    }
```

---

## Data Loading & Preprocessing

```python
def load_test_samples_multireso(data_path, n_samples=100):
    """Load test samples at available resolutions (64, 128, 256 channels)"""
    # Loads 256-ch reference data
    # Downsamples: 256 → 128 (every 2nd) → 64 (every 4th) for comparison
    # Applies bandpass filtering and standardization
    
    return {
        '64ch': all_processed[0], 
        '128ch': all_processed[1],
        '256ch_GT': all_processed[2]
    }
```

---

## Generated Output Examples

Files located in [New_LocalizeMI/topomap_results/](New_LocalizeMI/topomap_results/):

1. **topomap_multireso_sample_1-5.png** - Per-sample topomaps (5 frequency bands × 3 resolutions)
2. **topomap_multireso_averaged.png** - Averaged across all samples
3. **timeseries_sample_1-5.png** - Time-series for random channels
4. **channel_detail_comparison.png** - Single channel zoom view
5. **metrics_summary.json** - Quantitative evaluation metrics

---

## MNE Imports Used

```python
import mne
import mne.viz                              # Plotting functions
from mne.time_frequency import psd_array_multitaper  # PSD computation
mne.channels.make_standard_montage()        # Load standard montages
mne.channels.make_dig_montage()             # Create custom montages
mne.create_info()                           # Create MNE Info object
mne.io.RawArray()                          # Create Raw data object
raw.set_montage()                          # Apply montage to data
raw.interpolate_bads()                     # Bad channel interpolation
```

---

## Key Implementation Notes

1. **Montage Setup:**
   - Always create MNE `Info` object before plotting
   - Set montage with `info.set_montage(montage)`
   - Use `on_missing='warn'` to handle missing channels gracefully

2. **Topomap Plotting Parameters:**
   - `sphere='auto'` for automatic head sphere detection
   - `extrapolate='head'` to extend plot to head boundary
   - `sensors=True` to show electrode marker positions
   - `outlines='head'` for anatomical reference

3. **PSD Computation:**
   - Multitaper method preferred for robustness
   - Adaptive=True uses adaptive multitaper
   - Normalization='full' preserves absolute units

4. **Electrode Positioning:**
   - Normalize coordinates to [-1, 1] for cortical projection
   - Graph harmonics capture topological structure better than raw positions
   - Fallback to synthetic circular layout if standard montage unavailable

---

## References

- **MNE-Python**: https://mne.tools/
- **Standard 10-05/10-20 Electrode Systems**: Oostenveld & Praamstra (2001)
- **STAD Paper**: Spatio-Temporal Architecture for EEG Super-Resolution
- **SEED-IV Dataset Specifics**: Channel mapping CB1/CB2 → I1/I2
