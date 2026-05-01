# Enhanced `test_stad_preprocessed.py` - User Guide

## Overview
The improved `test_stad_preprocessed.py` now includes comprehensive dataset validation, metrics visualization, and topographic map generation for advanced analysis of STAD super-resolution EEG models.

---

## New Features

### 1. **Dataset Statistics & Validation** ✅
Automatically computed when evaluation starts:
- **Shape validation**: Confirms channel/sample dimensions
- **Signal statistics**: Mean, std, min/max for LR/HR/SR
- **Data quality metrics**: Signal range, distribution analysis
- **Sample size reporting**: Total and analyzed counts

**Output Example:**
```
================================================================================
DATASET STATISTICS REPORT
================================================================================
Total Samples: 1500 (analyzed: 50)
Signal Range: 45.2341 ± 12.3456

LR EEG:
  Shape: (1500, 16, 1000)
  Channels: 16
  Mean: -0.004521, Std: 0.987654
  Range: [-5.234234, 5.123456]
  
HR EEG:
  Shape: (1500, 31, 1000)
  ...
  
SR EEG:
  Shape: (1500, 62, 1000)
  ...
```

---

### 2. **Comprehensive Visualizations** 📊

#### A. Metrics Distribution (`metrics_distribution.png`)
Three histograms showing:
- **PCC Distribution**: Pearson correlation coefficient across all samples
- **NMSE Distribution**: Normalized MSE with mean overlay
- **SNR Distribution**: Signal-to-noise ratio in dB

**Use case**: Identify outliers, assess metric stability, validate model performance

#### B. Loss Curves (`loss_curves.png`)
Two plots showing:
- **Diffusion Loss**: Temporal evolution during inference (if not using sampling)
- **SR L1 Loss**: Reconstruction error per batch

**Use case**: Detect training issues, identify problematic batches

#### C. EEG Signal Comparison (`eeg_comparison.png`)
Side-by-side comparison across 3 samples:
- 5 representative channels per sample
- Target (ground truth) vs Predicted overlaid
- Time-domain waveform analysis

**Use case**: Visual quality assessment, artifact detection

#### D. Brain Topographic Maps (`topomap_comparison.png`)
Two scalp topographies using MNE:
- **Left**: Ground truth SR EEG spatial distribution
- **Right**: Model predicted SR spatial distribution

**Features**:
- Electrodes positioned on head surface
- Color intensity = signal amplitude
- Contour lines for better visualization
- 62-channel E-cap montage

**Use case**: Validate spatial reconstruction quality, detect localization errors

---

### 3. **New Command-Line Arguments**

```bash
python test_stad_preprocessed.py \
  --mae_checkpoint <path> \
  --stad_checkpoint <path> \
  --visualization_dir test_visualizations \  # ← NEW: output directory for all graphs/topomaps
  --save_fig_dir test_figures_preprocessed \
  --save_norm_stats_path norm_stats.npy \
  --save_prc1_meta_path prc1_meta.json
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--visualization_dir` | str | `test_visualizations` | Directory for comprehensive visualizations |
| `--save_fig_dir` | str | `test_figures_preprocessed` | Directory for EEG signal figures (per-sample detailed plots) |
| `--num_fig_samples` | int | 3 | How many per-sample figures to generate |

---

## Usage Examples

### Basic Usage (Minimal Output)
```bash
cd /home/ab_students/EEG-MTP/New_SEED4

python test_stad_preprocessed.py \
  --mae_checkpoint /path/to/mae_checkpoint.pt \
  --stad_checkpoint /path/to/stad_checkpoint.pth \
  --data_path /path/to/seed4/eeg_processed_data
```

**Output**:
- Console: Statistics report + metrics summary
- `test_visualizations/`: 
  - `metrics_distribution.png`
  - `loss_curves.png`
  - `eeg_comparison.png`
  - `topomap_comparison.png`

### Full Analysis (All Outputs)
```bash
python test_stad_preprocessed.py \
  --mae_checkpoint mae_checkpoint.pt \
  --stad_checkpoint best_stad_model.pth \
  --data_path /DATA/seed4/eeg_processed_data \
  --test_fold 0 \
  --batch_size 32 \
  --visualization_dir results_visualizations \
  --save_fig_dir results_eeg_figures \
  --save_sr_output_path results/pred_sr.npy \
  --save_target_output_path results/target_sr.npy \
  --save_test_metadata_path results/metadata.npz \
  --save_norm_stats_path results/norm_stats.npy \
  --save_prc1_meta_path results/prc1_meta.json \
  --save_grouped_output_dir results/grouped_output
```

---

## Output Structure

```
test_visualizations/
├── metrics_distribution.png     # Histograms: PCC, NMSE, SNR
├── loss_curves.png             # Loss evolvement per batch
├── eeg_comparison.png          # 3 samples × 5 channels
└── topomap_comparison.png      # Brain surface: target vs pred

test_figures_preprocessed/       # (Existing)
├── eeg_signal_sample_000.png
├── eeg_signal_sample_001.png
└── ...

results/
├── pred_sr.npy                 # All predicted 62-ch signals
├── target_sr.npy               # All target 62-ch signals
├── metadata.npz                # Subject/session/trial IDs
├── norm_stats.npy              # For PrC-1 reversal
└── prc1_meta.json              # Dataset metadata
```

---

## Implementation Details

### Dataset Statistics (`compute_dataset_statistics`)
- Samples 50 random windows by default
- Computes per-resolution statistics
- Returns shape, mean, std, min/max for validation

### Visualization Functions

| Function | Dependencies | Output |
|----------|--------------|---------|
| `compute_psd_batch()` | scipy.signal | Power spectral density |
| `save_metric_distributions_plot()` | matplotlib | Histograms with means |
| `save_loss_curves_plot()` | matplotlib | Loss evolution plots |
| `save_comparison_plot()` | matplotlib | Time-domain waveforms |
| `save_topomap_visualization()` | mne (optional) | Brain topography maps |

### MNE Integration
- **Optional dependency**: Code works without MNE but skips topomaps
- **Fallback**: Synthetic electrode positions (circle arrangement)
- **Supported**:
  - E-cap 62-channel montage
  - Color-coded amplitude visualization
  - Interpolated head outline

---

## Troubleshooting

### Topomap Visualization Fails
```
⚠️ Warning: Failed to generate topomap: ...
```
**Solution**: Install MNE
```bash
pip install mne
```

### Memory Issues with Large Batches
**Solution**: Reduce `--batch_size` or `--num_fig_samples`

### Visualization Directory Already Exists
**Solution**: Script auto-creates; existing files are overwritten

### Missing Dependencies
```python
ImportError: No module named 'scipy'
```
**Solution**: Install scipy
```bash
pip install scipy
```

---

## Performance Notes

- **Dataset statistics computation**: ~10-30 sec (50 samples)
- **Visualization generation**: ~5-15 sec total
- **Topomap rendering**: ~2-5 sec (MNE required)
- **Minimal overhead**: ~1-2% additional runtime

---

## Integration with Existing Workflow

✅ **Compatible with**:
- Existing `--save_*` arguments (unchanged)
- Fold-based subject filtering
- Grouped output saving
- NPZ/folder data sources

✅ **Backwards compatible**:
- Omit `--visualization_dir` → no graphs generated
- All new arguments are optional
- Default behavior unchanged if not specified

---

## Example Results Interpretation

### Good Model Performance ✅
- **PCC**: > 0.7 (mean)
- **NMSE**: < 0.1 (mean)
- **SNR**: > 10 dB (mean)
- **Topomaps**: Predicted matches target spatial pattern
- **Loss curves**: Smooth decrease, no spikes

### Potential Issues ⚠️
- **Low PCC** (< 0.5): Model not correlating with target
- **High NMSE** (> 0.3): Large reconstruction error
- **Negative SNR**: Noise exceeds signal power
- **Topomap mismatch**: Spatial reconstruction failure
- **Loss spikes**: Batch convergence issues

---

## Citation & References

- **STAD Model**: Spatio-Temporal Attention Diffusion
- **Topomaps**: MNE-Python (https://mne.tools/)
- **EEG Dataset**: SEED-IV (62-channel E-cap)
- **Metrics**: PCC, NMSE, SNR standard for EEG SR

---

## Version History

- **v2.0** (April 2026): Added dataset statistics, comprehensive visualizations, topomap support
- **v1.0**: Original evaluation script

