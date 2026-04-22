# STAD Training with MFE (Multiscale Fuzzy Entropy) Loss

This folder implements STAD (Spatio-Temporal diffusion for Anomaly Detection) training for SEED-IV EEG super-resolution using **Multiscale Fuzzy Entropy (MFE) loss** for complexity preservation.

## Overview

### What is MFE Loss?

**Multiscale Fuzzy Entropy (MFE)** is a differentiable loss function that compares the temporal complexity of signals across multiple time scales. Unlike amplitude-based losses (MSE, L1), MFE ensures that the generated super-resolved EEG preserves the physiological complexity and regularity patterns of the target signal.

**Key advantages:**
- **Temporal structure preservation**: Captures complexity across 4-80ms time scales
- **Scale-invariant**: Z-score normalization makes it robust to amplitude variations
- **Fully differentiable**: Uses smooth approximations for abs() and max() operations
- **Physiologically motivated**: Complexity is a key property of EEG signals

### Combined Loss Function

```
Total Loss = Diffusion Loss + λ_mse * MSE(pred, target) + λ_mfe * MFE(pred, target)
```

**Default weights:** λ_mse = 0.1, λ_mfe = 0.1 (tunable via command-line args)

## File Structure

```
new_SEED4_mfe/
├── seed_stad_train_mfe.py          # Main training script
├── mfe_profile_loss.py              # MFE loss implementation
├── stad_model_CORRECT.py            # STAD model architecture
├── config_seed4.py                  # Configuration
├── mae_for_eeg.py                   # MAE encoder/decoder
├── mtd_dreamdiff.py                 # Multi-scale transformer denoising
├── spatio_temporal_condition.py     # Conditioning module
├── diffusion_scheduler.py           # Diffusion schedule
├── utils.py                         # Utilities
├── README.md                        # This file
└── results/
    ├── best_stad_model.pth          # Best checkpoint
    ├── latest_stad_model.pth        # Latest checkpoint
    └── training_history.npy         # Training metrics
```

## Installation

All required dependencies are already available. MFE loss uses only PyTorch and NumPy.

```bash
cd /home/ab_students/EEG-MTP/new_SEED4_mfe
```

## Quick Start

### Basic Training Command

```bash
python seed_stad_train_mfe.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --sr_loss_weight 0.1 \
  --mfe_loss_weight 0.1 \
  --output_dir /home/ab_students/EEG-MTP/new_SEED4_mfe
```

### With MAE Checkpoint

```bash
python seed_stad_train_mfe.py \
  --mae_checkpoint /path/to/best_model.pth \
  --freeze_mae \
  --unfreeze_mae_epoch 50 \
  --epochs 100
```

### Resume Training

```bash
python seed_stad_train_mfe.py \
  --resume_stad_checkpoint /home/ab_students/EEG-MTP/new_SEED4_mfe/latest_stad_model.pth \
  --resume_optimizer \
  --epochs 100
```

## Command-Line Arguments

### Data & Model
- `--data_path`: Path to SEED-IV data (default: `/DATA/EEG-MTP/seed4/eeg_processed_data`)
- `--mae_checkpoint`: Path to pretrained MAE (auto-selected if empty)
- `--mae_kfold_dir`: K-fold results directory for MAE
- `--freeze_mae`: Freeze MAE encoder initially
- `--unfreeze_mae_epoch`: Epoch to unfreeze MAE (-1 = keep frozen)

### Training Hyperparameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.05)
- `--min_lr`: Minimum learning rate for scheduler (default: 1e-6)

### Loss Weights
- `--sr_loss_weight`: Weight for MSE reconstruction loss (default: 0.1)
- `--mfe_loss_weight`: Weight for MFE complexity loss (default: 0.1)

### MFE Parameters
- `--mfe_m`: Pattern length (default: 2) - standard for EEG
- `--mfe_n`: Fuzzy exponent (default: 2.0) - controls similarity sharpness
- `--mfe_tau_max`: Maximum time scale (default: 20) - covers ~4-80ms @ 250Hz
- `--mfe_r_fixed`: Tolerance threshold (default: 0.15) - after z-normalization

### Other Options
- `--diffusion_schedule`: 'linear' or 'cosine' (default: 'cosine')
- `--device`: 'cuda' or 'cpu' (default: 'cuda')
- `--output_dir`: Output directory (default: `/home/ab_students/EEG-MTP/new_SEED4_mfe`)
- `--test_only`: Run test mode only (skip training)

## MFE Loss Parameters - Tuning Guide

### Pattern Length (m)
- **m=2**: Default, matches consecutive pairs of samples
- **m=3**: More complex patterns, slower computation
- **Recommendation**: Keep m=2 for physiological signals

### Fuzzy Exponent (n)
- **n=1.0**: Linear similarity decay
- **n=2.0**: Gaussian-like membership (default, most stable)
- **n>2.0**: Sharper similarity cutoff
- **Recommendation**: n=2.0 provides smooth, stable gradients

### Time Scale Range (tau_max)
- **tau=1**: Original signal (fast dynamics)
- **tau=20**: 80ms window (slow dynamics at 250Hz)
- **Default**: tau_max=20 covers delta to gamma EEG bands
- **For longer windows**: Increase to tau_max=30-50
- **For shorter windows**: Decrease to tau_max=10-15

### Tolerance Threshold (r_fixed)
- **Purpose**: Controls strictness of pattern matching
- **r=0.15**: Standard threshold for z-scored signals
- **r<0.15**: Stricter matching, higher entropy
- **r>0.15**: Looser matching, lower entropy
- **Recommendation**: Keep r=0.15 (empirically optimal for EEG)

## Loss Weight Tuning

### Starter Configuration
```
--sr_loss_weight 0.1 --mfe_loss_weight 0.1
```

### If Output is Too Smooth
Increase MFE weight:
```
--sr_loss_weight 0.1 --mfe_loss_weight 0.3
```

### If Output is Too Noisy
Decrease MFE weight:
```
--sr_loss_weight 0.1 --mfe_loss_weight 0.05
```

### For Amplitude-Focused Training
Increase MSE weight:
```
--sr_loss_weight 0.3 --mfe_loss_weight 0.1
```

## Training Monitoring

### Loss Components in Output

```
Epoch 10/100 | Train: 0.234567 | Val: 0.245678 | PCC: 0.8456, NMSE: 0.0234, SNR: 15.23dB
```

- **Train/Val loss**: Total combined loss (should decrease)
- **PCC**: Pearson correlation coefficient (target: >0.8)
- **NMSE**: Normalized mean squared error (target: <0.05)
- **SNR**: Signal-to-noise ratio in dB (target: >15dB)

### Interpreting Training Behavior

✅ **Good training:**
- Total loss decreases monotonically
- PCC increases toward 0.8-0.9
- NMSE decreases toward 0.01-0.05
- SNR increases toward 15-25dB

⚠️ **MFE loss too high:**
- Training loss becomes unstable
- Try reducing `--mfe_loss_weight` to 0.05
- Increase learning rate slightly

⚠️ **MFE loss is zero/NaN:**
- Check signal lengths (must be ≥ tau_max + m + 10 samples)
- Verify data loading
- Check for constant/zero signals

## Output Files

### Checkpoints
- `best_stad_model.pth`: Best model based on validation loss
- `latest_stad_model.pth`: Latest checkpoint from last epoch

### Checkpoint Contents
```python
{
    'epoch': int,                          # Epoch number
    'model_state_dict': OrderedDict,       # Model weights
    'best_val_loss': float,               # Best validation loss so far
    'val_loss': float,                    # Current validation loss
    'train_loss': float,                  # Current training loss
}
```

### Training History
- `training_history.npy`: NumPy file with list of dicts containing:
  - `epoch`: Epoch number
  - `train_loss`: Training total loss
  - `val_loss`: Validation total loss
  - `val_pcc`: Validation PCC metric
  - `val_nmse`: Validation NMSE metric
  - `val_snr`: Validation SNR metric (dB)

Load history:
```python
import numpy as np
history = np.load('training_history.npy', allow_pickle=True).tolist()
for entry in history:
    print(f"Epoch {entry['epoch']}: Loss={entry['val_loss']:.6f}, PCC={entry['val_pcc']:.4f}")
```

## MFE Loss Details

### Mathematical Formulation

For each time scale τ ∈ {1, 2, ..., τ_max}:

1. **Coarse-grain**: Downsample by averaging non-overlapping windows
2. **Extract templates**: All sliding windows of length m (mean-centered)
3. **Compute distances**: Chebyshev distance with smooth approximations
4. **Fuzzy similarity**: S_ij = exp(-(d_ij^n)/r)
5. **Average similarity**: Φ_m = mean over all template pairs
6. **Fuzzy entropy**: FuzzyEn(τ) = log(Φ_m) - log(Φ_{m+1})

### Profile Loss

```
Loss_MFE = MSE over all scales: mean_τ |FuzzyEn_gen(τ) - FuzzyEn_target(τ)|
```

### Differentiability

All operations use smooth approximations:
- `|x|` → `sqrt(x² + ε)`
- `max(v)` → `log(sum(exp(βv))) / β`
- `log(x)` → `log(clamp(x, ε))`

This ensures stable gradients throughout training.

## Performance Expectations

With default settings on SEED-IV dataset:

| Metric | Target | Range |
|--------|--------|-------|
| PCC | >0.80 | 0.75-0.95 |
| NMSE | <0.05 | 0.01-0.10 |
| SNR (dB) | >15 | 10-25 |
| Training time | ~40-60s/epoch | (Depends on hardware) |

## Troubleshooting

### Issue: MFE Loss is NaN/Inf

**Causes:**
- Signal length too short (< 100 samples)
- All-zero or constant signals
- Extreme amplitude values

**Solutions:**
- Check data loading: `print(batch['sr'].shape, batch['sr'].min(), batch['sr'].max())`
- Reduce `tau_max` if signals are short
- Verify normalization in data preprocessing

### Issue: Training Loss Doesn't Decrease

**Causes:**
- Learning rate too high/low
- Loss weights imbalanced
- Model capacity insufficient

**Solutions:**
- Try different learning rates: [5e-5, 1e-4, 2e-4]
- Adjust loss weights: increase MSE weight if amplitude is off
- Verify diffusion model is working (check pure diffusion loss)

### Issue: Output is Too Smooth

**Causes:**
- MFE weight too high (over-penalizes complexity)
- MSE weight dominates

**Solutions:**
- Reduce `--mfe_loss_weight` to 0.05
- Increase `--sr_loss_weight` to 0.3
- Verify target signal complexity is captured by MFE

### Issue: Output is Too Noisy

**Causes:**
- MFE weight too low
- Data quality issues

**Solutions:**
- Increase `--mfe_loss_weight` to 0.3
- Check source data quality
- Verify preprocessing steps

## Comparison with HFD Loss

| Aspect | MFE | HFD |
|--------|-----|-----|
| **Metric** | Fuzzy entropy across scales | Fractal dimension slope |
| **Temporal focus** | Individual patterns & scale | Overall scaling behavior |
| **Computation** | Template matching | Curve length integration |
| **Complexity capture** | Directly measures regularity | Indirect via roughness |
| **Typical weight** | 0.1-0.3 | 0.1-0.5 |
| **Strengths** | Fine temporal structure | Global signal geometry |
| **Best for** | EEG with sharp transients | Smooth signals |

## References

1. **Multiscale Fuzzy Entropy**: Costa et al. (2002) - Multiscale entropy analysis of complex signals
2. **Fuzzy Entropy**: Chen et al. (2007) - Fuzzy entropy and approximate entropy methods
3. **Tweedie's Formula**: Used in diffusion model x0 prediction
4. **STAD Architecture**: Spatio-temporal diffusion for EEG

## Citation

If using this implementation, please cite:
```bibtex
@code{stad_mfe_2024,
  title={STAD with MFE Loss for EEG Super-Resolution},
  author={Your Name},
  year={2024},
  note={Implementation based on MFE methodology from Costa et al.}
}
```

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review `training_history.npy` for loss trends
3. Verify data shapes: `(B, C, T)` format
4. Check GPU memory: Monitor with `nvidia-smi`

---

**Status**: Ready for training
**Last Updated**: April 2024
**Folder**: `/home/ab_students/EEG-MTP/new_SEED4_mfe`
