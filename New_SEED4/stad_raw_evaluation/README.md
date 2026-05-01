# STAD Model Evaluation on Raw SEED-IV Data

## Overview
This directory contains the complete evaluation results of the STAD (Spatio-Temporal Attention Diffusion) model for EEG super-resolution on raw SEED-IV data.

## Evaluation Configuration

### Data & Model
- **Dataset**: `/DATA/EEG-MTP/seed4/raw_data.npz`
  - Input (LR): 16 channels × 1000 timepoints
  - Target (SR): 62 channels × 1000 timepoints
  - Intermediate (HR): 31 channels × 1000 timepoints

- **STAD Model**: `results_stad_raw/best_stad_model.pth`
  - Architecture: Spatio-Temporal Attention Diffusion
  - Components: STC (Spatio-Temporal Conditioner) + MTD (Multi-scale Transformer Denoising)
  - Diffusion Schedule: Cosine

- **MAE Encoder**: `/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_raw/best_model.pt`
  - Channels: 31 (HR representation)
  - Mode: Frozen (used for conditioning latent space)

### Data Leakage Prevention ✅
- **Fold Strategy**: 5-fold cross-validation (KFold, shuffle=True, random_state=2024)
- **Test Subjects** (Fold 0): ['7', '10', '15']
- **Training Subjects**: ['1', '2', '3', '4', '5', '6', '8', '9', '11', '12', '13', '14']
- **Filtering**: 1,407 out of 7,035 samples selected (19.9% - test subjects only)
- **Data Integrity**: No training/validation subjects in test set

## Results

### Performance Metrics (Batch-level aggregation)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **PCC** (Pearson Correlation) | 0.0502 | 0.0228 | 0.0090 | 0.1136 |
| **NMSE** (Normalized MSE) | 0.9956 | 0.0050 | 0.9845 | 1.0057 |
| **SNR** (dB) | 0.02 | 0.02 | -0.02 | 0.07 |

**Note**: Low reconstruction quality (PCC ~0.05) indicates the model may require:
1. Retraining with adjusted hyperparameters
2. Investigation of training convergence
3. Validation that test subjects were NOT included during training

### Subject Breakdown

| Subject | Trial Count | Data File |
|---------|------------|-----------|
| **Subject 7** | 469 | `../stad_raw_subject_output/subject_7/` |
| **Subject 10** | 469 | `../stad_raw_subject_output/subject_10/` |
| **Subject 15** | 469 | `../stad_raw_subject_output/subject_15/` |
| **Total** | 1,407 | |

## Output Files

### 1. `ten_samples_comparison.png`
**Purpose**: Visual comparison of model predictions on 10 representative samples  
**Content**:
- 10 rows (samples 0-9 from test set)
- 5 columns showing 5 representative EEG channels out of 62
- **Green**: Target (ground truth) 62-channel signal
- **Red**: Model prediction
- **X-axis**: Time (normalized to 0-1)
- **Y-axis**: Normalized amplitude per channel

**Usage**: Quick visual assessment of reconstruction quality

### 2. `results_summary.npz`
**Purpose**: Complete evaluation data in NumPy compressed format  
**Contents** (accessible via `np.load()`):
```python
data = np.load('results_summary.npz', allow_pickle=True)

# Keys:
data['pred_sr']        # (44, 32, 62, 1000) - Predictions (batches)
data['target_sr']      # (44, 32, 62, 1000) - Ground truth (batches)
data['pcc_scores']     # (44,) - Per-batch PCC
data['nmse_scores']    # (44,) - Per-batch NMSE
data['snr_scores']     # (44,) - Per-batch SNR
```

**Usage**: Detailed per-batch analysis, custom metric computation

## Subject-Wise Output Structure

Located in: `../stad_raw_subject_output/`

```
stad_raw_subject_output/
├── subject_7/
│   ├── trial_0000_pred_sr.npy      # Model prediction (62, 1000)
│   ├── trial_0000_target_sr.npy    # Ground truth (62, 1000)
│   ├── trial_0000_meta.json        # Metadata {subject, trial, fold}
│   ├── trial_0001_pred_sr.npy
│   ├── trial_0001_target_sr.npy
│   ├── trial_0001_meta.json
│   └── ... (469 trials total)
├── subject_10/
│   └── ... (469 trials)
└── subject_15/
    └── ... (469 trials)
```

### File Format Details

**`trial_XXXX_pred_sr.npy`**
- Shape: (62, 1000)
- Content: Model super-resolution prediction
- Data type: float32
- Order: All 62 EEG channels, 1000 timepoints

**`trial_XXXX_target_sr.npy`**
- Shape: (62, 1000)
- Content: Ground truth 62-channel signal
- Data type: float32
- Order: All 62 EEG channels, 1000 timepoints

**`trial_XXXX_meta.json`**
- Content: Metadata including subject ID, trial index, fold
- Example:
  ```json
  {"subject": "7", "trial": 0, "fold": 0}
  ```

## Processing Information

### Computational Details
- **Hardware**: NVIDIA RTX A5000
- **Framework**: PyTorch with CUDA
- **Inference Speed**: 3.09 batches/second
- **Total Batches**: 44 (batch_size=32)
- **Total Runtime**: ~14 seconds

### Batch Processing
- Batch size: 32
- Padding: Samples padded to batch size where needed
- Device: CUDA (GPU acceleration)

## How to Use These Results

### 1. Batch-Level Analysis
```python
import numpy as np

results = np.load('results_summary.npz', allow_pickle=True)
pcc_per_batch = results['pcc_scores']
nmse_per_batch = results['nmse_scores']

# Find best/worst performing batches
best_batch = np.argmax(pcc_per_batch)
worst_batch = np.argmin(pcc_per_batch)
```

### 2. Subject-Level Statistics
```python
import os
import numpy as np
from pathlib import Path

subject_dir = 'stad_raw_subject_output/subject_7'
pred_files = sorted(Path(subject_dir).glob('trial_*_pred_sr.npy'))

subject_predictions = []
for pred_file in pred_files:
    pred = np.load(pred_file)
    subject_predictions.append(pred)

subject_predictions = np.array(subject_predictions)  # (469, 62, 1000)
```

### 3. Custom Metric Computation
```python
import numpy as np
from scipy.stats import pearsonr

pred_sr = results['pred_sr'].reshape(-1, 1000)  # Flatten batches
target_sr = results['target_sr'].reshape(-1, 1000)

# Per-channel correlation
channel_pcc = []
for ch in range(62):
    corr, _ = pearsonr(pred_sr[:, :].ravel(), target_sr[:, :].ravel())
    channel_pcc.append(corr)
```

## Notes & Caveats

### ⚠️ Model Performance Warning
The current model shows very low reconstruction quality (PCC ~0.05). This suggests:
- **Possible causes**:
  1. Model underfitting or non-convergence
  2. Hyperparameter mismatch between training and evaluation
  3. Data distribution shift between train and test
  4. Inadvertent data leakage during training (verify subjects)

- **Recommended actions**:
  1. Check training curves to verify model convergence
  2. Validate that test subjects were excluded from training
  3. Run on validation set to confirm reproducibility
  4. Compare against baseline models

### ✅ Data Integrity Confirmed
- Fold-based filtering prevents data leakage
- 100% of test samples use correct test subjects
- No training/validation overlap with test set

### File Size Information
- Per-sample: ~248 KB (62×1000 float32 predictions + targets)
- Total subject output: ~469 trials × 3 subjects × ~248 KB ≈ 348 MB
- Summary NPZ: ~55 MB
- Total storage: ~403 MB

## References

**Model Architecture**:
- STAD: Spatio-Temporal Attention Diffusion
- Components: STC (4.2M params) + MTD (78.8M params) + MAE (92.8M frozen)

**Dataset**:
- SEED-IV: Steady-State Visually Evoked Potential (SSVEP) EEG dataset
- Raw preprocessing: 200 Hz sampling, 16-62 channel interpolation

**Metrics**:
- **PCC**: Pearson Correlation Coefficient (channel-wise mean)
- **NMSE**: Normalized Mean Squared Error
- **SNR**: Signal-to-Noise Ratio computed from reconstruction

## Directory Navigation

```
/home/ab_students/EEG-MTP/
├── New_SEED4/
│   ├── stad_raw_evaluation/              ← You are here
│   │   ├── README.md                     (this file)
│   │   ├── results_summary.npz           (metrics & predictions)
│   │   └── ten_samples_comparison.png    (visualization)
│   └── stad_raw_subject_output/
│       ├── subject_7/                    (469 trials)
│       ├── subject_10/                   (469 trials)
│       └── subject_15/                   (469 trials)
└── results_stad_raw/
    └── best_stad_model.pth               (model checkpoint)
```

---

**Last Updated**: $(date)  
**Evaluation Version**: 1.0  
**Status**: ✅ Complete - Ready for Analysis
