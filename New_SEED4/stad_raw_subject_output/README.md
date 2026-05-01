# Subject-Wise STAD Model Predictions

## Overview
This directory contains per-subject, per-trial predictions from the STAD model evaluation on raw SEED-IV data. Each trial is stored separately with its corresponding ground truth and metadata.

## Directory Structure

```
stad_raw_subject_output/
├── subject_7/              (469 trials)
├── subject_10/             (469 trials)
└── subject_15/             (469 trials)

Each subject contains structure:
subject_X/
├── trial_0000_pred_sr.npy      (62, 1000) - model prediction
├── trial_0000_target_sr.npy    (62, 1000) - ground truth
├── trial_0000_meta.json        - metadata
├── trial_0001_pred_sr.npy
├── trial_0001_target_sr.npy
├── trial_0001_meta.json
└── ...
```

## File Formats

### Prediction & Target Files (`.npy`)
- **Shape**: (62, 1000)
- **Data type**: float32
- **Dimensions**:
  - 62: EEG channels (full montage)
  - 1000: Time points (5 seconds at 200 Hz)
- **Values**: Normalized amplitude (0-centered)

**Loading in Python**:
```python
import numpy as np

pred = np.load('subject_7/trial_0000_pred_sr.npy')  # (62, 1000)
target = np.load('subject_7/trial_0000_target_sr.npy')  # (62, 1000)

# Compute correlation per channel
from scipy.stats import pearsonr
for ch in range(62):
    corr, pval = pearsonr(pred[ch], target[ch])
    print(f"Channel {ch}: correlation = {corr:.4f}")
```

### Metadata Files (`.json`)
- **Format**: JSON text
- **Fields**:
  - `subject` (str): Subject ID
  - `trial` (int): Trial index within subject
  - `fold` (int): Cross-validation fold (0=test)

**Example**:
```json
{"subject": "7", "trial": 0, "fold": 0}
```

## Subject Information

| Subject | Trial Count | Samples Range | Data File Group |
|---------|------------|---------------|-----------------|
| **7** | 469 | trial_0000 - trial_0468 | `subject_7/` |
| **10** | 469 | trial_0000 - trial_0468 | `subject_10/` |
| **15** | 469 | trial_0000 - trial_0468 | `subject_15/` |
| **Total** | 1,407 | | |

## Common Analysis Workflows

### 1. Load All Predictions for a Subject
```python
import numpy as np
from pathlib import Path

subject_id = '7'
subject_path = f'subject_{subject_id}'

predictions = []
targets = []

for i in range(469):
    pred_file = f'{subject_path}/trial_{i:04d}_pred_sr.npy'
    target_file = f'{subject_path}/trial_{i:04d}_target_sr.npy'
    
    predictions.append(np.load(pred_file))
    targets.append(np.load(target_file))

predictions = np.stack(predictions)  # (469, 62, 1000)
targets = np.stack(targets)          # (469, 62, 1000)
```

### 2. Extract Specific Channels
```python
# Get only channels 0-15 (original LR montage area)
pred_lr_area = predictions[:, :16, :]  # (469, 16, 1000)

# Get only channels 16-30 (interpolated area)
pred_interp_area = predictions[:, 16:31, :]  # (469, 15, 1000)

# Get all high-resolution channels
pred_hr_full = predictions[:, :31, :]  # (469, 31, 1000)
```

### 3. Compute Subject-Level Metrics
```python
from scipy.stats import pearsonr
from scipy.spatial.distance import correlation

def compute_subject_metrics(predictions, targets):
    """Compute metrics across all trials for a subject."""
    n_trials = predictions.shape[0]
    
    pcc_per_channel = []
    
    # Per-channel correlation
    for ch in range(62):
        pred_ch = predictions[:, ch, :].ravel()
        target_ch = targets[:, ch, :].ravel()
        corr, _ = pearsonr(pred_ch, target_ch)
        pcc_per_channel.append(corr)
    
    return {
        'mean_pcc': np.mean(pcc_per_channel),
        'per_channel_pcc': pcc_per_channel,
        'spatial_consistency': np.std(pcc_per_channel)
    }

metrics = compute_subject_metrics(predictions, targets)
print(f"Subject {subject_id}:")
print(f"  Mean PCC: {metrics['mean_pcc']:.4f}")
print(f"  Spatial consistency: {metrics['spatial_consistency']:.4f}")
```

### 4. Temporal Analysis per Trial
```python
# Analyze temporal consistency within a trial
trial_idx = 0
pred_trial = predictions[trial_idx]  # (62, 1000)
target_trial = targets[trial_idx]    # (62, 1000)

# Per-channel temporal correlation
temporal_corr = []
for ch in range(62):
    corr, _ = pearsonr(pred_trial[ch], target_trial[ch])
    temporal_corr.append(corr)

print(f"Trial {trial_idx} - Per-channel correlations:")
for ch, corr in enumerate(temporal_corr):
    print(f"  Channel {ch:2d}: {corr:+.4f}")
```

### 5. Compare Across All Subjects
```python
import pandas as pd

subject_metrics = {}

for subject_id in ['7', '10', '15']:
    # Load all predictions/targets for subject
    predictions = []
    targets = []
    for i in range(469):
        pred_file = f'subject_{subject_id}/trial_{i:04d}_pred_sr.npy'
        target_file = f'subject_{subject_id}/trial_{i:04d}_target_sr.npy'
        predictions.append(np.load(pred_file))
        targets.append(np.load(target_file))
    
    predictions = np.stack(predictions)
    targets = np.stack(targets)
    
    # Compute metrics
    metrics = compute_subject_metrics(predictions, targets)
    subject_metrics[subject_id] = metrics

# Create comparison table
df = pd.DataFrame({
    'Subject': list(subject_metrics.keys()),
    'Mean PCC': [subject_metrics[s]['mean_pcc'] for s in subject_metrics],
    'Spatial Std': [subject_metrics[s]['spatial_consistency'] for s in subject_metrics]
})

print(df)
```

## Data Characteristics

### Baseline Statistics (Expected Ranges)
- **Signal amplitude**: ±100 μV (normalized to ±1 in these files)
- **Temporal resolution**: 1000 samples = 5 seconds
- **Sampling frequency**: 200 Hz
- **Expected PCC**: Depends on model architecture
  - Poor model: PCC < 0.1
  - Acceptable model: PCC 0.3-0.7
  - Good model: PCC > 0.7

### Current Model Performance
See `../stad_raw_evaluation/README.md` for overall metrics.

**Per trial visual**: See `../stad_raw_evaluation/ten_samples_comparison.png`

## Troubleshooting

### File Not Found
```python
from pathlib import Path

subject_path = Path('subject_7')
print(f"Files in subject_7: {len(list(subject_path.glob('trial_*.npy')))}")

# Should be 938 files (469 trials × 2 files per trial, excluding metadata)
```

### Memory Issues with All Data
If loading all 1,407 samples × 62 channels × 1,000 timepoints causes memory issues:
```python
# Option 1: Load per-subject and process
for subject_id in ['7', '10', '15']:
    subject_preds = load_subject(f'subject_{subject_id}')  # Process immediately
    del subject_preds  # Free memory

# Option 2: Use memory-mapped arrays
predictions = np.load('subject_7/trial_0000_pred_sr.npy', mmap_mode='r')

# Option 3: Subsample channels or time points
pred_subset = predictions[::2, ::5]  # Every 2nd trial, every 5th sample
```

### Null/NaN Values
All files should contain valid numerical data. If you encounter NaN:
```python
pred = np.load('subject_7/trial_0000_pred_sr.npy')
print(f"NaN count: {np.isnan(pred).sum()}")
print(f"Inf count: {np.isinf(pred).sum()}")
print(f"Value range: [{np.nanmin(pred):.4f}, {np.nanmax(pred):.4f}]")
```

## Accessing via Command Line

### Count trials per subject
```bash
ls subject_7/trial_*_pred_sr.npy | wc -l  # Should be 469
```

### Find specific trial
```bash
ls -lh subject_7/trial_0042_*  # Find trial 42 files and size
```

### Get directory statistics
```bash
du -sh subject_7/  # Total size per subject
du -sh .           # Total size of all outputs
```

### Quick sample inspection
```bash
python << 'EOF'
import numpy as np

pred = np.load('subject_7/trial_0000_pred_sr.npy')
target = np.load('subject_7/trial_0000_target_sr.npy')

print(f"Shape: {pred.shape}")
print(f"Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
EOF
```

## Storage Information

| Item | Size |
|------|------|
| Per trial (both files) | ~496 KB |
| Per subject (469 trials) | ~232 MB |
| All subjects (1,407 trials) | ~696 MB |
| Metadata (JSON) | Negligible |

## Data Leakage Prevention ✅

- **Subjects 7, 10, 15**: Test subjects only
- **KFold split**: Random state = 2024 (reproducible)
- **No training subjects**: ['1', '2', '3', '4', '5', '6', '8', '9', '11', '12', '13', '14']
- **Verification**: Each trial includes fold=0 in metadata (test fold)

## Next Steps

1. **Analyze metrics**: See `../stad_raw_evaluation/README.md`
2. **Visualize predictions**: Load and plot per-subject samples
3. **Compare across subjects**: Identify subject-specific patterns
4. **Identify problematic trials**: Find worst-performing trials per subject
5. **Validate reconstruction**: Compare LR→SR interpolation vs model predictions

---

**Last Updated**: $(date)  
**Data Format Version**: 1.0  
**Total Trials**: 1,407  
**Status**: ✅ Complete - Ready for Analysis
