# MAE Training Comparison: Preprocessed vs Raw SEED-IV Data

## Overview
Running parallel k-fold cross-validation experiments to compare MAE performance on preprocessed vs raw SEED-IV data.

## Datasets

### Preprocessed Data
- **Source**: `/DATA/seed4/preprocessed_data.npz`
- **Processing**: PrC-1 pipeline (baseline correction, filtering, etc.)
- **Samples**: 14,280 windows
- **Format**: 31-channel HR (evenly distributed from 62 channels)
- **Sampling rate**: 250 Hz
- **Window size**: 1000 samples (4 seconds)

### Raw Data
- **Source**: `/DATA/seed4/eeg_raw_data/`
- **Processing**: Minimal (downsampling from 1000 Hz → 250 Hz)
- **Samples**: 7,035 windows
- **Format**: 31-channel HR (evenly distributed from 62 channels)
- **Sampling rate**: 250 Hz (downsampled from 1000 Hz)
- **Window size**: 1000 samples (4 seconds)

**Note**: Raw data has half the samples because the original data was recorded at 1000 Hz and was split into trials differently than the preprocessed version.

## Training Configuration

### Model Architecture (MAE)
- **Input channels**: 31 (HR resolution)
- **Time length**: 1000 samples
- **Patch size**: 8 → 125 patches
- **Embed dim**: 768
- **Encoder**: 12 layers, 12 heads
- **Decoder**: 4 layers, 8 heads, 384-dim
- **Parameters**: 92.9M
- **Mask ratio**: 0.75 (75% of patches masked)

### Training Hyperparameters
- **Epochs**: 100 per fold
- **Batch size**: 32
- **Learning rate**: 1e-3
- **Min LR**: 1e-6
- **Scheduler**: Cosine annealing
- **Weight decay**: 0.05
- **Optimizer**: AdamW
- **Gradient clipping**: 1.0
- **Mixed precision**: Yes (amp)

### K-Fold Configuration
- **Number of folds**: 5
- **Split strategy**: Subject-based (prevents data leakage)
- **Subjects per fold**:
  - Train: 12 subjects
  - Val: 3 subjects

## Training Progress

### Status Monitoring
```bash
# Check both experiments
cd /home/ab_students/EEG-MTP/trial_mae_SEED4
source /home/ab_students/miniconda3/bin/activate torch311
python3 monitor_comparison.py

# Check logs
tail -f train_31ch_kfold.log          # Preprocessed data
tail -f train_31ch_kfold_raw.log      # Raw data
```

### Output Directories
- **Preprocessed**: `trial_mae_SEED4/results_31ch_kfold/`
- **Raw**: `trial_mae_SEED4/results_31ch_kfold_raw/`

Each contains:
- `fold_1/`, `fold_2/`, ..., `fold_5/`: Per-fold results
  - `best_model.pt`: Best checkpoint
  - `history.json`: Training history
- `fold_splits.json`: K-fold split information
- `kfold_summary.json`: Final summary (created after all folds complete)

## Evaluation Metrics

### Primary Metrics
1. **Validation Loss**: MSE on masked patches
2. **Validation Correlation**: Pearson correlation between predicted and actual values on masked patches

### Per-Fold Results
Each fold will report:
- Best validation loss
- Best validation correlation
- Number of training/validation samples

### Final Comparison
After all folds complete, the summary will show:
- Average validation loss ± std
- Average validation correlation ± std
- Winner determination (which preprocessing strategy is better)

## Expected Timeline

### Preprocessed Data (14,280 samples)
- ~40 hours total (8 hours/fold × 5 folds)
- Currently: **Fold 2/5 in progress**

### Raw Data (7,035 samples)
- ~20 hours total (4 hours/fold × 5 folds)
- Currently: **Fold 1/5, Epoch 2/100**

## Next Steps

1. **Monitor**: Let both experiments run to completion
2. **Compare**: Use `monitor_comparison.py` to track progress
3. **Analyze**: Once complete, compare:
   - Convergence speed
   - Final correlation values
   - Stability across folds
4. **Decide**: Choose best preprocessing pipeline for STAD training
5. **STAD Training**: Train final STAD model using best MAE checkpoint

## Key Questions to Answer

1. Does PrC-1 preprocessing improve MAE reconstruction quality?
2. Is raw data sufficient, or does preprocessing add value?
3. Which approach generalizes better across subjects?
4. What is the correlation improvement with proper training?

## Files Created

### Dataset Preparation
- `New_SEED4/prepare_raw_dataset.py` - Creates multi-resolution raw data NPZ
- `/DATA/seed4/raw_data.npz` - Raw multi-resolution dataset (LR/HR/SR)

### Training Scripts
- `trial_mae_SEED4/train_mae_31ch_kfold.py` - K-fold training on preprocessed data
- `trial_mae_SEED4/train_mae_31ch_kfold_raw.py` - K-fold training on raw data

### Monitoring
- `trial_mae_SEED4/monitor_comparison.py` - Compare both experiments

## Background Processes

Running in background:
```bash
# Check processes
ps aux | grep "train_mae_31ch" | grep -v grep

# PIDs: Two main processes (one for each experiment)
```

## Notes

- **Subject-based splits**: Same 5-fold splits used for both experiments (fair comparison)
- **Normalization**: Per-sample zero mean, unit variance
- **Parallel execution**: Both experiments run simultaneously for time efficiency
- **Low initial correlation**: Expected at start of training (~0.0002)
- **Recovery**: Training can resume from checkpoints if interrupted
