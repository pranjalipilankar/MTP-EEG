# STAD Training for SEED-IV Dataset

## Overview
Super-resolution training (31→62 channels) using pretrained MAE from SEED-IV dataset.

## Key Differences from SEED Version

| Aspect | SEED | SEED-IV |
|--------|------|---------|
| **Channels** | 62 | 62 |
| **Sampling Rate** | 200 Hz | 250 Hz |
| **Window/Segment Length** | 4000 samples (20s) | 1000 samples (4s) |
| **Patch Size** | 8 | 8 |
| **Num Patches** | 500 | 125 |
| **MAE Latent Dim** | 1024 | 768 |
| **MAE Depth** | 24 | 12 |
| **Emotion Classes** | 3 | 4 |
| **Batch Size** | 32 | 64 |

## Pretrained MAE Checkpoint

Uses best checkpoint from k-fold cross-validation:
- Path: `/home/ab_students/EEG-MTP/trial_mae_SEED4/results_compare_kfold_5fold/preprocessed/fold_X/best_model.pth`
- Select the fold with highest validation correlation
- Expected correlation: 0.45-0.65

## Architecture

### MAE (Frozen until epoch 50)
- Encoder: 12 layers, 768 dim, 12 heads
- Decoder: 4 layers, 384 dim, 8 heads
- Patch size: 8
- Input: (B, 62, 1000) → Latent: (B, 125, 768)

### Spatio-Temporal Conditioning (STC)
- Input: LR EEG (B, 31, 1000)
- Output: Conditioning tokens + pooled features
- Embed dim: 768 (matches MAE)
- 8 harmonics for graph-based spectral features

### Multi-Scale Transformer Denoising (MTD)
- Input: Noisy latent (B, 125, 768)
- Output: Predicted noise
- 8 layers, 16 heads
- Uses conditioning from STC

## Dataset Structure

SEED-IV processed data (after PrC-1 preprocessing):
```
DATA/seed4/eeg_processed_data/
  {subject_id}_{session_id}_{trial_id}.npy  # Shape: (62, 1000)
```

- 15 subjects
- 3 sessions per subject
- 24 trials per session
- 62 channels per trial
- 1000 samples per trial (4 seconds @ 250Hz)

## Training Configuration

### Data Resolution Levels
- **LR (Low-Resolution)**: 16 channels (spatial downsampling from 62)
- **HR (High-Resolution)**: 31 channels (MAE trained on this)
- **SR (Super-Resolution)**: 62 channels (target output)

### Training Parameters
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 100
- Warmup: 10 epochs
- Diffusion timesteps: 1000
- Sampling steps: 50

### Loss Components
1. **Diffusion Loss**: L2 between predicted and actual noise
2. **MAE Reconstruction Loss**: After epoch 50, finetune MAE
3. **Spatial Consistency**: Encourage smooth spatial transitions

## Usage

### 1. Prepare Data
Ensure k-fold MAE training is complete:
```bash
cd /home/ab_students/EEG-MTP/trial_mae_SEED4
# Check fold results
cat results_compare_kfold_5fold/preprocessed/*/fold_results.json | grep '"test_correlation"'
```

### 2. Select Best MAE Checkpoint
```bash
# Find fold with highest test correlation
BEST_FOLD=2  # Example - replace with actual best fold
cp results_compare_kfold_5fold/preprocessed/fold_$BEST_FOLD/best_model.pth \
   /home/ab_students/EEG-MTP/New_SEED4/pretrained_mae_seed4.pth
```

### 3. Train STAD
```bash
cd /home/ab_students/EEG-MTP/New_SEED4
python3 train_stad_seed4.py \
  --mae_checkpoint pretrained_mae_seed4.pth \
  --batch_size 32 \
  --epochs 100 \
  --device cuda
```

### 4. Monitor Training
```bash
tail -f results/training.log
```

## Expected Results

### MAE Baseline (31 channels)
- Correlation: 0.45-0.65
- NMSE: 0.15-0.25
- SNR: 8-12 dB

### STAD Super-Resolution (62 channels)
- Correlation: 0.50-0.70 (higher due to spatial consistency)
- NMSE: 0.12-0.20
- SNR: 10-15 dB
- **Improvement**: 10-15% better correlation than MAE alone

## Files

- `config_seed4.py`: Configuration for SEED-IV dataset
- `train_stad_seed4.py`: Main training script
- `mae_for_eeg.py`: MAE model (shared with trial_mae_SEED4)
- `spatio_temporal_condition.py`: STC module
- `mtd_dreamdiff.py`: MTD module
- `utils.py`: Utility functions

## Notes

- SEED-IV windows are already 4 seconds, no need for segmentation
- Use subject-based k-fold cross-validation for proper evaluation
- MAE should be frozen for first 50 epochs to stabilize STAD training
- After epoch 50, jointly finetune MAE and STAD for best performance
