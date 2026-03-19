# SEED Dataset MAE Pretraining

## Overview
Masked Autoencoder (MAE) pretraining for SEED EEG dataset (62 channels, 200Hz).

## Key Differences from DEAP

| Feature | DEAP | SEED |
|---------|------|------|
| **Channels** | 32 | 62 |
| **Sampling Rate** | 128 Hz | 200 Hz |
| **Trial Length** | 8064 samples (63s) | 104000 samples (520s) |
| **Segmentation** | Full trials | 4000 samples (20s segments) with 50% overlap |
| **Patch Size** | 16 | 8 |
| **Num Patches** | 504 | 500 per segment |

## Dataset Structure
```
DATA/SEED_processed/
├── train/
│   ├── sub1.npy  (7, 62, 104000)
│   ├── sub2.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Usage

```bash
cd /home/ab_students/EEG-MTP/trial_mae_SEED
python train_mae_seed.py
```

## Configuration

Edit `config_seed.py` to adjust:
- `segment_length`: Length of each training segment (default: 4000 = 20s)
- `segment_overlap`: Overlap between segments (default: 0.5 = 50%)
- `patch_size`: Temporal patch size (default: 8)
- `mask_ratio`: Proportion of patches to mask (default: 0.75)

## Expected Results

- **Training loss**: Should drop to < 0.1 within 50 epochs
- **Correlation**: Should reach > 0.6 within 100 epochs
- **Use case**: Pretrained encoder can be used for 31→62 channel super-resolution

## Files

- `config_seed.py` - Configuration for SEED dataset
- `dataset_seed.py` - SEED data loader with segmentation
- `train_mae_seed.py` - Training script
- `mae_for_eeg.py` - MAE model (shared with DEAP)
- `trainer.py` - Training loop (shared)
- `utils.py` - Utilities (shared)

## Notes

- SEED trials are very long (520s), so we use 20s segments with 50% overlap
- This creates ~50 segments per trial, significantly increasing training data
- The model learns temporal-spatial patterns that generalize well for super-resolution
