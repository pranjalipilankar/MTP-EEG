# STAD Training for SEED Dataset

## Overview
Super-resolution training (31→62 channels) using pretrained MAE from SEED dataset.

## Key Differences from DEAP Version

| Aspect | DEAP | SEED |
|--------|------|------|
| **Channels** | 32 (HR), 16 (LR) | 62 (HR), 31 (LR) |
| **Sampling Rate** | 128 Hz | 200 Hz |
| **Segment Length** | 400 samples (~3.1s) | 4000 samples (20s) |
| **Patch Size** | 4 | 8 |
| **Num Patches** | 100 | 500 |
| **MAE Latent Dim** | 256 | 1024 |
| **MAE Depth** | 6 | 24 |
| **Batch Size** | 32 | 16 (reduced for 62 channels) |

## Pretrained MAE Checkpoint

Uses: `/home/ab_students/EEG-MTP/trial_mae_SEED/results/best_checkpoint.pth`

**MAE Performance:**
- Correlation: **0.743** (excellent reconstruction quality)
- Epoch: 71 (from training log)
- Loss: 0.4495

## Architecture

### MAE (Frozen until epoch 50)
- Encoder: 24 layers, 1024 dim, 16 heads
- Decoder: 8 layers, 512 dim, 16 heads
- Patch size: 8
- Input: (B, 62, 4000) → Latent: (B, 500, 1024)

### Spatio-Temporal Conditioning (STC)
- Input: LR EEG (B, 31, 4000)
- Output: Conditioning tokens + pooled features
- Embed dim: 1024 (matches MAE)
- 8 harmonics for graph-based spectral features

### Multi-Scale Transformer Denoising (MTD)
- Input: Noisy latent (B, 500, 1024)
- Output: Predicted noise
- 8 layers, 16 heads
- Uses conditioning from STC

## Dataset Structure

```
DATA/SEED_processed/
├── train/
│   ├── sub1.npy   (num_trials, 62, 104000)
│   ├── sub2.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each trial is 520 seconds @ 200Hz. For STAD training:
- Segment into 20-second windows (4000 samples)
- Non-overlapping segments
- Downsample to 31 channels for LR
- Bandpass filter: 1-40 Hz
- Per-segment per-channel normalization

## Usage

```bash
cd /home/ab_students/EEG-MTP/New_SEED
python train_stad_seed.py
```

## Training Process

1. **Epochs 0-49**: Train STC + MTD with frozen MAE
   - Only diffusion components learn
   - LR: 2e-4
   
2. **Epochs 50-299**: Fine-tune entire model
   - Unfreeze MAE encoder
   - LR: 2e-5 (10x smaller)
   - Full end-to-end optimization

## Expected Results

Based on DEAP version performance:

| Metric | Target | Description |
|--------|--------|-------------|
| **PCC** | > 0.6 | Pearson correlation (channel-wise) |
| **SNR** | > 10 dB | Signal-to-noise ratio |
| **NMSE** | < 0.2 | Normalized mean squared error |
| **Training Loss** | 0.2-0.8 | MSE on noise prediction |

## Output Files

- `best_stad_seed.pt` - Best model by validation loss
- `checkpoint_seed_epoch_*.pt` - Regular checkpoints every 20 epochs

## Memory Requirements

- **GPU Memory**: ~10-12 GB (62 channels, batch_size=16)
- Reduce batch size to 8 if OOM occurs

## Key Improvements Over DEAP

1. **Higher channel resolution**: 62 vs 32 channels
2. **Better pretrained MAE**: 0.743 correlation vs typical 0.6-0.7
3. **Longer temporal context**: 20s segments vs 3.1s
4. **More patches**: 500 vs 100 (richer latent representation)

## Troubleshooting

### If loss doesn't decrease:
- Check MAE checkpoint loaded correctly
- Verify channel positions match SEED layout
- Ensure data normalization is consistent

### If reconstruction quality is poor:
- Increase sampling steps (50 → 100)
- Fine-tune longer (extend from epoch 50)
- Check LR/HR channel correspondence

### If OOM:
- Reduce batch_size to 8 or 4
- Reduce MTD layers from 8 to 6
- Use gradient checkpointing

## Citation

If using this for SEED emotion recognition research, cite:
```bibtex
@article{zheng2015investigating,
  title={Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks},
  author={Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Autonomous Mental Development},
  year={2015}
}
```
