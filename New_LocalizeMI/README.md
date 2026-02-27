# STAD Training for Localize-MI HD-EEG Dataset

## Overview
Super-resolution training (128→256 channels) using pretrained MAE from Localize-MI dataset.
This is the most challenging STAD configuration due to ultra-high channel density and sampling rate.

## Key Differences from Other Datasets

| Aspect | DEAP | SEED | **Localize-MI** |
|--------|------|------|-----------------|
| **Channels** | 32 (HR), 16 (LR) | 62 (HR), 31 (LR) | **256 (HR), 128 (LR)** |
| **Sampling Rate** | 128 Hz | 200 Hz | **8000 Hz** |
| **Epoch Length** | 400 samples (~3.1s) | 4000 samples (20s) | **2080 samples (~260ms)** |
| **Patch Size** | 4 | 8 | **16** |
| **Num Patches** | 100 | 500 | **130** |
| **MAE Latent Dim** | 256 | 1024 | **1024** |
| **MAE Depth** | 6 | 24 | **24** |
| **Batch Size** | 32 | 16 | **8** |
| **Challenge Level** | Easy | Medium | **EXTREME** |

## Pretrained MAE Checkpoint

Uses: `/home/ab_students/EEG-MTP/trial_mae_Localize-MI/results/best_checkpoint.pth`

**Expected MAE Performance:**
- Correlation: > 0.45 (challenging due to 256 channels)
- Ultra-high spatial resolution (256 channels)
- Ultra-high temporal resolution (8000 Hz)

## Architecture

### MAE (Frozen until epoch 50)
- Encoder: 24 layers, 1024 dim, 16 heads
- Decoder: 8 layers, 512 dim, 16 heads
- Patch size: 16
- Input: (B, 256, 2080) → Latent: (B, 130, 1024)

### Spatio-Temporal Conditioning (STC)
- Input: LR EEG (B, 128, 2080)
- Output: Conditioning tokens + pooled features
- Embed dim: 1024 (matches MAE)
- 8 harmonics for graph-based spectral features
- Larger patch size (32) for HD-EEG complexity

### Multi-Scale Transformer Denoising (MTD)
- Input: Noisy latent (B, 130, 1024)
- Output: Predicted noise
- 8 layers, 16 heads
- Uses conditioning from STC

## Dataset Structure

```
DATA/Localize-MI/derivatives/epochs/
├── sub-01/
│   └── eeg/
│       ├── sub-01_task-seegstim_run-01_epochs.npy  (38, 256, 2081)
│       ├── sub-01_task-seegstim_run-02_epochs.npy
│       └── ...
├── sub-02/
│   └── eeg/
│       └── ...
└── ... (7 subjects total)
```

## Usage

```bash
cd /home/ab_students/EEG-MTP/New_LocalizeMI
python train_stad_localizemi.py
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

| Metric | Target | Description |
|--------|--------|-------------|
| **PCC** | > 0.45 | Pearson correlation (very challenging with 256 channels) |
| **SNR** | > 5 dB | Signal-to-noise ratio |
| **NMSE** | < 0.30 | Normalized mean squared error |
| **Training Loss** | 0.3-1.0 | MSE on noise prediction |

**Note:** Lower targets than DEAP/SEED due to extreme complexity:
- 256 channels = 4x more than SEED, 8x more than DEAP
- 8000 Hz = 40x faster than DEAP, 62x faster than typical EEG
- Only ~260ms of data per epoch (vs 20s for SEED)

## Output Files

- `best_stad_localizemi.pt` - Best model by validation loss
- `checkpoint_localizemi_epoch_*.pt` - Regular checkpoints every 20 epochs

## Memory Requirements

- **GPU Memory**: ~14-18 GB (256 channels, batch_size=8)
- Reduce batch size to 4 if OOM occurs
- May need A100 or H100 for comfortable training

## Challenges & Solutions

### Challenge 1: Extreme Channel Density (256 channels)
**Solution:**
- Smaller batch size (8 vs 16/32)
- Larger STC patch size (32 vs 16)
- More STC transformer layers (6 vs 4)

### Challenge 2: Ultra-High Sampling Rate (8000 Hz)
**Solution:**
- Shorter epochs (~260ms vs 3-20s)
- Wider bandpass filter (1-100 Hz vs 1-40 Hz)
- More patches despite short duration (130 patches)

### Challenge 3: Limited Data Per Sample
**Solution:**
- Use all 7 subjects × ~40 epochs each = ~2300 total epochs
- Heavy data augmentation may be needed
- Consider longer sequences if memory permits

## Key Improvements Over DEAP/SEED

1. **Unprecedented spatial resolution**: 256 channels
2. **Unprecedented temporal resolution**: 8000 Hz
3. **Ground truth for source localization**: Intracerebral stimulation
4. **Ideal for**: Forward/inverse modeling, electrode optimization

## Use Cases

This trained STAD model enables:

1. **EEG Source Localization**
   - Ground truth from intracerebral stimulation
   - Validate inverse solutions
   - Benchmark localization algorithms

2. **HD-EEG Super-Resolution**
   - Upsample 128→256 channels
   - Electrode selection and optimization
   - Virtual electrode placement

3. **Forward/Inverse Modeling**
   - Test conductivity models
   - Head model validation
   - Volume conductor analysis

4. **Temporal Super-Resolution**
   - Leverage 8000 Hz for precise timing
   - Event-related potential refinement
   - Spike detection and analysis

## Troubleshooting

### If loss doesn't decrease:
- Verify MAE checkpoint loaded correctly
- Check channel positions for 256-channel grid
- Ensure normalization is per-channel
- Try smaller learning rate (1e-4)

### If reconstruction quality is poor:
- Train longer (>300 epochs may be needed)
- Increase diffusion sampling steps
- Check for NaN in gradients
- Consider pre-trained weights from SEED

### If OOM:
- Reduce batch_size to 4 or even 2
- Reduce MTD/STC layers
- Use gradient checkpointing
- Clear cache more frequently

### If training is too slow:
- Use mixed precision (already enabled)
- Reduce num_workers if I/O bound
- Consider distributed training across GPUs

## Comparison with Literature

| Aspect | This Work | Typical HD-EEG |
|--------|-----------|----------------|
| Channels | 256 | 64-128 |
| Sampling Rate | 8000 Hz | 500-2000 Hz |
| Super-Resolution | 128→256 | N/A (not common) |
| Ground Truth | Intracranial | None |
| Application | Source localization | General EEG analysis |

## Citation

If you use this for research on Localize-MI, cite the original dataset:

```bibtex
@article{mikulan2020localize,
  title={Simultaneous human intracerebral stimulation and HD-EEG, ground-truth for source localization methods},
  author={Mikulan, Ezequiel and others},
  journal={Scientific Data},
  year={2020}
}
```

## Notes

- This is an extremely challenging task due to 256 channels + 8000 Hz
- May require significant computational resources
- Consider starting with fewer channels (64→128) if this is too demanding
- Results may be more variable than DEAP/SEED due to complexity
- Patience and careful hyperparameter tuning are essential
