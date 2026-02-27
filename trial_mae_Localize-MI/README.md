# Localize-MI HD-EEG MAE Pretraining

## Overview
Masked Autoencoder (MAE) pretraining for Localize-MI high-density EEG dataset (256 channels, 8000Hz).

## Key Characteristics

| Feature | Value |
|---------|-------|
| **Channels** | 256 (HD-EEG) |
| **Sampling Rate** | 8000 Hz |
| **Epoch Length** | 2080 samples (~260ms) |
| **Patch Size** | 16 |
| **Num Patches** | 130 per epoch |
| **Total Epochs** | ~2,300 epochs (61 files × ~38 epochs/file) |
| **Subjects** | 7 subjects |

## Dataset Comparison

| Dataset | Channels | Sampling Rate | Data Type |
|---------|----------|---------------|-----------|
| **DEAP** | 32 | 128 Hz | Emotion EEG |
| **SEED** | 62 | 200 Hz | Emotion EEG |
| **Localize-MI** | 256 | 8000 Hz | HD-EEG with intracranial stimulation |

## Dataset Structure
```
DATA/Localize-MI/derivatives/epochs/
├── sub-01/
│   └── eeg/
│       ├── sub-01_task-seegstim_run-01_epochs.npy  (38, 256, 2081)
│       ├── sub-01_task-seegstim_run-02_epochs.npy
│       ├── sub-01_task-seegstim_run-03_epochs.npy
│       └── ...
├── sub-02/
│   └── eeg/
│       └── ...
└── ...
```

## Usage

```bash
cd /home/ab_students/EEG-MTP/trial_mae_Localize-MI
python train_mae_localizemi.py
```

## Configuration

Edit `config_localizemi.py` to adjust:
- `time_len`: Epoch length (default: 2080, divisible by 16)
- `patch_size`: Temporal patch size (default: 16)
- `mask_ratio`: Proportion of patches to mask (default: 0.75)
- `batch_size`: Batch size (default: 16, smaller due to 256 channels)

## Expected Results

- **Training loss**: Should drop to < 0.1 within 50 epochs
- **Correlation**: Should reach > 0.6 within 100 epochs
- **Use cases**: 
  - Source localization
  - HD-EEG super-resolution (e.g., 128→256 channels)
  - Intracerebral stimulation analysis

## Special Considerations

### High-Density EEG (256 channels)
- Requires more GPU memory
- Smaller batch size (16 vs 32)
- Each patch: 256 channels × 16 timepoints = **4096 features**

### Ultra-High Sampling Rate (8000 Hz)
- Short time windows (~260ms per epoch)
- High temporal resolution
- Ideal for precise temporal dynamics

### Ground Truth for Source Localization
- This dataset includes intracerebral stimulation data
- Pretrained encoder can be used for:
  - EEG source localization
  - Forward/inverse modeling
  - Electrode selection

## Files

- `config_localizemi.py` - Configuration for Localize-MI
- `dataset_localizemi.py` - Localize-MI data loader
- `train_mae_localizemi.py` - Training script
- `mae_for_eeg.py` - MAE model (shared)
- `trainer.py` - Training loop (shared)
- `utils.py` - Utilities (shared)

## Dataset Citation

If you use this pretrained model for your research, please cite the original dataset:

```bibtex
@article{mikulan2020localize,
  title={Simultaneous human intracerebral stimulation and HD-EEG, ground-truth for source localization methods},
  author={Mikulan, Ezequiel and others},
  journal={Scientific Data},
  year={2020}
}
```

## Notes

- Dataset automatically split into train (70%), val (15%), test (15%)
- All epochs from all runs are used for maximum training data
- The model learns ultra-high spatial and temporal resolution patterns
- Excellent for applications requiring precise localization
