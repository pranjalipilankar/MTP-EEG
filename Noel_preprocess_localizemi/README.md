# PrC-1: EEG Preprocessing for Localize-MI

This folder provides a Localize-MI-specific PrC-1 pipeline, created to mirror the structure used in:
- `Noel-Preprocessing` (SEED)
- `Noel_preprocess_deap` (DEAP)

## Files

- `FinalPrC-1.py`
  - Forward preprocessing for Localize-MI epochs
  - Input: `DATA/Localize-MI/derivatives/epochs/sub-*/eeg/*_epochs.npy`
  - Output per subject: `X_prc1.npy`, `X_prc1_norm_stats.npy`, `X_prc1_reversed.npy`, `prc1_meta.json`

- `reconstruct_prc1.py`
  - Reconstructs SR model outputs back to uV scale using PrC-1 stats and metadata

- `load_localizemi_data.py`
  - Utility loader for processed subject-wise PrC-1 outputs

## Localize-MI Defaults in This Pipeline

- Sampling rate: `8000 Hz`
- Channels: `256`
- Epoch length: `2080` samples
- Bandpass: `1-100 Hz`
- Notch filters: `50, 100, 150, 200 Hz`
- Normalization: global z-score per epoch window
- Optional soft clipping: enabled (`a = 5.0`)

## Run

```bash
cd /home/ab_students/EEG-MTP/Noel_preprocess_localizemi
python FinalPrC-1.py
```

## Output Layout

Outputs are saved under:

`/home/ab_students/EEG-MTP/DATA/Localize-MI/derivatives/epochs_prc1/sub-XX/`

Each subject folder contains:
- `X_prc1.npy`
- `X_prc1_norm_stats.npy`
- `X_prc1_reversed.npy`
- `prc1_meta.json`
- `window_index.json`
