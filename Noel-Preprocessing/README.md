# PrC-1: EEG Preprocessing for Diffusion-Based Spatial Super-Resolution

Preprocessing pipeline for SEED-IV EEG data, designed to prepare clean, normalized input for diffusion-based spatial super-resolution models while preserving signal fidelity for downstream tasks like source localization (beamformers, eLORETA).

---

## Scripts

| Script | Purpose |
|--------|---------|
| `Final PrC-1.py` | Forward pipeline: raw `.mat` → preprocessed `.npy` |
| `reconstruct_prc1.py` | Inverse pipeline: SR model output → reconstructed µV EEG |

---

## Final PrC-1.py

### Pipeline

```
Raw EEG (62ch, 512 Hz)
  │
  ├─ IIR high-pass (0.1 Hz, 4th-order Butterworth, sosfiltfilt)
  ├─ FIR low-pass (100 Hz, 337-tap, 5 Hz transition band)
  ├─ 50 Hz notch filter (Q=30)
  ├─ Bad channel detection & spherical spline interpolation (MNE)
  ├─ Downsample 512 → 250 Hz (polyphase)
  ├─ 4s non-overlapping windows
  ├─ Per-window global z-normalization
  └─ Optional soft clipping (tanh, a=5.0)
```

### Configuration

All settings are at the top of the file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RAW_FS` | 512 | Raw sampling rate (Hz) |
| `TARGET_FS` | 250 | Output sampling rate (Hz) |
| `HPF` | 0.1 | High-pass cutoff (Hz) |
| `LPF` | 100 | Low-pass cutoff (Hz) |
| `NOTCH_FREQ` | 50 | Mains frequency for notch (Hz) |
| `NOTCH_Q` | 30 | Notch filter Q-factor |
| `WINDOW_SEC` | 4 | Window length (seconds) |
| `SOFT_CLIP_VAL` | 5.0 | Soft-clip threshold (σ units) |
| `ENABLE_SOFT_CLIP` | True | Enable tanh soft clipping |
| `SPIKE_SAFE_MODE` | False | Overrides soft clip to off when True |
| `ENABLE_BAD_CHANNEL_INTERP` | True | Enable bad channel detection & interpolation |
| `ENABLE_VALIDATION` | False | Run full validation suite after processing |
| `FLAT_STD_THRESH` | 1e-6 | Flatline detection threshold |
| `VAR_Z_THRESH` | 5.0 | Variance outlier z-score threshold |
| `CORR_THRESH` | 0.15 | Min mean absolute correlation for a channel |
| `MAX_BAD_CHANNELS` | 0.2 | Max fraction of channels allowed to be bad per trial |
| `MAT_FILE` | (path) | Path to SEED-IV `.mat` file |

### Outputs

Saved to `Final_PrC-1_Outputs/<dataset_name>/`:

| File | Shape | Description |
|------|-------|-------------|
| `X_prc1.npy` | (N, 62, 1000) | Preprocessed windows (z-normed, optionally soft-clipped) |
| `X_prc1_reversed.npy` | (N, 62, 1000) | Reversed (reconstructed) reference in µV — for evaluation |
| `X_prc1_norm_stats.npy` | (N, 2) | Per-window [µ, σ] for reconstruction |
| `prc1_meta.json` | — | Pipeline metadata (fs, window size, soft clip config, etc.) |
| `inspection/` | — | Stage-by-stage plots for one representative window |

### Filter Design

- **High-pass (0.1 Hz)**: 4th-order Butterworth IIR via `sosfiltfilt`. Uses IIR instead of FIR because a 0.1 Hz FIR would require ~16,900 taps at 512 Hz. SOS form ensures numerical stability; `sosfiltfilt` gives zero-phase response.
- **Low-pass (100 Hz)**: FIR with `firwin`, ~337 taps (5 Hz transition bandwidth). Applied via `filtfilt` for zero-phase.
- **Notch (50 Hz)**: IIR notch with Q=30, applied via `filtfilt`. Only the fundamental is notched; the 2nd harmonic (100 Hz) is at the LPF cutoff edge and already attenuated.

### Bad Channel Detection

Three criteria (any triggers interpolation):
1. **Flatline**: std < 1e-6
2. **Variance outlier**: |z-score of variance| > 5.0
3. **Uncorrelated**: mean |correlation| with other channels < 0.15

If >20% of channels are flagged, the trial is treated as globally noisy and no interpolation is done. Interpolation uses MNE's spherical spline method with the `standard_1005` montage. SEED-IV labels CB1/CB2 are mapped to I1/I2 per Oostenveld & Praamstra (2001).

### Validation Suite

Enabled via `ENABLE_VALIDATION = True`. Runs 15 checks with plots:

| # | Check | What it verifies |
|---|-------|------------------|
| 1 | Cross-correlation lag | Zero-sample delay through pipeline |
| 2 | Phase preservation | Quantitative phase QC (MAE, PLV, circular correlation) |
| 3 | PSD comparison | Pearson r, % power change, KS test, log spectral distance |
| 4 | Band-power ratios | Delta/theta/alpha/beta/gamma preservation |
| 5 | Topography correlation | Spatial pattern preservation |
| 6 | Corr-matrix similarity | Channel correlation structure preserved |
| 7 | Distribution symmetry | Skewness, kurtosis; raw vs reconstructed histogram |
| 8 | µ/σ stability | Per-window normalization parameter ranges |
| 9 | SR learnability proxy | R² of predicting ch0 from remaining channels |
| 10 | HF noise floor | Power in 60–80 Hz band |
| 11 | Bad channel diagnosis | Per-trial and union counts |
| 12 | Statistical tests | Shapiro-Wilk, t-tests, Wilcoxon, Cohen's d, KS vs N(0,1) |
| 13 | SNR | In-band vs out-of-band, per-channel gain |
| 14 | Per-channel distortion | Interpolated vs non-interpolated channel correlation/MSE |
| 15 | Extra distribution checks | KS, Wasserstein, percentile comparison, QQ plot, variance ratio |

All plots are saved to `<output_dir>/validation/`.

---

## reconstruct_prc1.py

### Purpose

Inverts PrC-1 transformations on super-resolution model output to recover µV-scale EEG.

### Reconstruction Pipeline

```
SR Model Output (N, C_hr, T)
  │
  ├─ Invert soft clipping: arctanh (optional)
  └─ Invert z-normalization: x * σ + µ (per window)
  │
  → Reconstructed EEG in µV (N, C_hr, T)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SR_OUTPUT_PATH` | `""` | Path to SR model `.npy` output (required) |
| `PRC1_DIR` | `""` | PrC-1 output directory containing stats + meta (required) |
| `OUTPUT_PATH` | `""` | Output path (auto: `<sr_dir>/reconstructed.npy`) |
| `INVERT_SOFT_CLIP` | False | Invert soft clipping if it was applied |
| `SAVE_CONTINUOUS` | False | Also save windows concatenated into continuous signal |

### Usage

1. Set `SR_OUTPUT_PATH` and `PRC1_DIR` in the config section
2. Run: `python reconstruct_prc1.py`

### Outputs

| File | Shape | Description |
|------|-------|-------------|
| `reconstructed.npy` | (N, C_hr, T) | Reconstructed signal in µV |
| `reconstructed_continuous.npy` | (C_hr, N×T) | Continuous signal (if `SAVE_CONTINUOUS = True`) |

---

## Dependencies

```
numpy
scipy
scikit-learn
matplotlib
mne
```

## Dataset

**SEED-IV** — 62-channel EEG, ESI NeuroScan system, 512 Hz, recorded at SJTU (50 Hz mains). Each `.mat` file contains trials as `cz_eeg1` through `cz_eeg24`.

---

## Evaluation: Correct Comparison Workflow

Pointwise metrics (RMSE, MAE, correlation, etc.) must compare signals that have passed through the **same forward→inverse path**.

### Correct

```
                      PrC-1                    Reversal
  Raw EEG ──────────► X_prc1.npy ────────────► X_prc1_reversed.npy   (reference)
                          │
                      SR Model
                          │
                          ▼                    Reversal
                     SR output  ────────────► reconstructed.npy      (SR result)

  Evaluate:  RMSE( X_prc1_reversed.npy , reconstructed.npy )  ✓
```

Both `X_prc1_reversed.npy` and `reconstructed.npy` are in µV and have been through the same PrC-1 forward + inverse pipeline. Any preprocessing distortion (filtering, interpolation, soft clip approximation) cancels out, isolating the SR model's contribution.

### Incorrect — Do NOT Do This

```
  RMSE( raw EEG , reconstructed.npy )  ✗
```

> **Warning**: Comparing the SR model's reversed output directly against the raw EEG conflates preprocessing distortion with SR model error. The PrC-1 pipeline applies filtering, bad channel interpolation, and downsampling that irreversibly alter the signal. Comparing against raw would attribute these expected changes to the SR model, inflating error metrics.

### Summary

| Reference | SR Output | Fair? | Why |
|-----------|-----------|-------|-----|
| `X_prc1_reversed.npy` | `reconstructed.npy` | **Yes** | Same forward→inverse path |
| Raw EEG | `reconstructed.npy` | **No** | Includes irreversible preprocessing distortion |
