# Enhanced test_stad_preprocessed.py - Data & Processing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: Checkpoint Paths + Data Path + CLI Arguments              │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1️⃣ LOAD DATASET                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  • Load test data (NPZ or folder format)                           │
│  • Filter by test subjects (fold-based split)                      │
│  • Create train/val/test split for 15 subjects                     │
│  • Instantiate SEED4PreprocessedDataset                            │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2️⃣ DATASET VALIDATION ✨ NEW ✨                                  │
├─────────────────────────────────────────────────────────────────────┤
│  compute_dataset_statistics()                                       │
│  ├─ Sample 50 windows from dataset                                 │
│  ├─ Compute statistics per resolution:                             │
│  │  ├─ LR:  16 channels × 1000 samples                            │
│  │  ├─ HR:  31 channels × 1000 samples                            │
│  │  └─ SR:  62 channels × 1000 samples                            │
│  └─ Validate shapes & distributions                               │
│                                                                     │
│  print_dataset_report()                                             │
│  └─ Pretty-print statistics to console                             │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3️⃣ LOAD MODEL COMPONENTS                                         │
├─────────────────────────────────────────────────────────────────────┤
│  • Load MAE encoder (31-channel) from checkpoint                   │
│  • Build STADModel with MAE encoder                                │
│  • Load STAD checkpoint weights                                    │
│  • Set models to eval mode                                         │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4️⃣ EVALUATION LOOP                                               │
├─────────────────────────────────────────────────────────────────────┤
│  For each batch:                                                    │
│  ├─ LR input (16ch) ────┐                                          │
│  ├─ HR input (31ch) ────┼─→ STADModel ──→ Predicted SR (62ch)    │
│  └─ SR target (62ch) ───┘                                          │
│                                                                     │
│  Compute metrics (per-sample):                                      │
│  ├─ PCC:  Pearson Correlation Coefficient                         │
│  ├─ NMSE: Normalized Mean Squared Error                           │
│  └─ SNR:  Signal-to-Noise Ratio (dB)                              │
│                                                                     │
│  Collect losses & metrics into lists                               │
│  Store predictions for later visualization                         │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5️⃣ METRICS AGGREGATION                                           │
├─────────────────────────────────────────────────────────────────────┤
│  mean_pcc  = avg(pcc_scores)                                       │
│  mean_nmse = avg(nmse_scores)                                      │
│  mean_snr  = avg(snr_scores)                                       │
│  mean_diff = avg(diff_losses)  [or NaN if sampling mode]          │
│  mean_sr   = avg(sr_losses)                                        │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ├──→ Print to console
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6️⃣ COMPREHENSIVE VISUALIZATIONS ✨ NEW ✨                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 VISUALIZATION 1: Metrics Distribution                          │
│  ├─ Input: pcc_scores, nmse_scores, snr_scores (lists)           │
│  ├─ output: metrics_distribution.png                               │
│  └─ 3 histograms with mean overlay lines                          │
│                                                                     │
│  📉 VISUALIZATION 2: Loss Curves                                   │
│  ├─ Input: diff_losses, sr_losses (lists)                        │
│  ├─ Output: loss_curves.png                                        │
│  └─ 2 plots: Diffusion loss + SR L1 loss per batch                │
│                                                                     │
│  📈 VISUALIZATION 3: EEG Comparison                                │
│  ├─ Input: predicted_sr, target_sr (numpy arrays)                │
│  ├─ Output: eeg_comparison.png                                     │
│  ├─ 3 samples × 5 channels = 15 subplots                          │
│  └─ Overlay: Target (black) vs Predicted (blue)                   │
│                                                                     │
│  🧠 VISUALIZATION 4: Brain Topomaps                               │
│  ├─ Input: predicted_sr, target_sr                                │
│  ├─ Output: topomap_comparison.png                                 │
│  ├─ Process:                                                       │
│  │  ├─ Average SR across time dimension                           │
│  │  ├─ Set up MNE info (62-channel, synthetic positions)         │
│  │  ├─ Create head montage                                        │
│  │  └─ Plot_topomap with RdBu_r colormap                        │
│  └─ 2 topographies: Left=Target, Right=Predicted                 │
│                                                                     │
│  All 4 PNGs saved to: visualization_dir/                          │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7️⃣ OPTIONAL: SAVE PREDICTIONS & METADATA                         │
├─────────────────────────────────────────────────────────────────────┤
│  • pred_sr.npy          (if --save_sr_output_path)                │
│  • target_sr.npy        (if --save_target_output_path)            │
│  • metadata.npz         (if --save_test_metadata_path)            │
│  • norm_stats.npy       (if --save_norm_stats_path)               │
│  • prc1_meta.json       (if --save_prc1_meta_path)                │
│  • grouped_output/      (if --save_grouped_output_dir)            │
└────┬──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ COMPLETE                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Output directories:                                                │
│  ├─ test_visualizations/         (new - 4 PNGs)                  │
│  │  ├─ metrics_distribution.png                                   │
│  │  ├─ loss_curves.png                                            │
│  │  ├─ eeg_comparison.png                                         │
│  │  └─ topomap_comparison.png                                     │
│  ├─ test_figures_preprocessed/   (existing - per-sample figs)     │
│  └─ results/                     (optional outputs)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Statistics Report Flowchart

```
DATASET VALIDATION
│
├─→ Load 50 samples from dataset
│   ├─ Extract SR EEG (62ch × 1000 samples)
│   ├─ Extract HR EEG (31ch × 1000 samples)
│   └─ Extract LR EEG (16ch × 1000 samples)
│
├─→ Compute statistics for each resolution:
│   ├─ Shape: (n_samples, channels, time_steps)
│   ├─ Mean, Std, Min, Max
│   ├─ Signal range (max - min per window)
│   └─ Channel count validation
│
└─→ Print formatted report to console
    ├─ Total samples in dataset
    ├─ Per-resolution stats
    └─ Data quality indicators
```

---

## Visualization Pipeline

```
PREDICTIONS ARRAY (N × 62 × 1000)
│
├──────────────────────┬──────────────────────┬──────────────────────┐
│                      │                      │                      │
▼                      ▼                      ▼                      ▼
│                      │                      │                      │
├─ Flatten to  ├─ Keep shape   ├─ Average     ├─ Average
│  (N, 62000)  │  per sample   │  over time   │  first 3 samples
│              │               │              │
▼              ▼               ▼              ▼

PCC/NMSE   Loss Curves   Frequency   Topomaps
Histograms

│         │        │        │
├─ Loop   ├─ Plot ├─ Welch ├─ MNE setup
│  batches │  per  │  PSD   │  Colormaps
│         │  batch│        │  Interpolate
└─────────┴────────┴────────┴────────────
          │                 │
          └──→ MATPLOTLIB ←─┘
              │
              ├─ GridSpec for layout
              ├─ Subplots
              ├─ Colorbars
              └─ Save PNG at 150 DPI
                     ▼
              [visualization_dir/]
              ├─ metrics_distribution.png
              ├─ loss_curves.png
              ├─ eeg_comparison.png
              └─ topomap_comparison.png
```

---

## CLI Argument Flow

```
python test_stad_preprocessed.py \\
  --mae_checkpoint <path>         ──→ Load MAE weights
  --stad_checkpoint <path>         ──→ Load STAD weights
  --data_path <path>               ──→ Load test data
  --test_fold 0                    ──→ Select test subjects
  --visualization_dir results      ──→ Output location ✨ NEW ✨
  --save_fig_dir figs              ──→ Per-sample figures
  --batch_size 32                  ──→ Inference batch size
  --device cuda                    ──→ GPU/CPU selection

                ║
                ▼
        ┌──────────────────┐
        │  Argument Parser │
        └──────┬───────────┘
               │
        ┌──────▼───────────────────────┐
        │  evaluate(args)              │
        │                              │
        │  uses:                       │
        │  • args.visualization_dir    │
        │  • args.mae_checkpoint       │
        │  • args.stad_checkpoint      │
        │  • args.data_path            │
        │  • etc.                      │
        └──────────────────────────────┘
```

---

## Data Shapes Through Pipeline

```
STEP 1: Dataset Loading
└─ LR:  (batch=32, channels=16, time=1000)
└─ HR:  (batch=32, channels=31, time=1000)
└─ SR:  (batch=32, channels=62, time=1000)

STEP 2: Model Inference
├─ MAE encoder: HR (32,31,1000) → latent (32,125,768)
├─ STAD model: latent + LR → noise prediction
└─ Reconstruct: noise + latent → Pred SR (32,62,1000)

STEP 3: Metrics Computation
├─ PCC: (32,) → scalar via mean
├─ NMSE: (32,) → scalar via mean
├─ SNR: (32,) → scalar via mean
└─ Collect into lists across all batches

STEP 4: Visualization
├─ Metrics list: pcc_scores (N,) → Histogram
├─ Predictions: (total,62,1000) → Compare samples
└─ PSD: (total,62,1000) → (total,62,freq) → Topomaps
```

---

## File I/O Structure

```
Working Directory/
│
├─ test_stad_preprocessed.py         (main script)
│
├─ test_visualizations/              ✨ NEW ✨
│  ├─ metrics_distribution.png        (50-100 KB)
│  ├─ loss_curves.png                 (30-50 KB)
│  ├─ eeg_comparison.png              (100-200 KB)
│  └─ topomap_comparison.png          (80-150 KB)
│
├─ test_figures_preprocessed/        (existing)
│  ├─ eeg_signal_sample_000.png
│  ├─ eeg_signal_sample_001.png
│  └─ ...
│
└─ results/                          (optional)
   ├─ pred_sr.npy
   ├─ target_sr.npy
   ├─ metadata.npz
   ├─ norm_stats.npy
   └─ prc1_meta.json
```

---

## Error Handling Flow

```
MNE Library Check
│
├─ Is MNE installed?
│  ├─ YES → Import successfully
│  │        Use standard montages
│  │        Generate topomap
│  └─ NO  → Import fails (caught)
│          Set HAS_MNE = False
│          Skip topomap section
│          Continue with other visualizations
│
├─ Data validation
│  ├─ Empty dataset? → ValueError early
│  ├─ Wrong shape? → Warning + skip sample
│  └─ NaN values? → Filter before plotting
│
└─ File I/O
   ├─ Directory exists? → Create with exist_ok
   ├─ Write permission? → Exception if failed
   └─ Disk space? → Could add pre-flight check
```

---

## Summary

✅ **Data flows through 7 major stages:**
1. Load and validate dataset
2. Print statistics
3. Load models
4. Run evaluation loop
5. Aggregate metrics
6. Generate 4 comprehensive visualizations
7. Save all outputs

✅ **Each visualization aims to answer:**
- **Metrics dist**: Is model consistent across samples?
- **Loss curves**: Is inference converging properly?
- **EEG comparison**: Are predictions faithful to target?
- **Topomaps**: Is spatial reconstruction accurate?

✅ **Backwards compatible** - all new features optional
