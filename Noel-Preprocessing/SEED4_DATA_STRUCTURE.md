# SEED-IV Processed Data Structure

## Overview
The SEED-IV dataset contains EEG recordings for emotion recognition with 4 emotion classes.

### Quick Reference
| Aspect | Value |
|--------|-------|
| **Subjects** | 15 |
| **Sessions per subject** | 3 |
| **Total recordings** | 45 |
| **Trials per session** | 24 |
| **Total trials** | 1,080 |
| **Trial duration** | ~90-100 seconds |
| **Window size** | 4 seconds |
| **Windows per trial** | ~20-25 |
| **Total windows** | ~24,000-27,000 |
| **Channels** | 62 (10-20 system) |
| **Sampling rate (raw)** | 512 Hz |
| **Sampling rate (processed)** | 250 Hz |
| **Samples per window** | 1,000 |
| **Emotion classes** | 4 (neutral, sad, fear, happy) |

## Dataset Structure Summary

### Subjects
- **15 unique subjects** (ID: 1-15)
- Each subject participates in all 3 sessions
- File naming: `{subject_id}_{date}.mat` (e.g., `1_20160518.mat`)

### Sessions
- **3 sessions per subject** (recorded on different dates)
- **Total recordings**: 45 (15 subjects × 3 sessions)
- Sessions are organized in folders `1/`, `2/`, and `3/`

### Trials
- **24 trials per recording session**
- Each trial = one video clip showing an emotion-inducing scene
- **Total trials**: 1,080 (15 subjects × 3 sessions × 24 trials)
- Trial labels vary by session (different video presentation order)

### Time Structure
- **Raw sampling rate**: 512 Hz
- **Processed sampling rate**: 250 Hz (after downsampling)
- **Window size**: 4 seconds (matches video clip duration)
- **Window samples**: 1,000 samples per window @ 250 Hz
- **Windowing strategy**: Non-overlapping 4-second segments
- **Average trial duration**: ~90-100 seconds (varies slightly)
- **Windows per trial**: ~20-25 windows (depending on exact trial length)
- **Total windows per recording**: ~480-600 windows
- **Approximate total windows**: ~24,000-27,000 across entire dataset

### Channels
- **62 EEG channels** (10-20 international system)
- High-density coverage across scalp
- Standard montage compatible with MNE-Python

### Data Hierarchy
```
SEED-IV Dataset
├── 15 Subjects (ID: 1-15)
│   └── 3 Sessions each (different dates)
│       └── 24 Trials each (4 emotions × 6 repetitions)
│           └── ~20-25 Windows each (4-second segments)
│               └── 62 Channels
│                   └── 1000 Samples @ 250 Hz
```

### Experimental Design
- **Within-subject design**: Same subjects across 3 sessions
- **Balanced emotions**: Each emotion appears 6 times per session
- **Counterbalanced**: Trial order varies across sessions (controls order effects)
- **Repeated measures**: Track same subjects over time
- **Total recording time per session**: ~40-45 minutes (24 trials × ~90s + breaks)

## Emotion Labels
- **0**: Neutral
- **1**: Sad  
- **2**: Fear
- **3**: Happy

## Dataset Structure

### Raw Data
```
DATA/seed4/eeg_raw_data/
├── 1/          # Session 1
│   ├── 1_20160518.mat
│   ├── 2_20150915.mat
│   └── ... (15 subjects)
├── 2/          # Session 2
│   └── ... (15 subjects, same IDs, different dates)
└── 3/          # Session 3
    └── ... (15 subjects, same IDs, different dates)
```

### Processed Data (after running FinalPrC-1.py)
```
DATA/seed4/eeg_processed_data/
├── 1/          # Session 1
│   ├── 1_20160518/
│   │   ├── X_prc1.npy              # Preprocessed EEG windows (n_windows, 62, 1000)
│   │   ├── labels.npy              # Emotion labels per window (n_windows,)
│   │   ├── X_prc1_norm_stats.npy   # Normalization stats (n_windows, 2)
│   │   ├── X_prc1_reversed.npy     # Reconstructed µV-scale reference
│   │   ├── prc1_meta.json          # Metadata (fs, window_sec, label_dist, etc.)
│   │   ├── trial_labels.json       # Trial-level label info
│   │   └── inspection/             # Visual QC plots
│   └── ... (14 more subjects)
├── 2/
└── 3/
```

## File Descriptions

### X_prc1.npy
- **Shape**: `(n_windows, 62, 1000)`
- **Content**: Preprocessed EEG data
  - 62 channels (10-20 system)
  - 1000 samples per window (4 seconds @ 250 Hz)
  - Z-normalized, optionally soft-clipped
- **Usage**: Direct input to deep learning models

### labels.npy
- **Shape**: `(n_windows,)`
- **Content**: Emotion label for each 4-second window
- **Values**: 0 (neutral), 1 (sad), 2 (fear), 3 (happy)
- **Note**: Multiple windows can have the same label (from same trial)

### X_prc1_norm_stats.npy
- **Shape**: `(n_windows, 2)`
- **Content**: `[mean, std]` for each window (for denormalization)
- **Usage**: Reverse normalization: `X_orig = X_norm * std + mean`

### X_prc1_reversed.npy
- **Shape**: `(n_windows, 62, 1000)`
- **Content**: Preprocessed data reversed back to µV scale
- **Usage**: Reference for evaluating super-resolution reconstruction

### prc1_meta.json
JSON file with:
```json
{
  "target_fs": 250,
  "window_sec": 4,
  "soft_clip_applied": true,
  "soft_clip_val": 5.0,
  "n_channels": 62,
  "n_windows": 552,
  "window_samples": 1000,
  "dataset": "1_20160518",
  "session": "1",
  "n_trials": 24,
  "emotion_labels": {"0": "neutral", "1": "sad", "2": "fear", "3": "happy"},
  "label_distribution": {"0": 120, "1": 144, "2": 132, "3": 156}
}
```

### trial_labels.json
JSON file with trial-level information:
```json
{
  "trial_labels": [1, 2, 3, 0, ...],
  "emotion_mapping": {"0": "neutral", "1": "sad", "2": "fear", "3": "happy"},
  "n_trials": 24,
  "windows_per_trial": [23, 23, 23, ...]
}
```

## Trial Structure
Each subject recording contains **24 trials**:
- Each trial corresponds to one video clip showing an emotion-inducing scene
- **Trial duration**: ~90-100 seconds (continuous EEG recording during video)
- **Windowing**: Each trial is divided into multiple 4-second non-overlapping windows
- **Windows per trial**: ~20-25 windows (depending on exact trial duration)
- Trial order is the same across all subjects within a session
- Different sessions have different trial orders (counterbalanced design)

### Session-specific trial labels:
**Session 1**: `[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]`  
**Session 2**: `[2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]`  
**Session 3**: `[1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]`

### Window Labeling
- All windows extracted from a single trial receive the same emotion label
- Example: Trial 1 in Session 1 has label "1" (sad)
  - If this trial produces 23 windows, all 23 windows are labeled as "1" (sad)

## Data Loading

### Quick Start
```python
import numpy as np

# Load processed data
X = np.load('DATA/seed4/eeg_processed_data/1/1_20160518/X_prc1.npy')
labels = np.load('DATA/seed4/eeg_processed_data/1/1_20160518/labels.npy')

print(f"Data shape: {X.shape}")      # e.g., (552, 62, 1000)
print(f"Labels shape: {labels.shape}")  # e.g., (552,)
```

### Using Helper Functions
```python
from load_seed4_data import load_subject_session, load_all_data, get_data_by_emotion

# Load single subject
X, labels, meta = load_subject_session(session='1', subject='1_20160518')

# Load all data
all_data = load_all_data()

# Get only happy emotion windows
X_happy = get_data_by_emotion(session='1', subject='1_20160518', emotion='happy')

# Create train/test splits (leave-one-session-out)
from load_seed4_data import create_dataset_splits
all_data = load_all_data()
X_train, y_train, X_test, y_test = create_dataset_splits(
    all_data, 
    train_sessions=['1', '2'], 
    test_session='3'
)
```

## Preprocessing Pipeline (PrC-1)

1. **Bandpass Filter**: 0.1 - 100 Hz (IIR high-pass + FIR low-pass)
2. **Notch Filter**: 50 Hz (power line noise)
3. **Bad Channel Detection & Interpolation**: Spherical spline interpolation
4. **Downsampling**: 512 Hz → 250 Hz
5. **Windowing**: 4-second sliding windows with no overlap
6. **Global Z-normalization**: Per window across all channels
7. **Soft Clipping** (optional): `tanh(x/5.0) * 5.0`

## Channel Layout
62 channels following the 10-20 system:
```
FP1, FPZ, FP2, AF3, AF4, F7, F5, F3, F1, FZ,
F2, F4, F6, F8, FT7, FC5, FC3, FC1, FCZ, FC2,
FC4, FC6, FT8, T7, C5, C3, C1, CZ, C2, C4,
C6, T8, TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6,
TP8, P7, P5, P3, P1, PZ, P2, P4, P6, P8,
PO7, PO5, PO3, POZ, PO4, PO6, PO8, I1, O1, OZ,
O2, I2
```

## Label Distribution

### Trial Level (per session)
Each session has 24 trials with balanced emotion distribution:
- **Neutral**: 6 trials (25%)
- **Sad**: 6 trials (25%)
- **Fear**: 6 trials (25%)
- **Happy**: 6 trials (25%)

### Window Level (after preprocessing)
Windows extracted from trials (actual distribution may vary slightly):
- **Windows per trial**: ~20-25 (depending on trial duration)
- **Expected windows per emotion per session**: ~120-150 windows
- **Expected windows per emotion across dataset**: ~6,000-6,750 windows
- **Note**: Slight imbalance may occur due to variable trial durations

## Dataset Statistics
- **Subjects**: 15 (same subjects across all sessions)
- **Sessions per subject**: 3 (different recording dates)
- **Total recordings**: 45 (15 subjects × 3 sessions)
- **Trials per recording**: 24 (6 per emotion class)
- **Total trials**: 1,080 (45 recordings × 24 trials)
- **Windows per trial**: ~20-25 (non-overlapping 4-second segments)
- **Approx. total windows**: ~24,000-27,000 (depending on preprocessing)
- **Data shape per window**: (62 channels, 1000 samples)
- **Sampling rate**: 250 Hz (downsampled from 512 Hz)
- **Window duration**: 4 seconds
- **Total recording time**: ~30-33 hours (45 sessions × ~40-45 min)

### Label Balance
Per session (24 trials):
- **Neutral**: 6 trials (25%)
- **Sad**: 6 trials (25%)
- **Fear**: 6 trials (25%)
- **Happy**: 6 trials (25%)

Per window (varies due to trial duration):
- Actual windows per trial: ~20-25 depending on signal length
- Expected windows per emotion per session: ~120-150 windows
- Total windows per emotion across dataset: ~6,000-6,750 windows

## Citation
```
@article{zheng2015investigating,
  title={Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks},
  author={Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Autonomous Mental Development},
  volume={7},
  number={3},
  pages={162--175},
  year={2015},
  publisher={IEEE}
}

@article{zheng2018emotionmeter,
  title={EmotionMeter: A Multimodal Framework for Recognizing Human Emotions},
  author={Zheng, Wei-Long and Liu, Wei and Lu, Yifei and Lu, Bao-Liang and Cichocki, Andrzej},
  journal={IEEE Transactions on Cybernetics},
  volume={49},
  number={3},
  pages={1110--1122},
  year={2018},
  publisher={IEEE}
}
```

## Common Use Cases

### 1. Emotion Classification (4-class)
Traditional task: Classify each window into one of 4 emotions (neutral, sad, fear, happy).

### 2. Cross-Session Generalization
- **Train**: Sessions 1 & 2
- **Test**: Session 3
- Tests model robustness across different recording days

### 3. Subject-Independent Classification
- **Leave-one-subject-out cross-validation**
- Train on 14 subjects, test on 1 subject
- Tests generalization to new individuals

### 4. Subject-Dependent Classification
- Train and test on same subject (across sessions)
- Higher accuracy, useful for personalized BCI systems

### 5. Temporal Analysis
- Study emotion dynamics within trials
- Analyze how emotions evolve over the ~90s trial duration

### 6. Transfer Learning
- Pre-train on SEED-IV, fine-tune on other emotion datasets
- Study cross-dataset generalization

## Notes
- Each subject has 3 sessions recorded on different days
- The same 15 subjects appear across all 3 sessions (matched by subject ID, different dates)
- Labels are consistent within a session but trial order varies across sessions (counterbalancing)
- **Trial duration**: Each trial is ~90-100 seconds (full video clip + EEG recording)
- **Window size**: 4 seconds is used for feature extraction (as per ReadMe)
- **No overlap**: Windows are extracted with no overlap from each trial
- Each trial produces ~20-25 windows depending on its exact duration
- All windows from the same trial have the same emotion label
- Sessions are counterbalanced to control for order effects and learning
- Videos used for emotion induction are professionally selected film clips

## Data Quality Notes
- Bad channels are automatically detected and interpolated (spherical spline)
- Line noise (50 Hz) is removed via notch filtering
- Signals are bandpass filtered (0.1-100 Hz) to remove DC drift and high-frequency noise
- Z-normalization ensures consistent scale across subjects and sessions
- Quality control plots are generated during preprocessing for visual inspection
