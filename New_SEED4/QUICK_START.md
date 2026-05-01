# STAD Raw Evaluation - Quick Navigation Guide

## Overview
Complete evaluation of STAD model on SEED-IV raw data with:
- ✅ Data leakage prevention (KFold-based subject filtering)
- ✅ 1,407 test samples processed (subjects 7, 10, 15)
- ✅ Three output formats (NPZ, PNG, subject-wise structure)
- ✅ Comprehensive documentation

---

## 📊 Main Output Files

### 1. **EVALUATION_REPORT.md** (This Directory)
→ **START HERE** for comprehensive overview  
Content:
- Executive summary of all findings
- Detailed metrics analysis
- Data leakage prevention confirmation
- Usage guide with code examples
- Recommendations for next steps

**Size**: ~45 KB | **Read time**: 10-15 minutes

### 2. **stad_raw_evaluation/** (Aggregated Results)
→ Batch-level metrics and visualizations

**Contents**:
```
stad_raw_evaluation/
├── README.md                    ← Per-batch data format
├── results_summary.npz          ← 665 MB (all metrics)
└── ten_samples_comparison.png   ← 1.8 MB (visualization)
```

**Files to use**:
- `results_summary.npz`: For batch-level analysis
- `ten_samples_comparison.png`: For quick visual check
- `README.md`: For detailed metric definitions

### 3. **stad_raw_subject_output/** (Subject-Wise Data)
→ Individual trial predictions (3 subjects × 469 trials each)

**Contents**:
```
stad_raw_subject_output/
├── README.md          ← Per-trial data format
├── subject_7/         ← 222 MB (469 trials)
├── subject_10/        ← 222 MB (469 trials)
└── subject_15/        ← 222 MB (469 trials)
    ├── trial_0000_pred_sr.npy       (62, 1000)
    ├── trial_0000_target_sr.npy     (62, 1000)
    ├── trial_0000_meta.json         {metadata}
    └── ... (468 more trials)
```

**Files to use**:
- `subject_X/trial_NNNN_pred_sr.npy`: Model predictions
- `subject_X/trial_NNNN_target_sr.npy`: Ground truth
- `subject_X/trial_NNNN_meta.json`: Trial metadata

---

## 🎯 Quick Start by Use Case

### "I want to see performance metrics"
```
1. Read: EVALUATION_REPORT.md (Part 2: Metrics)
2. See: ten_samples_comparison.png (visual check)
3. Load: results_summary.npz (PCC, NMSE, SNR arrays)
```

### "I want to analyze results per subject"
```
1. Navigate: stad_raw_subject_output/subject_X/
2. Load: trial_NNNN_pred_sr.npy and trial_NNNN_target_sr.npy
3. Compute: Per-channel or per-trial metrics
```

### "I want to verify data leakage prevention"
```
1. Read: EVALUATION_REPORT.md (Part 1: Configuration)
2. Check: Test subjects [7, 10, 15] disjoint from training
3. Verify: KFold split consistency (random_state=2024)
```

### "I want to understand the data format"
```
1. Subject data: stad_raw_subject_output/README.md
2. Batch data: stad_raw_evaluation/README.md
3. Complete guide: EVALUATION_REPORT.md (Part 9: Quick Reference)
```

### "I want to investigate low metrics (~0.05 PCC)"
```
1. Read: EVALUATION_REPORT.md (Part 7: Issues & Limitations)
2. Follow: Recommended next steps for model investigation
3. Compare: Against baseline models (linear interpolation)
```

---

## 📈 Key Metrics at a Glance

```
Performance on Test Set (Subjects 7, 10, 15):

PCC (Pearson Correlation Coefficient)
  Mean: 0.0502  ← VERY LOW - indicates poor reconstruction
  Range: [0.0090, 0.1136]

NMSE (Normalized Mean Squared Error)
  Mean: 0.9956  ← NEAR 1.0 - baseline is 1.0
  Range: [0.9845, 1.0057]

SNR (Signal-to-Noise Ratio)
  Mean: 0.02 dB  ← NEGLIGIBLE - noise dominates
  Range: [-0.02, 0.07] dB

⚠️ Interpretation: Model shows very poor reconstruction.
   Possible causes: underfitting, non-convergence, or training data leakage.
```

---

## 📁 File Organization

```
.
├── EVALUATION_REPORT.md                 ← Start here (comprehensive guide)
├── stad_raw_evaluation/
│   ├── README.md                        (detailed metric definitions)
│   ├── results_summary.npz              (665 MB - batch metrics)
│   └── ten_samples_comparison.png       (1.8 MB - visualization)
│
└── stad_raw_subject_output/
    ├── README.md                        (data format guide)
    ├── subject_7/                       (222 MB - 469 trials)
    ├── subject_10/                      (222 MB - 469 trials)
    └── subject_15/                      (222 MB - 469 trials)
        ├── trial_0000_pred_sr.npy
        ├── trial_0000_target_sr.npy
        ├── trial_0000_meta.json
        └── ... (468 more trials)

Total: 1.33 GB
```

---

## 🔧 Common Operations

### Load Batch Metrics
```python
import numpy as np

results = np.load('stad_raw_evaluation/results_summary.npz', allow_pickle=True)
pcc = np.mean(results['pcc_scores'])
nmse = np.mean(results['nmse_scores'])
snr = np.mean(results['snr_scores'])

print(f"PCC: {pcc:.4f}, NMSE: {nmse:.4f}, SNR: {snr:.2f} dB")
```

### Load Subject Trial
```python
import numpy as np

subject_id = '7'
trial_id = 0

pred = np.load(
    f'stad_raw_subject_output/subject_{subject_id}/trial_{trial_id:04d}_pred_sr.npy'
)
target = np.load(
    f'stad_raw_subject_output/subject_{subject_id}/trial_{trial_id:04d}_target_sr.npy'
)

print(f"Prediction shape: {pred.shape}, dtype: {pred.dtype}")
print(f"Target shape: {target.shape}, dtype: {target.dtype}")
```

### Compute Subject-Level Metrics
```python
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

def analyze_subject(subject_id, n_trials=469):
    preds = []
    targets = []
    
    for i in range(n_trials):
        pred = np.load(
            f'stad_raw_subject_output/subject_{subject_id}/trial_{i:04d}_pred_sr.npy'
        )
        target = np.load(
            f'stad_raw_subject_output/subject_{subject_id}/trial_{i:04d}_target_sr.npy'
        )
        preds.append(pred)
        targets.append(target)
    
    # Flatten and compute correlation
    preds = np.stack(preds)    # (469, 62, 1000)
    targets = np.stack(targets)
    
    corr, _ = pearsonr(preds.ravel(), targets.ravel())
    return corr

for subj in ['7', '10', '15']:
    corr = analyze_subject(subj)
    print(f"Subject {subj}: PCC = {corr:.4f}")
```

---

## ✅ Validation Checklist

- [x] **Data collection**: 1,407 samples from test subjects [7, 10, 15]
- [x] **Data leakage prevention**: KFold filtering applied, no training subjects
- [x] **Batch processing**: 44 batches × 32 samples processed
- [x] **Metrics computed**: PCC, NMSE, SNR per batch
- [x] **Visualizations**: 10-sample comparison PNG generated
- [x] **Subject-wise storage**: 3 subjects × 469 trials saved
- [x] **Documentation**: 3 comprehensive README files
- [x] **Total outputs**: 1.33 GB organized in 2 main directories

---

## ⚠️ Important Notes

### Model Performance is Low
**Current PCC: 0.0502** (on scale [0, 1])

This indicates:
- Model produces predictions barely correlated with ground truth
- NMSE ≈ 1.0 means model is essentially predicting the mean
- SNR negligible suggests noise drowns out signal

**Action required**: Investigate training quality and verify test subject isolation

### Memory Requirements
- Loading all 1,407 samples: ~6.5 GB RAM
- Batch processing: ~1.1 GB (batch_size=32)
- Solution: Process per-subject or use memory-mapped arrays

### GPU Performance
- Processing speed: 3.09 batches/sec (NVIDIA RTX A5000)
- Batch size: 32
- Total time: ~14 seconds

---

## 📚 Documentation Map

| Document | Purpose | Size | Read Time |
|----------|---------|------|-----------|
| **EVALUATION_REPORT.md** | Complete overview & analysis | 45 KB | 15 min |
| **stad_raw_evaluation/README.md** | Batch data format & metrics | 35 KB | 10 min |
| **stad_raw_subject_output/README.md** | Per-trial data format | 40 KB | 10 min |

---

## 🔍 Verification Commands

```bash
# Check total samples
find stad_raw_subject_output -name "*.npy" | wc -l
# Expected: 2814 (1407 trials × 2 files per trial)

# Check subjects
ls -d stad_raw_subject_output/subject_*/
# Expected: subject_7, subject_10, subject_15

# Check metrics file
ls -lh stad_raw_evaluation/results_summary.npz
# Expected: 665 MB

# Quick Python check
python << 'EOF'
import numpy as np
results = np.load('stad_raw_evaluation/results_summary.npz')
print(f"Batches: {len(results['pcc_scores'])}, Samples: {len(results['pcc_scores']) * 32}")
EOF
```

---

## 🚀 Next Steps

1. **Immediate** (review metrics):
   - Read EVALUATION_REPORT.md
   - Check ten_samples_comparison.png
   - Load results_summary.npz

2. **Investigation** (understand low metrics):
   - Verify test subject isolation
   - Check training convergence
   - Compare against baseline

3. **Analysis** (deep dive):
   - Per-channel performance
   - Subject-specific patterns
   - Temporal consistency

4. **Optional** (advanced):
   - Frequency-domain analysis
   - Spatial smoothness metrics
   - Implement MNE topomaps

---

## 📞 Support Resources

**For error "file not found"**:
- Verify from `/home/ab_students/EEG-MTP/New_SEED4/` directory
- Paths are relative to New_SEED4

**For "memory error"**:
- Process one subject at a time
- Use `mmap_mode='r'` for memory-mapped loading

**For "metric interpretation"**:
- Read: EVALUATION_REPORT.md Part 2 (Metrics)
- See: stad_raw_evaluation/README.md (definitions)

**For "data format questions"**:
- Read: stad_raw_subject_output/README.md
- Check: Example loading code in Common Operations

---

**Last Updated**: After complete evaluation run  
**Status**: ✅ READY FOR ANALYSIS  
**Next Phase**: Model investigation or advanced analysis
