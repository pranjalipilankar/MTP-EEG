# Quick Start Guide - MFE Loss Training

## 🚀 5-Minute Setup

### 1. Verify Files
All required files have been copied to `/home/ab_students/EEG-MTP/new_SEED4_mfe/`:
- ✅ `seed_stad_train_mfe.py` - Main training script
- ✅ `mfe_profile_loss.py` - MFE loss implementation
- ✅ `stad_model_CORRECT.py` - STAD model
- ✅ All supporting modules (MAE, MTD, config, etc.)
- ✅ `README.md` - Full documentation

### 2. Minimal Training Command
```bash
cd /home/ab_students/EEG-MTP/new_SEED4_mfe
python seed_stad_train_mfe.py \
  --epochs 10 \
  --batch_size 16 \
  --test_fold 0
```

### 3. Monitor Training
Watch for these metrics in each epoch:
```
Epoch 1/10 | Train: 0.345678 | Val: 0.356789 | PCC: 0.7234, NMSE: 0.0456, SNR: 12.34dB
```

- **PCC** (Pearson Correlation): Target >0.80
- **NMSE** (Normalized MSE): Target <0.05
- **SNR** (Signal-to-Noise): Target >15dB

### 4. Default Configuration
```yaml
Loss Formula: diffusion + 0.1*MSE + 0.1*MFE

MFE Parameters:
  Pattern length (m): 2
  Fuzzy exponent (n): 2.0
  Max time scale (τ_max): 20 (covers 4-80ms @ 250Hz)
  Tolerance (r): 0.15

Data:
  Batch size: 32
  Learning rate: 1e-4
  Epochs: 100
```

## 📊 Key Differences from HFD Version

| Feature | HFD | MFE |
|---------|-----|-----|
| Loss metric | Fractal dimension | Fuzzy entropy |
| Script | `seed_stad_train_hfd.py` | `seed_stad_train_mfe.py` |
| Folder | `new_SEED4_hfd/` | `new_SEED4_mfe/` |
| Weight range | 0.1-0.5 | 0.1-0.3 |
| Default weight | 0.3 | 0.1 |

## �� Common Scenarios

### Scenario 1: Test Installation (5 minutes)
```bash
python seed_stad_train_mfe.py \
  --epochs 1 \
  --batch_size 8 \
  --test_only
```
Should complete without errors and show latent shape.

### Scenario 2: Quick Training (1 hour)
```bash
python seed_stad_train_mfe.py \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4
```
Expect PCC ~0.75-0.80 at epoch 10.

### Scenario 3: Production Training (overnight)
```bash
python seed_stad_train_mfe.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --freeze_mae \
  --unfreeze_mae_epoch 50 \
  --mfe_loss_weight 0.15
```
Expect PCC ~0.85-0.90 at epoch 100.

### Scenario 4: Adjust for Output Quality

**If too smooth:**
```bash
python seed_stad_train_mfe.py \
  --mfe_loss_weight 0.3 \
  --sr_loss_weight 0.05
```

**If too noisy:**
```bash
python seed_stad_train_mfe.py \
  --mfe_loss_weight 0.05 \
  --sr_loss_weight 0.2
```

## 📈 Expected Training Progress

### Epoch 1-5
- Loss: High (0.3-0.5)
- PCC: Low (0.5-0.65)
- Training: Model learning basic reconstruction

### Epoch 10-30
- Loss: Decreasing (0.2-0.3)
- PCC: Improving (0.70-0.80)
- Training: Starting to preserve complexity

### Epoch 50-100
- Loss: Stable (0.15-0.25)
- PCC: High (0.80-0.90)
- Training: Fine-tuning complexity matching

## 🔧 Adjusting MFE Parameters

### To capture faster dynamics:
```bash
--mfe_tau_max 30  # Extends to ~120ms
--mfe_n 2.5       # Sharper fuzzy membership
```

### To capture slower dynamics:
```bash
--mfe_tau_max 10  # Reduces to ~40ms
--mfe_r_fixed 0.25 # Looser tolerance
```

### To improve gradient flow:
```bash
--mfe_m 2         # Keep at 2
--mfe_n 2.0       # Standard Gaussian
```

## 📋 Pre-Training Checklist

- [ ] Data path exists: `/DATA/EEG-MTP/seed4/eeg_processed_data`
- [ ] MAE checkpoint available: Check with `--mae_kfold_dir`
- [ ] GPU memory: 20GB+ recommended (check with `nvidia-smi`)
- [ ] Disk space: 5-10GB for checkpoints/history
- [ ] Python packages: PyTorch, NumPy (already installed)

## 🚦 Execution Examples

### Example 1: Basic run (no arguments)
```bash
python seed_stad_train_mfe.py
```
Uses all defaults. MAE checkpoint auto-selected from k-fold results.

### Example 2: With specific MAE
```bash
python seed_stad_train_mfe.py \
  --mae_checkpoint /path/to/best_model.pth \
  --epochs 50
```

### Example 3: Resume from checkpoint
```bash
python seed_stad_train_mfe.py \
  --resume_stad_checkpoint /home/ab_students/EEG-MTP/new_SEED4_mfe/latest_stad_model.pth \
  --epochs 100
```

### Example 4: Full production training
```bash
nohup python seed_stad_train_mfe.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --freeze_mae \
  --unfreeze_mae_epoch 50 \
  --mfe_loss_weight 0.15 \
  --sr_loss_weight 0.10 \
  > training.log 2>&1 &
```

## 📂 Output Location

All results saved to: `/home/ab_students/EEG-MTP/new_SEED4_mfe/`

```
new_SEED4_mfe/
├── best_stad_model.pth         # Best checkpoint
├── latest_stad_model.pth       # Latest checkpoint
├── training_history.npy        # Full metrics history
└── (this folder also contains source code)
```

Load and analyze results:
```python
import numpy as np
import torch

# Load history
history = np.load('training_history.npy', allow_pickle=True).tolist()
print(f"Trained for {len(history)} epochs")
print(f"Best epoch: {history[-1]['epoch']}")
print(f"Final PCC: {history[-1]['val_pcc']:.4f}")

# Load model
checkpoint = torch.load('best_stad_model.pth', map_location='cpu')
print(f"Epoch {checkpoint['epoch']}: Loss={checkpoint['best_val_loss']:.6f}")
```

## 🆘 Troubleshooting Quick Links

See **README.md** for detailed troubleshooting:
- MFE Loss is NaN/Inf
- Training loss doesn't decrease
- Output is too smooth or too noisy
- GPU memory issues
- Data loading errors

## 📞 Next Steps

1. **Run test**: `python seed_stad_train_mfe.py --test_only`
2. **Start training**: `python seed_stad_train_mfe.py --epochs 10`
3. **Monitor progress**: Watch `training_history.npy`
4. **Tune parameters**: Adjust loss weights based on results
5. **Full training**: Run 100+ epochs for production model

---

**Folder**: `/home/ab_students/EEG-MTP/new_SEED4_mfe`
**Training script**: `seed_stad_train_mfe.py`
**Documentation**: `README.md`
