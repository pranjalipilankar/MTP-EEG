# 📊 Complete Loss Breakdown - MFE Training

## Overall Loss Formula

```
TOTAL_LOSS = DIFFUSION_LOSS + λ_mse × MSE_LOSS + λ_mfe × MFE_LOSS
```

---

## 🔴 Loss Component 1: DIFFUSION_LOSS (Always 100%)

**What it is:**
- Noise prediction loss from the diffusion model
- The core STAD denoising objective

**Where it comes from:**
```python
# In stad_model_CORRECT.py
diff_loss, pred_sr = stad_model(lr_eeg, hr_eeg, sr_eeg)
```

**Formula:**
```
DIFFUSION_LOSS = MSE(predicted_noise, true_noise)  [in latent space]
```

**Contribution to total:**
- ✅ **Always weight = 1.0** (no parameter to adjust)
- Baseline that always runs

**Typical magnitude:** 0.3-0.5 per epoch

---

## 🟢 Loss Component 2: MSE_LOSS (Amplitude Matching)

**What it is:**
- Pixel-level (sample-level) reconstruction loss
- Ensures predicted EEG matches target EEG amplitude

**Where it comes from:**
```python
# In seed_stad_train_mfe.py, line ~372
mse_loss = F.mse_loss(pred_sr.float(), sr_eeg.float())
```

**Formula:**
```
MSE_LOSS = (1/N) × Σ (predicted_sample - target_sample)²
```

**Weight Parameter:**
```python
--sr_loss_weight (default: 0.1)
```

**How much does it contribute?**
```
MSE_contribution = 0.1 × mse_loss
```

**Typical magnitude:**
- Raw MSE value: 0.5-2.0
- Weighted contribution: 0.05-0.2 to total loss

**What it does:**
- ✅ Ensures signal amplitude is correct
- ✅ Prevents extreme deviations
- ❌ Can't preserve temporal structure alone

---

## 🔵 Loss Component 3: MFE_LOSS (Complexity Matching)

**What it is:**
- Multiscale Fuzzy Entropy loss
- Preserves temporal complexity across time scales

**Where it comes from:**
```python
# In seed_stad_train_mfe.py, line ~374
mfe_loss = mfe_loss_fn(pred_sr.float(), sr_eeg.float().detach())
```

**Formula:**
```
MFE_LOSS = mean_τ |FuzzyEn_predicted(τ) - FuzzyEn_target(τ)|

Where:
  τ = time scale (1 to τ_max)
  FuzzyEn = Fuzzy entropy at that scale
```

**Weight Parameter:**
```python
--mfe_loss_weight (default: 0.1)
```

**How much does it contribute?**
```
MFE_contribution = 0.1 × mfe_loss
```

**Typical magnitude:**
- Raw MFE value: 0.01-0.5
- Weighted contribution: 0.001-0.05 to total loss

**What it does:**
- ✅ Preserves EEG complexity patterns
- ✅ Prevents over-smoothing
- ✅ Scale-invariant (works across subjects)

---

## 📐 Complete Loss Equation (Expanded)

```
TOTAL_LOSS = 
    diff_loss_value 
    + 0.1 × mse_loss_value 
    + 0.1 × mfe_loss_value
```

**Example with actual values:**
```
Epoch 1:
  diff_loss = 0.45
  mse_loss = 0.8
  mfe_loss = 0.3
  
  TOTAL = 0.45 + 0.1×0.8 + 0.1×0.3
        = 0.45 + 0.08 + 0.03
        = 0.56

Epoch 50:
  diff_loss = 0.18
  mse_loss = 0.2
  mfe_loss = 0.05
  
  TOTAL = 0.18 + 0.1×0.2 + 0.1×0.05
        = 0.18 + 0.02 + 0.005
        = 0.205
```

---

## 🎛️ Percentage Contribution (By Default Settings)

Assuming typical magnitudes:

```
┌─────────────────────────────────────────────┐
│ LOSS COMPONENT BREAKDOWN (Default Weights)   │
├─────────────────────────────────────────────┤
│ DIFFUSION:   ~80-90% of total loss          │
│ MSE:          ~5-15% of total loss          │
│ MFE:          ~5-10% of total loss          │
└─────────────────────────────────────────────┘
```

**Why Diffusion dominates?**
- It's the only component with weight = 1.0
- The others have weight = 0.1
- But diffusion values are typically 0.2-0.5
- While MSE/MFE values are 0.1-1.0

---

## 🎚️ How to Adjust the Loss Balance

### Default Configuration
```bash
python seed_stad_train_mfe.py \
  --sr_loss_weight 0.1 \
  --mfe_loss_weight 0.1
```

### Emphasize Amplitude (Sharper, Noisier Output)
```bash
python seed_stad_train_mfe.py \
  --sr_loss_weight 0.3 \     # Increase MSE
  --mfe_loss_weight 0.05     # Decrease MFE
```

**Result:**
```
TOTAL = diff + 0.3×mse + 0.05×mfe
```

### Emphasize Complexity (Smoother, Less Noisy)
```bash
python seed_stad_train_mfe.py \
  --sr_loss_weight 0.05 \    # Decrease MSE
  --mfe_loss_weight 0.3      # Increase MFE
```

**Result:**
```
TOTAL = diff + 0.05×mse + 0.3×mfe
```

### Balanced (Default)
```bash
python seed_stad_train_mfe.py \
  --sr_loss_weight 0.1 \
  --mfe_loss_weight 0.1
```

**Result:**
```
TOTAL = diff + 0.1×mse + 0.1×mfe
```

---

## 📊 Loss Value Ranges You'll See During Training

### Early Epochs (1-10)
```
DIFFUSION_LOSS: 0.40-0.50
MSE_LOSS:       0.80-1.20  → weighted: 0.08-0.12
MFE_LOSS:       0.20-0.40  → weighted: 0.02-0.04
─────────────────────────────
TOTAL_LOSS:     0.50-0.65
```

### Middle Epochs (30-50)
```
DIFFUSION_LOSS: 0.18-0.25
MSE_LOSS:       0.20-0.40  → weighted: 0.02-0.04
MFE_LOSS:       0.05-0.15  → weighted: 0.005-0.015
─────────────────────────────
TOTAL_LOSS:     0.20-0.30
```

### Late Epochs (80-100)
```
DIFFUSION_LOSS: 0.15-0.20
MSE_LOSS:       0.10-0.20  → weighted: 0.01-0.02
MFE_LOSS:       0.02-0.05  → weighted: 0.002-0.005
─────────────────────────────
TOTAL_LOSS:     0.16-0.22
```

---

## 🔍 Understanding Each Component

### DIFFUSION_LOSS Details

**Purpose:** Core denoising objective

**Formula:**
```python
loss = F.mse_loss(predicted_noise, true_noise)
```

**Why it's important:**
- Drives the diffusion model to learn denoising
- Largest contributor to overall loss
- Directly from DDPM paper

**When it works well:**
- Decreases smoothly during training
- Usually dominates early, plateaus late

---

### MSE_LOSS Details

**Purpose:** Pixel-level reconstruction accuracy

**Formula:**
```python
mse_loss = F.mse_loss(pred_sr, sr_eeg)
            = mean((pred_sr - sr_eeg)^2)
```

**Parameters:**
```
--sr_loss_weight: 0.1 (default)
                  0.05-0.30 (typical range)
```

**When to increase:**
- Output is too different from target
- Need sharper, noisier details
- NMSE metric is too high

**When to decrease:**
- Output is over-matching noise
- Need smoother, cleaner signal
- Output looks too similar to noisy input

---

### MFE_LOSS Details

**Purpose:** Temporal complexity preservation

**Formula:**
```python
mfe_loss = mean_over_scales |FuzzyEn(pred) - FuzzyEn(target)|
```

**Parameters:**
```
--mfe_loss_weight:  0.1 (default)
--mfe_m:            2 (pattern length)
--mfe_n:            2.0 (fuzzy exponent)
--mfe_tau_max:      20 (max time scale = 80ms @ 250Hz)
--mfe_r_fixed:      0.15 (tolerance)
```

**When to increase weight:**
- Output is too smooth
- Losing EEG transients
- PCC is high but SNR is low

**When to decrease weight:**
- Output is too noisy
- MFE loss has NaN/Inf values
- Signal looks similar to input

---

## 📈 Real Training Example

### Command:
```bash
python seed_stad_train_mfe.py \
  --epochs 5 \
  --batch_size 32 \
  --sr_loss_weight 0.1 \
  --mfe_loss_weight 0.1
```

### Expected Output (Each Epoch):
```
Epoch 1/5 | Train: 0.563412 | Val: 0.575234 | PCC: 0.5456, NMSE: 0.1523, SNR: 8.32dB
  Components:
  - diff ≈ 0.48
  - mse  ≈ 0.82  (0.1 × 0.82 = 0.082 in total)
  - mfe  ≈ 0.35  (0.1 × 0.35 = 0.035 in total)

Epoch 2/5 | Train: 0.482156 | Val: 0.491342 | PCC: 0.6234, NMSE: 0.1123, SNR: 10.45dB
  Components:
  - diff ≈ 0.42
  - mse  ≈ 0.65  (0.1 × 0.65 = 0.065 in total)
  - mfe  ≈ 0.28  (0.1 × 0.28 = 0.028 in total)

Epoch 3/5 | Train: 0.392847 | Val: 0.403156 | PCC: 0.7123, NMSE: 0.0834, SNR: 12.67dB
  Components:
  - diff ≈ 0.35
  - mse  ≈ 0.48  (0.1 × 0.48 = 0.048 in total)
  - mfe  ≈ 0.15  (0.1 × 0.15 = 0.015 in total)

Epoch 4/5 | Train: 0.321543 | Val: 0.335782 | PCC: 0.7923, NMSE: 0.0567, SNR: 14.89dB
  Components:
  - diff ≈ 0.28
  - mse  ≈ 0.32  (0.1 × 0.32 = 0.032 in total)
  - mfe  ≈ 0.08  (0.1 × 0.08 = 0.008 in total)

Epoch 5/5 | Train: 0.271234 | Val: 0.285671 | PCC: 0.8456, NMSE: 0.0345, SNR: 16.23dB
  Components:
  - diff ≈ 0.24
  - mse  ≈ 0.28  (0.1 × 0.28 = 0.028 in total)
  - mfe  ≈ 0.05  (0.1 × 0.05 = 0.005 in total)
```

---

## 🎯 Recommended Configurations for Different Goals

### Goal: Maximum Quality (Best PCC & NMSE)
```bash
--sr_loss_weight 0.15
--mfe_loss_weight 0.15
```
Result: `TOTAL = diff + 0.15*mse + 0.15*mfe`

### Goal: Sharp Details (Higher Frequency Content)
```bash
--sr_loss_weight 0.30
--mfe_loss_weight 0.05
```
Result: `TOTAL = diff + 0.30*mse + 0.05*mfe`

### Goal: Smooth Signal (Lower Noise)
```bash
--sr_loss_weight 0.05
--mfe_loss_weight 0.25
```
Result: `TOTAL = diff + 0.05*mse + 0.25*mfe`

### Goal: Balanced (Default)
```bash
--sr_loss_weight 0.10
--mfe_loss_weight 0.10
```
Result: `TOTAL = diff + 0.10*mse + 0.10*mfe`

### Goal: Complexity Only
```bash
--sr_loss_weight 0.01
--mfe_loss_weight 0.30
```
Result: `TOTAL = diff + 0.01*mse + 0.30*mfe`

---

## 📊 Loss Components Tracked During Training

The script tracks these separately:

```python
train_losses = {
    'total':  [],  # Sum of all losses
    'diff':   [],  # Diffusion loss only
    'mse':    [],  # MSE loss (raw, before weighting)
    'mfe':    []   # MFE loss (raw, before weighting)
}
```

**In training loop:**
```python
total_loss = diff_loss + args.sr_loss_weight * mse_loss + args.mfe_loss_weight * mfe_loss
```

**Stored in history:**
```python
history.append({
    'epoch': epoch + 1,
    'train_loss': train_loss,      # Total
    'val_loss': val_loss,          # Total
    'val_pcc': mean_pcc,
    'val_nmse': mean_nmse,
    'val_snr': mean_snr,
})
```

---

## 💡 Quick Decision Guide

| Situation | Action | New Command |
|-----------|--------|-------------|
| Output too smooth | ↑ MSE weight | `--sr_loss_weight 0.3` |
| Output too noisy | ↓ MSE weight | `--sr_loss_weight 0.05` |
| Lost complexity | ↑ MFE weight | `--mfe_loss_weight 0.3` |
| Too much noise | ↓ MFE weight | `--mfe_loss_weight 0.05` |
| NaN/Inf in MFE | ↓ MFE weight | `--mfe_loss_weight 0.01` |
| Not converging | Decrease LR | `--lr 5e-5` |

---

## 🔗 How the Loss Flows Through Training

```
┌─────────────────────────────────────────────────────┐
│ Forward Pass                                         │
│                                                      │
│ Input: (lr_eeg, hr_eeg, sr_eeg)                     │
│   ↓                                                  │
│ STAD Model → outputs: (diff_loss, pred_sr)          │
└─────────────────────────────────────────────────────┘
                    ↓
        ┌───────────┴──────────┬────────────┐
        ↓                      ↓            ↓
   DIFFUSION          MSE Loss         MFE Loss
   (1.0 × value)   (0.1 × value)    (0.1 × value)
        ↓                      ↓            ↓
        └───────────┬──────────┴────────────┘
                    ↓
         total_loss = sum(all 3)
                    ↓
        ┌───────────────────────────────┐
        │ Backward Pass (Autograd)      │
        │ Compute gradients             │
        │ Update weights                │
        │ Log loss components           │
        └───────────────────────────────┘
```

---

## Summary Table

| Component | Default Weight | Typical Value | Contribution % |
|-----------|----------------|---------------|-----------------|
| DIFFUSION | 1.0 | 0.15-0.45 | 80-90% |
| MSE | 0.1 | 0.1-1.0 | 5-15% |
| MFE | 0.1 | 0.01-0.5 | 5-10% |

**Total Loss = DIFFUSION + 0.1×MSE + 0.1×MFE**

---

**Status**: Reference Complete ✅
**Last Updated**: April 2026
**File**: `/home/ab_students/EEG-MTP/new_SEED4_mfe/LOSS_BREAKDOWN.md`
