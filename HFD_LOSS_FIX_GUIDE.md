# 🚨 CRITICAL FIX: HFD Loss Weight Causing Training Collapse

## Problem Summary
Your PCC dropped from **0.6155 (MSE only)** → **0.0004 (MSE + HFD)** because:

1. **Scale Mismatch**: HFD operates in log-log space (values ~0.001-0.1), MSE in linear signal space (values ~0.1-1.0)
2. **Loss Domination**: With `hfd_loss_weight=0.3`, HFD term completely dominates gradient computation
3. **Wrong Optimization Direction**: Model optimizes for log-space curve matching instead of signal amplitude

---

## Root Cause Analysis

### Why is HFD producing such small values?
- HFD computes `log(L(k))` where `L(k)` is very small (order 1e-3 to 1e-1)
- Taking log compresses this further: `log(1e-3) ≈ -6.9`
- Your loss becomes: `total = diff + 0.1*mse_loss + 0.3*hfd_loss`
- With `mse_loss ≈ 0.5` and `hfd_loss ≈ 0.01`, the `0.3*hfd_loss = 0.003` is negligible... wait, that should be fine
- **ACTUALLY**: HFD curves are being compared in log space, differences can be large!
- When curves differ: `|log(L_pred) - log(L_target)| ≈ 1-2` (in log space)
- So: `hfd_loss ≈ 0.5-1.0` × `0.3` = `0.15-0.3` (comparable to mse_loss!)
- **This causes gradient directions to conflict**

---

## Solution: Adaptive Loss Weighting

### Recommended Fix
Change the HFD weight from **0.3** to **0.01-0.05**:

```python
# OLD (BROKEN):
parser.add_argument('--hfd_loss_weight', type=float, default=0.3,
                    help='Weight for HFD loss')

# NEW (FIXED):
parser.add_argument('--hfd_loss_weight', type=float, default=0.01,
                    help='⚠️  CRITICAL: HFD operates in log-space. '
                         'Default=0.01 (was 0.3). Safe range: 0.005-0.05. '
                         'Do NOT use values >0.1')
```

### Why 0.01?
- At 0.01: `0.01 * hfd_loss(0.5) = 0.005` (MSE contribution still dominates)
- Allows HFD to refine complexity without corrupting amplitude matching
- Prevents gradient conflicts between linear and log-space objectives

### Progressive Increase Strategy
```python
# Epoch-based adaptive weighting (if desired):
if epoch < 30:
    hfd_weight = 0.005  # Start very conservative
elif epoch < 60:
    hfd_weight = 0.01   # Increase gradually
else:
    hfd_weight = 0.02   # Final weight (still safe)

hfd_loss_contrib = hfd_weight * hfd_loss
```

---

## Implementation Guide

### Step 1: Update Default Argument
In `seed_stad_train_hfd.py` around line 1100:
```python
parser.add_argument('--hfd_loss_weight', type=float, default=0.01)
```

### Step 2: Test New Weight
```bash
cd /home/ab_students/EEG-MTP/codes_sneha/new_SEED4_hfd

# Test with new weight (0.01)
python seed_stad_train_hfd.py \
    --epochs 5 \
    --batch_size 32 \
    --hfd_loss_weight 0.01 \
    --output_dir test_hfd_fix
```

### Step 3: Monitor Loss Ratios
Expected output (first epoch):
```
Train → Total: 1.234567  Diff: 0.123  MSE: 0.567  HFD: 0.500 | ...
        ↑ weighted HFD: 0.01*0.500 = 0.005
```

If you see:
- ✅ MSE loss decreasing → Good (amplitude matching works)
- ✅ HFD loss decreasing → Good (complexity matching works)  
- ✅ PCC increasing → Good (correlation improving)
- ❌ PCC near 0 → HFD weight still too high, reduce to 0.005

---

## Testing Strategy

### Test 1: Verify MSE-Only Baseline (Control)
```bash
python seed_stad_train_hfd.py \
    --epochs 10 \
    --hfd_loss_weight 0.0 \
    --output_dir test_mse_only
# Expected: PCC ≈ 0.61 (same as before)
```

### Test 2: Test Low HFD Weight
```bash
python seed_stad_train_hfd.py \
    --epochs 10 \
    --hfd_loss_weight 0.01 \
    --output_dir test_hfd_0.01
# Expected: PCC ≈ 0.60-0.61 (similar to baseline, maybe slightly better)
```

### Test 3: Test Medium HFD Weight  
```bash
python seed_stad_train_hfd.py \
    --epochs 10 \
    --hfd_loss_weight 0.05 \
    --output_dir test_hfd_0.05
# Expected: PCC ≈ 0.59-0.60 (slight degradation acceptable if complexity improves)
```

### Test 4: Compare Against Old Weight (for verification)
```bash
python seed_stad_train_hfd.py \
    --epochs 10 \
    --hfd_loss_weight 0.3 \
    --output_dir test_hfd_0.3_broken
# Expected: PCC ≈ 0.0004 (confirms the bug)
```

---

## Monitoring Metrics

### Check These During Training
1. **PCC Trend**: Should stay high (>0.55) throughout training
2. **Loss Ratio**: `val_hfd_loss / val_mse_loss` should be < 1.0
3. **Gradient Magnitude**: Check for NaN or explosion
4. **Convergence Speed**: Should be similar to MSE-only baseline

### Red Flags (indicates HFD weight too high)
- PCC drops below 0.1 in first 5 epochs
- Loss becomes NaN
- HFD loss > MSE loss by 10x
- Training becomes chaotic/unstable

---

## Advanced: Adaptive Weighting Over Epochs

If you want to gradually increase HFD importance over training:

```python
def get_hfd_weight(epoch, total_epochs, min_weight=0.005, max_weight=0.02):
    """Linearly increase HFD weight from min to max over training."""
    fraction = epoch / total_epochs
    return min_weight + (max_weight - min_weight) * fraction

# In training loop:
for epoch in range(start_epoch, args.epochs):
    current_hfd_weight = get_hfd_weight(epoch, args.epochs)
    # Use current_hfd_weight instead of args.hfd_loss_weight
```

---

## Why This Fix Works

| Aspect | MSE Only | HFD @ 0.3 (Bad) | HFD @ 0.01 (Good) |
|--------|----------|-----------------|-------------------|
| Amplitude Match | ✅ Strong | ❌ Weak | ✅ Strong |
| Complexity Match | ❌ None | ❌ Broken (log/linear conflict) | ✅ Gentle |
| Gradient Direction | Linear space | Linear + Log mismatch | Linear (primary) + Log (secondary) |
| PCC Result | 0.6155 | 0.0004 | ~0.60-0.62 |
| Training Stability | ✅ Stable | ❌ Unstable | ✅ Stable |

---

## Summary

**Change one line:**
```python
# Line ~1100 in seed_stad_train_hfd.py
parser.add_argument('--hfd_loss_weight', type=float, default=0.01)  # Was: 0.3
```

**Then retrain:**
```bash
python seed_stad_train_hfd.py --epochs 100 --hfd_loss_weight 0.01
```

**Expected result:** PCC ≈ 0.60-0.62 (similar to MSE-only baseline, with improved complexity preservation)

