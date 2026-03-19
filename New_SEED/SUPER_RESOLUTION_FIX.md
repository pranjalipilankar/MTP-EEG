# Super-Resolution Architecture Fix

## Problems Identified

### ❌ Problem 1: Not True Super-Resolution

**Before:**
```python
# Naive upsampling that ignores LR conditioning
hr_eeg = mae.decode(latent)  # → 31 channels
sr_eeg = nn.Linear(31, 62)(hr_eeg)  # ← Just learned interpolation!
```

**Issues:**
- The 16-channel LR input **wasn't used** in the final 31→62 upsampling
- Simple linear projection doesn't add spatial detail
- No guidance from low-resolution observation

### ❌ Problem 2: Dimension Mismatch

**STC Output vs MTD Input:**
```
STC: cond_tokens (B, 16, 1024)  # 16 LR channels
MTD: Query (B, 500, 1024)       # 500 temporal patches
```

Cross-attention between 16 channels and 500 patches is **semantically wrong**:
- Should be: 16 LR channels → guide → 62 SR channels (spatial)
- Currently: 16 channels → condition → 500 patches (temporal)

### ❌ Problem 3: Dynamic Layer Creation

```python
# In forward() - BAD!
if not hasattr(self, 'cond_proj'):
    self.cond_proj = nn.Linear(...)  # Not in optimizer!
```

Layers created during forward:
- Won't be in `model.parameters()`
- Won't be optimized
- Won't be saved in checkpoints

## Solutions Implemented

### ✅ Solution 1: Conditioning-Aware Upsampling

**After:**
```python
# Proper fusion of HR reconstruction + LR spatial guidance
hr_feat = project_31(hr_eeg)   # (B, T, 62)
lr_feat = project_16(lr_eeg)   # (B, T, 62)  ← Uses LR!
fused = hr_feat + lr_feat      # Residual fusion
sr_eeg = fusion_net(fused)     # (B, T, 62)
```

**Benefits:**
- LR provides **spatial guidance** for where to add detail
- HR provides **reconstructed structure** from latent
- Learned fusion balances both sources

**Architecture:**
```python
self.sr_upsample = nn.ModuleDict({
    'hr_proj': nn.Linear(31, 62),    # HR contribution
    'lr_proj': nn.Linear(16, 62),    # LR spatial guide
    'fusion': nn.Sequential(         # Intelligent fusion
        nn.Linear(62, 124),
        nn.GELU(),
        nn.Linear(124, 62)
    )
})
```

### ✅ Solution 2: Proper Conditioning Flow

**New data flow:**
```
1. Training:
   16ch LR → [STC] → conditioning
   62ch SR → 31ch → [MAE] → latent → [+noise] → noisy latent
   noisy latent + conditioning → [MTD] → predicted noise
   Loss = MSE(predicted, actual_noise)

2. Inference:
   16ch LR → [STC] → conditioning
   noise → [MTD + conditioning] → clean latent
   clean latent + 16ch LR → [decode_with_LR] → 62ch SR
                    ↑
              Uses LR for spatial guidance!
```

### ✅ Solution 3: Fixed Initialization

```python
# In MTD.__init__():
self.cond_proj = None  # Explicitly defined

# In MTD.forward():
if cond_proj is None and dims mismatch:
    self.cond_proj = nn.Linear(...)
    self.add_module('cond_proj_dynamic', self.cond_proj)  # Register!
    print("WARNING: ...")  # Alert user to fix config
```

## True Super-Resolution Now

### What Makes It Super-Resolution?

**Definition:** Reconstructing high-resolution signal from low-resolution observation.

**Before (NOT SR):**
- Target: 62ch
- Process: 62ch → 31ch → latent → 31ch → **linear(62ch)**
- LR input: **Ignored in upsampling**

**After (TRUE SR):**
- Input: **16ch LR** (what we observe)
- Target: 62ch SR (what we want)
- Process: 
  - Encode: 62ch → 31ch → latent (training target)
  - Denoise: latent + **LR conditioning** → clean latent
  - Decode: clean latent + **LR spatial guide** → 62ch SR
  - The **16ch LR actively guides** where to add spatial detail

### How LR Guides SR

1. **During Diffusion (MTD):**
   ```python
   # STC extracts spatio-temporal features from 16ch LR
   cond_tokens, cond_pooled = STC(16ch_LR)
   
   # MTD uses these to guide denoising
   clean_latent = denoise(noisy_latent, conditioning=LR_features)
   ```

2. **During Upsampling:**
   ```python
   # 16ch LR provides spatial hints for 31→62 upsampling
   hr_feat = project(31ch)  # Reconstructed structure
   lr_feat = project(16ch)  # Spatial guidance
   sr_62ch = fuse(hr_feat, lr_feat)  # Intelligent combination
   ```

## Expected Improvements

### Metrics Should Improve Because:

1. **Better Spatial Coherence:**
   - LR channels guide where to add detail
   - Not just blind interpolation

2. **Consistent Conditioning:**
   - Same 16ch LR used in both diffusion and upsampling
   - End-to-end learning of LR→SR mapping

3. **Learned Fusion:**
   - Network learns how much to trust HR vs LR
   - Adaptive spatial detail enhancement

### What to Monitor:

After retraining with this fix:
- **PCC should increase** (better correlation with target)
- **SNR should improve** (less reconstruction noise)
- **Spatial coherence** (adjacent channels more correlated)

## Code Changes Summary

### train_stad_seed.py

1. **Added `sr_upsample` module** (line ~330):
   - Replaces naive `channel_upsample = nn.Linear(31, 62)`
   - Uses both HR and LR for intelligent fusion

2. **Updated `decode_latent_to_sr()`** (line ~340):
   - Now takes `lr_eeg` as parameter
   - Fuses HR reconstruction with LR guidance

3. **Updated `reconstruct_eeg_fixed()`** (line ~120):
   - Passes `x_lr` to decoder

### mtd_dreamdiff.py

1. **Added `cond_proj` initialization** (line ~92):
   - Defined in `__init__` instead of forward
   - Proper registration with `add_module`

2. **Added warning message** (line ~175):
   - Alerts if STC/MTD dimensions mismatch
   - Should be fixed in config, not dynamically

## Next Steps

1. **Kill current training** (old architecture)
2. **Restart with fixed model**
3. **Monitor metrics at epoch 30**:
   - Should see PCC > 0.1 (vs ~0.0001 before)
   - SNR > 0 dB (vs -1.53 dB before)

4. **Compare epoch 60 results**:
   - Expect significant improvement
   - True SR learning curve

## Theoretical Justification

### Why This Is Better:

**Super-resolution in images:**
```
Low-res image → CNN → High-res image
  (observed)           (predicted)
```

**Our EEG super-resolution:**
```
16ch LR → [Diffusion + MAE + Fusion] → 62ch SR
(observed)                              (predicted)
         ↓
      Guides both:
      1. Latent denoising (what to reconstruct)
      2. Spatial upsampling (where to add detail)
```

**Key principle:** The low-resolution observation should **actively guide** the high-resolution reconstruction, not just be a side input.

Now it does! 🎉
