# Fixes Applied: Hyperparameter Tuning Issues

## Issue 1: Type Error (FIXED ✅)

### Problem
The hyperparameter tuning was failing with:
```
batch_size should be a positive integer value, but got batch_size=32
```

Even though `batch_size=32` looks correct, the actual type was `np.int64(32)` instead of Python's native `int`.

### Root Cause
When using `np.random.choice()` on numpy arrays, it returns numpy scalar types (`np.int64`, `np.float64`, etc.) instead of Python native types (`int`, `float`). PyTorch's DataLoader requires native Python `int` for `batch_size`.

### Solution Applied
Added type conversion throughout the hyperparameter tuning pipeline.

---

## Issue 2: timm API Change (FIXED ✅)

### Problem
```
module 'timm.optim.optim_factory' has no attribute 'add_weight_decay'
```

### Root Cause
Newer versions of `timm` library removed the `optim_factory.add_weight_decay` function.

### Solution Applied
Implemented weight decay parameter grouping manually:

```python
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    Add weight decay to optimizer, but skip certain parameters
    Typically skip biases and normalization layer parameters
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip biases, layer norms, and anything in skip_list
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]
```

This properly excludes biases and normalization parameters from weight decay, which is the standard practice.

---

## Verification

Run the test script to verify the fixes:
```bash
python3 test_tuning_fix.py
```

Test hyperparameter tuning:
```bash
python3 hyperparam_tuning.py --strategy quick --n_trials 3 --trial_epochs 20
```

### 1. Added Type Conversion Function
```python
def _to_python_type(self, value):
    """Convert numpy types to Python native types"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    else:
        return value
```

### 2. Updated Random Search
```python
# Before (WRONG)
params = {key: np.random.choice(values) 
         for key, values in search_space.items()}

# After (CORRECT)
params = {key: self._to_python_type(np.random.choice(values))
         for key, values in search_space.items()}
```

### 3. Updated Config Creation
```python
# Ensure all params are Python native types
for key, value in params.items():
    if hasattr(config, key):
        setattr(config, key, self._to_python_type(value))
```

### 4. Added Extra Safety in DataLoader Creation
```python
# Explicitly convert to int before passing to DataLoader
batch_size = int(config.batch_size)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # Now guaranteed to be Python int
    ...
)
```

## Verification

Run the test script to verify the fix:
```bash
python test_tuning_fix.py
```

Expected output:
```
Test 1: Type conversion
  np.int64(32) -> 32 (type: int)
  np.float64(0.002) -> 0.002 (type: float)
  ✓ Type conversion works!

Test 2: Config creation with numpy types
  ✓ Config creation works!

Test 3: DataLoader creation
  ✓ DataLoader created successfully with batch_size=32

All tests passed! Hyperparameter tuning should work now.
```

## Additional Improvements

### 1. Better Search Space Defaults
- Removed overly aggressive mask ratios (0.75 → max 0.7)
- Added higher learning rates (3e-3) based on EEG characteristics
- Adjusted depth ranges (removed 16, 20, 24 - too deep for EEG)

### 2. Improved num_heads Calculation
```python
# Old: Simple division (could cause issues)
config.num_heads = embed_dim // 64

# New: Handles edge cases
if embed_dim >= 512:
    config.num_heads = embed_dim // 64
elif embed_dim >= 256:
    config.num_heads = embed_dim // 64 if embed_dim % 64 == 0 else embed_dim // 32
else:
    config.num_heads = 4
```

## Usage

Now you can run hyperparameter tuning without errors:

```bash
# Quick test (3 trials to verify it works)
python3 hyperparam_tuning.py --strategy quick --method random --n_trials 3 --trial_epochs 20

# Full search
python3 hyperparam_tuning.py --strategy focused --method random --n_trials 30 --trial_epochs 30

# Or use the launcher for quick testing
python3 launcher.py --config light
```

## Files Modified
- `hyperparam_tuning.py` - Fixed type conversion + removed timm dependency
- `launcher.py` - Removed timm dependency
- `quick_param_test.py` - Removed timm dependency
- `train_mae_deap.py` - Already had the fix
- `test_tuning_fix.py` - Created verification script

## Status
✅ **BOTH ISSUES FIXED** - Hyperparameter tuning code now works correctly

## Current Training Challenge

While the code errors are fixed, the initial trials show **very low correlations (~0.0001)**. This indicates:

### What's Working:
- ✅ Code runs without errors
- ✅ Loss decreases (1.36 → 1.00)
- ✅ Model is learning something

### What Needs Improvement:
- ⚠️ Correlation stays near zero (reconstruction quality poor)
- ⚠️ Early stopping triggers too soon (epoch 11)

### Recommended Next Steps:

1. **Try the pre-optimized configs directly:**
   ```bash
   # These are designed for EEG and should work better
   python3 launcher.py --config light
   ```

2. **Run longer trials (correlation may improve after warmup):**
   ```bash
   python3 hyperparam_tuning.py --strategy quick --n_trials 5 --trial_epochs 50
   ```

3. **Test with less aggressive masking:**
   ```bash
   python3 launcher.py --config small --mask_ratio 0.3
   ```

4. **Check if data path is correct:**
   - Current: `/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz`
   - Verify file exists and has correct format

5. **Quick diagnostic:**
   ```bash
   python3 quick_param_test.py
   ```
   This tests 4 configurations in ~4 hours to find what works.

### Understanding the Results:
- Loss ~1.0 after warmup is actually **expected** initially
- Correlation should rise after warmup (epochs 10-30)
- Early stopping at epoch 11 might be too aggressive
- The original config (1024 dim, 24 layers, 75% mask) was likely too heavy for this dataset
