# Hyperparameter Tuning Guide for MAE-DEAP

## Problem Diagnosis

Your current training shows **very low correlation (~0.0001)** after 10 epochs, indicating the hyperparameters are not optimal. Key issues:

1. **Model too deep** (24 layers) - overkill for EEG
2. **Masking too aggressive** (75%) - EEG has lower SNR than images  
3. **Learning rate may be too low** (1e-3)
4. **Model too large** (1024 dim) - more parameters than needed

## Quick Start

### Option 1: Test Pre-Defined Configurations (RECOMMENDED)

Test 4 carefully designed configurations in ~4 hours:

```bash
python quick_param_test.py
```

This will test:
- **Light Model** (512 dim, 12 layers, 50% mask, 2e-3 LR)
- **Small Patches** (8 patch size for fine temporal detail)
- **Very Light** (256 dim, 8 layers - fastest baseline)
- **Balanced** (384 dim, 10 layers - production ready)

### Option 2: Automated Hyperparameter Search

#### Random Search (30 trials, ~15 hours)
```bash
python hyperparam_tuning.py \
    --strategy focused \
    --method random \
    --n_trials 30 \
    --trial_epochs 30
```

#### Quick Exploration (10 trials, ~5 hours)
```bash
python hyperparam_tuning.py \
    --strategy quick \
    --method random \
    --n_trials 10 \
    --trial_epochs 25
```

#### Grid Search (limited combinations)
```bash
python hyperparam_tuning.py \
    --strategy focused \
    --method grid \
    --n_trials 20 \
    --trial_epochs 30
```

### Option 3: Use Recommended Configs Directly

Import and use pre-made configurations:

```python
from configs_recommended import (
    Config_MAE_DEAP_Light,      # Good starting point
    Config_MAE_DEAP_Small,      # Fastest training
    Config_MAE_DEAP_Balanced,   # Recommended for production
    Config_MAE_DEAP_FinePatch   # Better temporal resolution
)

# Use in your training
config = Config_MAE_DEAP_Light()
```

## Key Parameters to Tune

### Critical Parameters (Tune First)

| Parameter | Current | Recommended Range | Impact |
|-----------|---------|-------------------|--------|
| `lr` | 1e-3 | **1e-3 to 3e-3** | Higher LR often better for EEG |
| `mask_ratio` | 0.75 | **0.4 to 0.6** | Less masking for noisy EEG |
| `depth` | 24 | **8 to 12** | Lighter models work better |
| `embed_dim` | 1024 | **256 to 512** | Smaller sufficient for EEG |
| `patch_size` | 16 | **8 to 32** | Trade-off: detail vs efficiency |

### Secondary Parameters

| Parameter | Current | Recommended | Notes |
|-----------|---------|-------------|-------|
| `batch_size` | 32 | **64** | Larger = more stable |
| `warmup_epochs` | 20 | **5 to 10** | Shorter warmup |
| `min_lr` | 1e-6 | **1e-5** | Non-zero minimum crucial |
| `weight_decay` | 0.01 | **0.01 to 0.03** | Regularization |
| `num_epoch` | 200 | **80 to 120** | Converges faster |

## Expected Results

### Good Performance Indicators
- ✓ Correlation > 0.45 within 30 epochs
- ✓ Correlation > 0.60 at convergence (excellent)
- ✓ Steady loss decrease without plateaus
- ✓ No NaN losses

### Warning Signs
- ⚠️ Correlation < 0.05 after 20 epochs → restart with different params
- ⚠️ Loss not decreasing after warmup → LR too low
- ⚠️ NaN losses → LR too high or gradient issues
- ⚠️ Loss oscillating wildly → batch size too small

## Understanding Search Strategies

### 'quick' Strategy
- **10-15 combinations**
- Focus on 3 key parameters (LR, mask_ratio, depth)
- Good for: Initial exploration, limited compute
- Runtime: ~3-5 hours

### 'focused' Strategy (RECOMMENDED)
- **20-30 combinations**  
- Tests 4-5 key parameters
- Good for: Finding optimal config
- Runtime: ~10-15 hours

### 'exhaustive' Strategy
- **50+ combinations**
- Tests all parameters thoroughly
- Good for: Research, thorough optimization
- Runtime: ~24+ hours

## Output Files

After tuning, you'll get:

```
tuning_results/
├── tuning_results.csv           # All trial results
├── tuning_log.txt               # Detailed logs
└── config_deap_optimized.py     # Best configuration found
```

## Using Tuning Results

### 1. Check Results
```bash
# View sorted results
column -t -s, tuning_results/tuning_results.csv | less
```

### 2. Use Best Config
```python
# Auto-generated optimal config
from tuning_results.config_deap_optimized import Config_MAE_DEAP_Tuned

config = Config_MAE_DEAP_Tuned()
# Use for full training
```

### 3. Analyze Trends
```python
import pandas as pd
df = pd.read_csv('tuning_results/tuning_results.csv')

# Top 5 by correlation
print(df.nlargest(5, 'final_cor'))

# Impact of mask_ratio
print(df.groupby('mask_ratio')['final_cor'].mean())
```

## Best Practices

### For Limited Compute
1. Start with `quick_param_test.py` (tests 4 configs in ~4 hours)
2. Pick best performer
3. Fine-tune around it with random search

### For Thorough Optimization
1. Run `focused` random search (30 trials)
2. Identify top 3 configs
3. Run full training (200 epochs) on top 3
4. Pick final winner

### For Research
1. Run `exhaustive` grid search
2. Analyze parameter interactions
3. Document findings

## EEG-Specific Insights

Unlike natural images, EEG has:
- **Lower SNR** → need less aggressive masking (0.4-0.6 vs 0.75)
- **Simpler structure** → lighter models work (8-12 layers vs 24)
- **Temporal dependencies** → smaller patches better (8-16 vs 16-32)
- **Less data** → higher LR works (1e-3 to 3e-3)

## Troubleshooting

### Correlation stays near 0
- **Try:** Lower mask_ratio to 0.4-0.5
- **Try:** Increase learning rate to 2e-3
- **Try:** Reduce model depth to 8-10

### Loss not decreasing
- **Check:** Learning rate schedule (should not go to 0)
- **Try:** Increase `min_lr` to 1e-5
- **Try:** Shorten warmup to 5-10 epochs

### Training unstable (NaN)
- **Try:** Reduce learning rate
- **Try:** Increase gradient clipping (1.5-2.0)
- **Try:** Reduce batch size

### Out of memory
- **Try:** Reduce batch_size (32 → 16)
- **Try:** Reduce embed_dim (512 → 384)
- **Try:** Increase patch_size (16 → 32)

## Quick Decision Tree

```
Start Here
│
├─ Have time for full search? (15+ hours)
│  └─ YES → python hyperparam_tuning.py --strategy focused --n_trials 30
│
├─ Need results quickly? (4 hours)
│  └─ YES → python quick_param_test.py
│
└─ Want safe default?
   └─ YES → Use Config_MAE_DEAP_Light from configs_recommended.py
```

## Example: Full Workflow

```bash
# 1. Quick test of promising configs
python quick_param_test.py

# 2. If you want more thorough search
python hyperparam_tuning.py \
    --strategy focused \
    --method random \
    --n_trials 30 \
    --trial_epochs 30 \
    --output_dir tuning_results

# 3. Check results
cat tuning_results/tuning_log.txt | grep "Rank 1" -A 10

# 4. Use best config for full training
python train_mae_deap.py --config tuning_results/config_deap_optimized.py
```

## Next Steps After Tuning

Once you find good hyperparameters (correlation > 0.45):

1. **Full Training**: Run 200 epochs with best config
2. **Validation**: Check on val set
3. **Use for STAD**: Use pretrained model for super-resolution
4. **Fine-tuning**: Optionally fine-tune on specific task

## Contact

If you encounter issues, check:
- CUDA memory (use `nvidia-smi`)
- Data path is correct
- All dependencies installed (`pip install pandas timm`)
