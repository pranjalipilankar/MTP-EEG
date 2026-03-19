# Hyperparameter Tuning - Quick Reference

## 🚀 Quick Start (Choose One)

### 0️⃣ Having Issues? Diagnose First (1 minute)
```bash
python3 diagnose_training.py
```
Checks data, model, and configuration for common problems.

### 1️⃣ Compare Configurations First (1 minute)
```bash
python compare_configs.py
```
See all configurations side-by-side without training.

### 2️⃣ Test Best Configurations (4 hours)
```bash
python quick_param_test.py
```
Tests 4 pre-selected configurations with 50 epochs each.

### 3️⃣ Use Recommended Config (Immediate)
```bash
# Light configuration (recommended starting point)
python launcher.py --config light

# Small configuration (fastest, debugging)
python launcher.py --config small --epochs 50

# Balanced (production ready)
python launcher.py --config balanced
```

### 4️⃣ Full Hyperparameter Search (15+ hours)
```bash
python hyperparam_tuning.py --strategy focused --n_trials 30
```

---

## 📊 Configuration Options

| Config | Description | When to Use | Speed |
|--------|-------------|-------------|-------|
| **small** | 256 dim, 8 layers | Quick tests, debugging | ⚡⚡⚡ Fastest |
| **light** | 512 dim, 12 layers | **Recommended start** | ⚡⚡ Fast |
| **balanced** | 384 dim, 10 layers | Production deployment | ⚡ Medium |
| **finepatch** | 8 patch size | High temporal detail | 🐌 Slower |
| **original** | 1024 dim, 24 layers | ❌ Too heavy for EEG | 🐌🐌 Slowest |

---

## ⚙️ Key Parameters Explained

### Learning Rate (`lr`)
- **Original:** 1e-3
- **Recommended:** 1.5e-3 to 2e-3
- **Why:** EEG benefits from higher LR due to simpler structure

### Mask Ratio (`mask_ratio`)
- **Original:** 0.75 (75% masked)
- **Recommended:** 0.4 to 0.6 (40-60% masked)
- **Why:** EEG has more noise, aggressive masking hurts learning

### Model Depth (`depth`)
- **Original:** 24 layers
- **Recommended:** 8 to 12 layers
- **Why:** EEG doesn't need deep hierarchies like images

### Embed Dimension (`embed_dim`)
- **Original:** 1024
- **Recommended:** 256 to 512
- **Why:** EEG has fewer features than natural images

### Patch Size (`patch_size`)
- **Options:** 8, 16, 32
- **Trade-off:** 
  - Smaller (8): More detail, slower
  - Larger (32): Faster, less detail
  - **Recommended:** 16 (good balance)

---

## 📈 Success Metrics

### Target Correlation
- ✅ **Excellent:** > 0.60
- ✅ **Good:** 0.45 - 0.60
- ⚠️ **Moderate:** 0.20 - 0.45
- ❌ **Poor:** < 0.20

### What to Watch
1. **First 10 epochs:** Correlation should reach > 0.01
2. **After warmup:** Correlation should grow steadily
3. **Convergence:** Plateau around 0.45-0.70

---

## 🔧 Common Issues

### Issue: Correlation stays ~0.00
**Solution:**
```bash
# Try lighter model with less masking
python launcher.py --config light --mask_ratio 0.4
```

### Issue: Loss not decreasing
**Solution:**
```bash
# Increase learning rate
python launcher.py --config light --lr 0.002
```

### Issue: NaN losses
**Solution:**
```bash
# Reduce learning rate
python launcher.py --config light --lr 0.0005
```

### Issue: Out of memory
**Solution:**
```bash
# Reduce batch size or use smaller model
python launcher.py --config small --batch_size 32
```

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `launcher.py` | Easy way to run different configs |
| `compare_configs.py` | Compare configs without training |
| `quick_param_test.py` | Test 4 configs (4 hours) |
| `hyperparam_tuning.py` | Full automated search (15+ hours) |
| `configs_recommended.py` | Pre-made optimal configs |
| `TUNING_GUIDE.md` | Detailed documentation |

---

## 🎯 Recommended Workflow

### Beginner / Limited Time
```bash
# 1. Compare options
python compare_configs.py

# 2. Use recommended config
python launcher.py --config light

# 3. If not good, try balanced
python launcher.py --config balanced
```

### Intermediate / Have Time
```bash
# 1. Quick test of 4 configs
python quick_param_test.py

# 2. Use best performer for full training
python launcher.py --config <best> --epochs 200
```

### Advanced / Research
```bash
# 1. Full hyperparameter search
python hyperparam_tuning.py --strategy focused --n_trials 30

# 2. Use auto-generated optimal config
python -c "
from tuning_results.config_deap_optimized import Config_MAE_DEAP_Tuned
# Train with this config
"
```

---

## 💡 Pro Tips

1. **Always check correlation in first 20 epochs**
   - If < 0.01, something is wrong
   
2. **Don't run full 200 epochs initially**
   - Test with 50 epochs first
   - Only do full run when config is proven

3. **GPU memory issues?**
   - Use smaller config or reduce batch size
   - 16GB GPU: comfortable with all configs
   - 8GB GPU: use small/light with batch_size=32

4. **Best value for time:**
   - Start with `python launcher.py --config light`
   - If correlation < 0.3 after 30 epochs, stop and try different config
   - Don't waste compute on bad configs

---

## 📞 Quick Decision Guide

**I want the fastest test:**
→ `python launcher.py --config small --epochs 30`

**I want a good starting point:**
→ `python launcher.py --config light`

**I want best performance:**
→ `python quick_param_test.py` then use winner

**I want to explore thoroughly:**
→ `python hyperparam_tuning.py --strategy focused`

---

## 📊 Expected Training Time

| Config | GPU | Epoch Time | 100 Epochs | 200 Epochs |
|--------|-----|------------|------------|------------|
| Small | RTX 3090 | ~1 min | ~2 hrs | ~3.5 hrs |
| Light | RTX 3090 | ~2 min | ~3.5 hrs | ~7 hrs |
| Balanced | RTX 3090 | ~1.5 min | ~2.5 hrs | ~5 hrs |
| Original | RTX 3090 | ~4 min | ~7 hrs | ~14 hrs |

*Times are approximate and vary by GPU*

---

## ✅ Checklist Before Training

- [ ] Data path correct in config? (`/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz`)
- [ ] CUDA available? (check with `nvidia-smi`)
- [ ] Output directory writable?
- [ ] Dependencies installed? (`pip install pandas timm`)
- [ ] Enough disk space? (~500MB for checkpoints)

---

**Last Updated:** February 17, 2026
**For Issues:** See TUNING_GUIDE.md for detailed troubleshooting
