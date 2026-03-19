# SEED-IV Dataset MAE Pretraining

## Overview
Masked Autoencoder (MAE) pretraining for SEED-IV EEG dataset with emotion labels.

## Dataset Specifications

| Feature | SEED | SEED-IV (PrC-1 Preprocessed) |
|---------|------|------------------------------|
| **Channels** | 62 → 31 selected | 62 (all channels) |
| **Sampling Rate** | 200 Hz | 250 Hz |
| **Window Length** | 4000 samples (20s segments) | 1000 samples (4s windows) |
| **Subjects** | 15 | 15 |
| **Sessions** | 3 per subject | 3 per subject |
| **Trials** | 24 per session | 24 per session |
| **Windows** | ~50 per trial | ~20-25 per trial |
| **Total Windows** | ~30,000+ | ~24,000-27,000 |
| **Emotions** | 3 classes | 4 classes (neutral, sad, fear, happy) |
| **Patch Size** | 8 | 8 |
| **Num Patches** | 500 per segment | 125 per window |

## Prerequisites

### 1. Preprocess SEED-IV Data
First, run the preprocessing pipeline to generate the required data format:

```bash
cd /home/ab_students/EEG-MTP/Noel-Preprocessing
python FinalPrC-1.py
```

This creates the processed data structure:
```
DATA/seed4/eeg_processed_data/
├── 1/          # Session 1
│   ├── 1_20160518/
│   │   ├── X_prc1.npy              # (n_windows, 62, 1000)
│   │   ├── labels.npy              # (n_windows,)
│   │   ├── prc1_meta.json
│   │   └── ...
│   └── ... (14 more subjects)
├── 2/          # Session 2
└── 3/          # Session 3
```

### 2. Verify Data Structure

```bash
cd /home/ab_students/EEG-MTP/trial_mae_SEED4
python3 dataset_seed4.py
```

This will test data loading and print dataset statistics.

## Training

### Quick Start

```bash
cd /home/ab_students/EEG-MTP/trial_mae_SEED4
python train_mae_seed4.py --device cuda
```

### Training Options

```bash
# Training with custom epochs
python train_mae_seed4.py --epochs 100

# Training with custom batch size
python train_mae_seed4.py --batch_size 128

# Resume from checkpoint
python train_mae_seed4.py --resume results/checkpoint_epoch_50.pth

# CPU training (slower)
python train_mae_seed4.py --device cpu
```

### Training Configuration

Edit `config_seed4.py` to adjust:

```python
# Model architecture
self.embed_dim = 768              # Encoder embedding dimension
self.depth = 12                   # Encoder depth
self.decoder_embed_dim = 384      # Decoder embedding dimension
self.decoder_depth = 4            # Decoder depth

# Training parameters
self.lr = 1e-3                    # Initial learning rate
self.batch_size = 64              # Batch size
self.mask_ratio = 0.75            # Mask 75% of patches
self.num_epoch = 200              # Total epochs
self.warmup_epochs = 20           # Warmup epochs

# Data split (session-based)
self.train_sessions = ['1', '2']  # Sessions for training
self.test_sessions = ['3']        # Session for testing
```

## Running Training with nohup

### K-Fold Cross-Validation Training

#### RAW vs PREPROCESSED Comparison (Recommended)

```bash
cd /home/ab_students/EEG-MTP/trial_mae_SEED4

# Compare both data sources across all 5 folds
nohup python3 train_compare_kfold.py \
  --data_source both \
  --n_folds 5 \
  --device cuda \
  --epochs 100 \
  > logs/kfold_compare_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the process ID
echo $! > logs/train_compare.pid
echo "Training started with PID: $(cat logs/train_compare.pid)"

# Monitor progress in real-time
tail -f logs/kfold_compare_*.log
```

#### 31-Channel RAW Data Training

```bash
cd /home/ab_students/EEG-MTP/trial_mae_SEED4

# Train on 31-channel HR data with k-fold CV
nohup python3 train_mae_31ch_kfold_raw.py \
  --n_folds 5 \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --data_path /home/ab_students/EEG-MTP/DATA/seed4/raw_data.npz \
  --save_dir /home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_raw \
  > logs/train_31ch_kfold_raw_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > logs/train_31ch.pid
echo "31-channel training started with PID: $(cat logs/train_31ch.pid)"

# Monitor
tail -f logs/train_31ch_kfold_raw_*.log
```

#### Single Data Source Training

```bash
# RAW data only
nohup python3 train_compare_kfold.py \
  --data_source raw \
  --n_folds 5 \
  --device cuda \
  --epochs 100 \
  > logs/kfold_raw_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PREPROCESSED data only
nohup python3 train_compare_kfold.py \
  --data_source preprocessed \
  --n_folds 5 \
  --device cuda \
  --epochs 100 \
  > logs/kfold_prep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Monitoring Running Jobs

```bash
# Create logs directory if it doesn't exist
mkdir -p logs

# Check if process is still running
ps aux | grep train_compare_kfold

# Or check by PID
ps -p $(cat logs/train_compare.pid)

# Monitor GPU usage
watch -n 1 nvidia-smi

# Follow log output with highlighting
tail -f logs/kfold_compare_*.log | grep --color=auto -E "FOLD|BEST|Epoch|✓|✗|⚠"

# Check progress for specific fold
grep "Fold.*complete" logs/kfold_compare_*.log

# Check best correlations so far
grep "BEST" logs/kfold_compare_*.log | tail -20
```

### Stopping/Killing Jobs

```bash
# Get PID
PID=$(cat logs/train_compare.pid)

# Gracefully stop
kill $PID

# Force kill if needed
kill -9 $PID

# Or kill by name
pkill -f train_compare_kfold
```

### Using `screen` (Better for Long Training)

`screen` is recommended over `nohup` for long-running tasks because you can re-attach:

```bash
# Start a screen session
screen -S mae_kfold

# Inside screen, run training (NO nohup needed)
cd /home/ab_students/EEG-MTP/trial_mae_SEED4
python3 train_compare_kfold.py --n_folds 5 --device cuda --epochs 100

# Detach from screen: Press Ctrl+A, then D

# List all screens
screen -ls

# Re-attach to screen
screen -r mae_kfold

# Kill screen when done
screen -X -S mae_kfold quit
```

### Checking Results After Training

```bash
# View final summary
tail -100 logs/kfold_compare_*.log

# Check if training completed successfully
grep "✅.*complete" logs/kfold_compare_*.log

# View comparison summary
cat results_compare_kfold_5fold/comparison_summary.json | python3 -m json.tool

# Check individual fold results
for fold in 0 1 2 3 4; do
  echo "=== FOLD $fold ==="
  cat results_compare_kfold_5fold/raw/fold_$fold/fold_results.json | python3 -m json.tool | grep -A2 "test_correlation"
  cat results_compare_kfold_5fold/preprocessed/fold_$fold/fold_results.json | python3 -m json.tool | grep -A2 "test_correlation"
done
```

### Example: Complete Workflow

```bash
# 1. Setup
cd /home/ab_students/EEG-MTP/trial_mae_SEED4
mkdir -p logs

# 2. Start training with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
nohup python3 train_compare_kfold.py \
  --data_source both \
  --n_folds 5 \
  --device cuda \
  --epochs 100 \
  > logs/kfold_compare_${TIMESTAMP}.log 2>&1 &

# 3. Save PID
PID=$!
echo $PID > logs/train_${TIMESTAMP}.pid
echo "Training started at $TIMESTAMP with PID: $PID"

# 4. Monitor in new terminal
tail -f logs/kfold_compare_${TIMESTAMP}.log

# 5. Check specific metrics
grep "Best Val Cor" logs/kfold_compare_${TIMESTAMP}.log

# 6. When done, cleanup
rm logs/train_${TIMESTAMP}.pid
```

### Email Notification (Optional)

```bash
# Install mail utility (if not already)
sudo apt-get install mailutils

# Run with email notification
nohup python3 train_compare_kfold.py \
  --n_folds 5 \
  --device cuda \
  --epochs 100 \
  > logs/kfold.log 2>&1 \
  && echo "K-fold training completed successfully! Check results at: $(hostname):$(pwd)/results_compare_kfold_5fold/" | mail -s "MAE Training Completed" your.email@example.com \
  || echo "K-fold training FAILED! Check logs at: $(hostname):$(pwd)/logs/kfold.log" | mail -s "MAE Training Failed" your.email@example.com &
```

### Useful Aliases

Add these to `~/.bashrc` for convenience:

```bash
# MAE training aliases
alias mae_gpu='watch -n 1 nvidia-smi'
alias mae_logs='tail -f /home/ab_students/EEG-MTP/trial_mae_SEED4/logs/*.log'
Linear: 384 → 62×8 (unpatchify)
       ↓
Reconstruction: (batch, 62, 1000)
```

## Output Files

### Training Outputs (`results/`)
- `best_model.pth` - Best model checkpoint (based on validation correlation)
- `checkpoint_epoch_N.pth` - Periodic checkpoints
- `training_log.txt` - Detailed training log
- `training_history.png` - Training curves (loss, correlation, LR)

### Evaluation Outputs (`results/evaluation/`)
- `evaluation_metrics.npz` - Detailed metrics in NumPy format
- `reconstruction_samples.png` - Visual comparison of original vs. reconstructed signals
- `metrics_distribution.png` - Distribution of reconstruction quality across test samples

## Key Differences from SEED

### 1. **Shorter Sequences**
- SEED-IV windows are 1000 samples (4s) vs. SEED segments of 4000 samples (20s)
- Fewer patches: 125 vs. 500
- Smaller model: 768-dim encoder vs. 1024-dim

### 2. **All Channels**
- SEED-IV uses all 62 channels (no channel selection needed)
- SEED typically uses 31 selected channels

### 3. **Different Emotions**
- SEED-IV: 4 classes (neutral, sad, fear, happy)
- SEED: 3 classes (positive, neutral, negative)

### 4. **Preprocessing**
- SEED-IV uses PrC-1 preprocessing (bandpass 0.1-100 Hz, notch 50 Hz, soft clipping)
- Already z-normalized and ready for MAE training

## Troubleshooting

### Issue: "No data files found"
**Solution**: Run preprocessing first (`FinalPrC-1.py`)

### Issue: Out of memory
**Solutions**:
```bash
# Reduce batch size
python train_mae_seed4.py --batch_size 32

# Or reduce model size in config_seed4.py:
self.embed_dim = 512
self.depth = 8
```

### Issue: Training not converging
**Solutions**:
- Check learning rate (try 5e-4 or 2e-3)
- Increase warmup epochs (try 30-40)
- Verify data is properly normalized (check dataset_seed4.py test output)

### Issue: Reconstruction quality is poor
**Possible causes**:
- Not enough training (try more epochs)
- Mask ratio too high (try 0.5 or 0.6 instead of 0.75)
- Model too small (increase embed_dim or depth)

## Use Cases

### 1. **Unsupervised Pretraining**
Train MAE on all SEED-IV data (ignoring emotion labels) to learn general EEG representations.

### 2. **Emotion Recognition**
Use pretrained encoder as initialization for emotion classification models.

### 3. **Super-Resolution**
Use pretrained model for channel or temporal super-resolution tasks.

### 4. **Transfer Learning**
Pretrain on SEED-IV, fine-tune on other EEG datasets (e.g., DEAP, AMIGOS).

## Files

### Core Training Files
- `config_seed4.py` - Configuration for SEED-IV dataset
- `mae_for_eeg.py` - MAE model architecture (shared with SEED)
- `trainer.py` - Training loop utilities (shared)
- `utils.py` - Helper functions (shared)

### Dataset Loaders
- `dataset_seed4.py` - SEED-IV preprocessed data loader (session-based split, deprecated)
- `dataset_seed4_raw.py` - SEED-IV raw data loader (session-based split, deprecated)
- `dataset_seed4_kfold.py` - **NEW**: Preprocessed data with subject-based k-fold splits
- `dataset_seed4_raw_kfold.py` - **NEW**: Raw data with subject-based k-fold splits

### Training Scripts
- `train_mae_seed4.py` - Standard training (session-based, deprecated for comparison)
- `train_kfold.py` - **RECOMMENDED**: Subject-based k-fold training for single data source
- `train_compare.py` - RAW vs PREP comparison (session-based, deprecated)
- `train_compare_kfold.py` - **RECOMMENDED**: RAW vs PREP with k-fold cross-validation

### Evaluation Scripts
- `test_mae_seed4.py` - Evaluation script with visualization
- `compare_models.py` - Model comparison and visualization script

### Documentation
- `README.md` - This file
- `KFOLD_GUIDE.md` - Comprehensive guide to subject-based k-fold cross-validation

## Citation

If you use SEED-IV dataset:
```bibtex
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

## Notes

- **Training time**: ~2-3 hours on single GPU for 200 epochs
- **Memory usage**: ~6-8 GB GPU memory with batch_size=64
- **Recommended GPU**: NVIDIA RTX 3090 or better
- **CPU training**: Possible but ~20x slower

## Contact

For issues specific to SEED-IV preprocessing or MAE training, refer to:
- Preprocessing: `Noel-Preprocessing/README.md`
- Data structure: `Noel-Preprocessing/SEED4_DATA_STRUCTURE.md`
- Loading utilities: `Noel-Preprocessing/load_seed4_data.py`
