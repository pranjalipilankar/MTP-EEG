# STAD Architecture for SEED Dataset

## Channel Resolution Hierarchy

- **LR (Low-Resolution)**: 16 channels - Input to STAD
- **HR (High-Resolution)**: 31 channels - MAE trained on this
- **SR (Super-Resolution)**: 62 channels - Final output

## Architecture Flow

```
Input: 16-channel LR EEG (4000 samples @ 200Hz)
         ↓
    [STC Module]
         ↓
   Conditioning Features (16ch → latent embeddings)
         ↓
Training Target: 62-channel SR EEG
         ↓
    62 → 31 channels (downsample for MAE compatibility)
         ↓
   [Pretrained MAE Encoder] (trained on 31 channels)
         ↓
   Latent Space (500 patches × 1024 dim)
         ↓
   [Diffusion Process with MTD]
      - Noise: ε ~ N(0,1)
      - Conditioning: from STC(16-ch LR)
      - Denoising: MTD predicts ε
         ↓
   Clean Latent (500 patches × 1024 dim)
         ↓
   [MAE Decoder]
         ↓
   31-channel reconstruction
         ↓
   [Channel Upsampling] (Linear 31→62)
         ↓
Output: 62-channel SR EEG
```

## Key Components

### 1. Pretrained MAE
- **Input**: 31 channels × 4000 samples
- **Trained on**: SEED dataset with 31 channels (downsampled from 62)
- **Purpose**: Provides latent representation that captures EEG structure
- **Status**: Pretrained, frozen initially, fine-tuned after epoch 50

### 2. STC (Spatio-Temporal Conditioning) Module
- **Input**: 16-channel LR EEG
- **Features**:
  - Graph harmonic spatial embeddings
  - Temporal convolution + transformer
  - Patch-based processing
- **Output**: Conditioning tokens for cross-attention in MTD

### 3. MTD (Multi-Scale Transformer Denoising) Module
- **Works in**: Latent space (500 patches × 1024 dim)
- **Mechanism**: 
  - Cross-attention with STC conditioning
  - Self-attention for denoising
  - Predicts noise ε to be removed
- **Output**: Denoised latent representation

### 4. Channel Upsampling
- **Linear projection**: 31 → 62 channels
- **Applied**: After MAE decoding
- **Purpose**: Map from MAE output (31ch) to target SR (62ch)

## Training Details

### Data Processing
1. Load 62-channel SEED data
2. Create three versions per segment:
   - **SR**: 62 channels (target)
   - **HR**: 31 channels (for MAE compatibility)
   - **LR**: 16 channels (input)
3. Bandpass filter: 1-40 Hz
4. Normalize per channel per segment

### Loss Function
```python
Loss = MSE(predicted_noise, actual_noise)
```

### Metrics (logged every 30 epochs)
- **PCC**: Pearson Correlation Coefficient
- **RMSE**: Root Mean Square Error
- **SNR**: Signal-to-Noise Ratio (dB)
- **MAE**: Mean Absolute Error

### Training Schedule
- **Epochs**: 300
- **Batch size**: 16
- **Learning rate**: 2e-4
- **MAE unfreezing**: Epoch 50 (LR reduced to 2e-5)
- **Optimizer**: AdamW with cosine annealing

## Why This Architecture?

1. **31-channel MAE**: Provides strong latent representations from pretrained model
2. **16-channel input**: Simulates low-resolution scenario for super-resolution task
3. **62-channel output**: Full SEED resolution for maximum spatial detail
4. **Latent diffusion**: More stable than pixel-space diffusion
5. **STC conditioning**: Guides super-resolution using LR spatial-temporal patterns

## File Structure

- `train_stad_seed.py`: Main training script
- `config_seed.py`: Configuration (documents channel setup)
- `mtd_dreamdiff.py`: MTD module (denoising)
- `spatio_temporal_condition.py`: STC module (conditioning)
- `mae_for_eeg.py`: MAE architecture
- `utils.py`: Helper functions
