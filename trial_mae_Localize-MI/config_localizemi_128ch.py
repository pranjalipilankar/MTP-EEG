import os
import numpy as np

class Config_MAE_LocalizeMI_128ch:
    """Configuration for 128-channel MAE on Localize-MI (downsampled from 256)"""
    
    def __init__(self):
        # ============================================
        # Training Parameters
        # ============================================
        self.lr = 1e-3              # Initial learning rate
        self.min_lr = 1e-6          # Minimum learning rate (non-zero for stability)
        self.warmup_epochs = 20     # Warmup epochs
        self.num_epoch = 200        # Total epochs
        
        # ============================================
        # Training Stability
        # ============================================
        self.batch_size = 64        # Larger batch for better gradient estimates
        self.clip_grad = 1.0        # Gradient clipping
        self.weight_decay = 0.05    # Increased weight decay for regularization
        self.accum_iter = 1         # Gradient accumulation
        
        # ============================================
        # Model Parameters (Reduced for small dataset)
        # ============================================
        self.mask_ratio = 0.75      # Mask 75% of patches
        self.patch_size = 16        # Patch size for time dimension
        self.embed_dim = 512        # Reduced from 1024 (less overfitting)
        self.decoder_embed_dim = 256  # Reduced from 512
        self.depth = 12             # Reduced from 24 (half depth)
        self.num_heads = 8          # Reduced from 16 (matches embed_dim/64)
        self.decoder_num_heads = 8  # Reduced from 16
        self.mlp_ratio = 2.0        # Increased from 1.0 for better feature learning
        
        # ============================================
        # Localize-MI Dataset Parameters (128 channels)
        # ============================================
        self.data_path = '/home/ab_students/EEG-MTP/DATA/Localize-MI/derivatives/epochs'
        self.num_channels = 128     # Downsampled from 256 via EGI montage (every 2nd channel)
        self.time_len = 256         # 512ms @ 500Hz → 16 patches (better temporal context)
        self.sampling_rate = 500    # Sampling rate
        
        # Use individual epochs (already segmented)
        self.use_all_epochs = True  # Use all epochs from all runs
        
        # ============================================
        # Channel Downsampling (256 → 128)
        # ============================================
        self.downsample_channels = True  # Enable channel downsampling
        self.original_channels = 256     # Source channels from raw data
        self.downsample_factor = 2       # EGI montage: every 2nd channel (256/2 = 128)
        
        # ============================================
        # Optional: Focus mechanism
        # ============================================
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6
        
        # ============================================
        # Paths
        # ============================================
        self.root_path = '/home/ab_students/EEG-MTP/'
        self.output_path = os.path.join(self.root_path, 'trial_mae_Localize-MI/results_128ch/')
        self.seed = 2024
        self.local_rank = 0
