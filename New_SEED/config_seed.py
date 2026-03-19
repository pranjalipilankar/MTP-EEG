import os
import numpy as np

class Config_MAE_SEED:
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
        self.batch_size = 32        # Batch size
        self.clip_grad = 1.0        # Gradient clipping
        self.weight_decay = 0.01    # Weight decay
        self.accum_iter = 1         # Gradient accumulation
        
        # ============================================
        # Model Parameters
        # ============================================
        self.mask_ratio = 0.75      # Mask 75% of patches
        self.patch_size = 8         # Patch size for time dimension (104000 / 8 = 13000 patches)
        self.embed_dim = 1024       # Encoder embedding dimension
        self.decoder_embed_dim = 512  # Decoder embedding dimension
        self.depth = 24             # Encoder depth
        self.num_heads = 16         # Number of attention heads
        self.decoder_num_heads = 16  # Decoder attention heads
        self.mlp_ratio = 1.0        # MLP ratio
        
        # ============================================
        # SEED Dataset Parameters
        # ============================================
        self.data_path = '/home/ab_students/EEG-MTP/DATA/SEED_processed'
        self.num_channels = 31      # MAE trained on 31 channels (downsampled from 62)
        self.time_len = 104000      # Full trial length (520 seconds @ 200Hz)
        self.sampling_rate = 200    # SEED sampling rate
        
        # STAD channel configuration:
        # - LR: 16 channels (low-resolution input)
        # - HR: 31 channels (high-resolution, MAE trained on this)
        # - SR: 62 channels (super-resolution output)
        
        # Use segments instead of full trials (too long)
        self.use_segments = True
        self.segment_length = 4000  # 20 seconds @ 200Hz (4000 / 8 = 500 patches)
        self.segment_overlap = 0.5  # 50% overlap between segments
        
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
        self.output_path = os.path.join(self.root_path, 'trial_mae_SEED/results/')
        self.seed = 2024
        self.local_rank = 0
