import os
import numpy as np

class Config_MAE_LocalizeMI:
    def __init__(self):
        # ============================================
        # Parameters
        # ============================================
        self.lr = 1e-3              # Initial learning rate
        self.min_lr = 1e-6          # Minimum learning rate (non-zero for stability)
        self.warmup_epochs = 20     # Warmup epochs
        self.num_epoch = 300        # Total epochs
        self.batch_size = 16        # Smaller batch size due to 256 channels
        self.clip_grad = 1.0        # Gradient clipping
        self.weight_decay = 0.01    # Weight decay
        self.accum_iter = 1         # Gradient accumulation
        self.mask_ratio = 0.75      # Mask 75% of patches in pretrained MAE
        self.embed_dim = 1024       # Encoder embedding dimension
        self.decoder_embed_dim = 512  # Decoder embedding dimension
        self.depth = 24             # Encoder depth
        self.num_heads = 16         # Number of attention heads
        self.decoder_num_heads = 16  # Decoder attention heads
        self.sampling_rate = 500
        self.num_channels = 256     # SR-EEG with 256 
        self.hr_channels = 128
        self.lr_channels = 64

        # ============================================
        # Localize-MI Dataset Parameters
        # ============================================
        self.data_path = '/home/ab_students/EEG-MTP/DATA/Localize-MI/derivatives/epochs'
        self.num_channels = 256     # HD-EEG with 256 channels
        self.time_len = 2080        # Truncate to 2080 (divisible by 16) - ~260ms @ 8000Hz
        
        # Use individual epochs (already segmented)
        self.use_all_epochs = True  # Use all epochs from all runs
        
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
