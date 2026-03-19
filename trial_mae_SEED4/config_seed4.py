import os
import numpy as np

class Config_MAE_SEED4:
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
        self.batch_size = 64        # Batch size (can be larger since windows are shorter)
        self.clip_grad = 1.0        # Gradient clipping
        self.weight_decay = 0.01    # Weight decay
        self.accum_iter = 1         # Gradient accumulation
        
        # ============================================
        # Model Parameters
        # ============================================
        self.mask_ratio = 0.75      # Mask 75% of patches
        self.patch_size = 8         # Patch size for time dimension (1000 / 8 = 125 patches)
        self.embed_dim = 768        # Encoder embedding dimension (smaller due to shorter sequences)
        self.decoder_embed_dim = 384  # Decoder embedding dimension
        self.depth = 12             # Encoder depth (reduced for shorter sequences)
        self.num_heads = 12         # Number of attention heads
        self.decoder_num_heads = 8  # Decoder attention heads
        self.decoder_depth = 4      # Decoder depth
        self.mlp_ratio = 4.0        # MLP ratio
        
        # ============================================
        # SEED-IV Dataset Parameters (after PrC-1 preprocessing)
        # ============================================
        self.data_path = '/home/ab_students/EEG-MTP/DATA/seed4/eeg_processed_data'
        self.num_channels = 62      # SEED-IV has 62 EEG channels
        self.time_len = 1000        # Window length (4 seconds @ 250Hz)
        self.sampling_rate = 250    # SEED-IV processed sampling rate
        
        # No segmentation needed - windows are already 4 seconds
        self.use_segments = False
        
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
        self.output_path = os.path.join(self.root_path, 'trial_mae_SEED4/results/')
        self.seed = 2024
        self.local_rank = 0
        
        # ============================================
        # Additional Settings
        # ============================================
        self.num_workers = 4
        self.pin_memory = True
        self.save_frequency = 20    # Save checkpoint every N epochs
        
    def __str__(self):
        """Print configuration"""
        config_str = "\n" + "="*80 + "\n"
        config_str += "SEED-IV MAE Configuration\n"
        config_str += "="*80 + "\n"
        config_str += f"Dataset:\n"
        config_str += f"  Channels: {self.num_channels}\n"
        config_str += f"  Sampling Rate: {self.sampling_rate} Hz\n"
        config_str += f"  Window Length: {self.time_len} samples ({self.time_len/self.sampling_rate:.1f}s)\n"
        config_str += f"  Num Patches: {self.time_len // self.patch_size}\n"
        config_str += f"\nModel:\n"
        config_str += f"  Encoder Dim: {self.embed_dim}\n"
        config_str += f"  Encoder Depth: {self.depth}\n"
        config_str += f"  Decoder Dim: {self.decoder_embed_dim}\n"
        config_str += f"  Mask Ratio: {self.mask_ratio}\n"
        config_str += f"\nTraining:\n"
        config_str += f"  Batch Size: {self.batch_size}\n"
        config_str += f"  Learning Rate: {self.lr}\n"
        config_str += f"  Epochs: {self.num_epoch}\n"
        config_str += "="*80 + "\n"
        return config_str

# If RAW data is not normalized:
using_raw_data = False  # Example flag, set this based on your data
if using_raw_data:
    config = Config_MAE_SEED4()
    config.lr = 1e-4  # Reduce from 1e-3
    config.min_lr = 1e-6
