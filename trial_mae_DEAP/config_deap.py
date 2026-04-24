import os
import numpy as np

class Config_MAE_DEAP():
    def __init__(self):
        super().__init__()
        
        # ============================================
        # Critical Learning Rate Fixes
        # ============================================
        self.lr = 1e-3              # ✓ Higher initial LR (was 2.5e-4)
        self.min_lr = 1e-6          # ✓ CRITICAL: Non-zero minimum (was 0.0)
        self.warmup_epochs = 20     # ✓ Shorter warmup (was 40)
        self.num_epoch = 200        # ✓ Fewer epochs (was 500)
        
        # ============================================
        # Training Stability Improvements
        # ============================================
        self.batch_size = 32        # ✓ Smaller for stability (was 64)
        self.clip_grad = 1.0        # ✓ Relaxed clipping (was 0.8)
        self.weight_decay = 0.01    # ✓ Less regularization (was 0.05)
        self.accum_iter = 1
        
        # ============================================
        # Model Parameters (SEED4-like MAE backbone)
        # ============================================
        self.mask_ratio = 0.75      # ✓ Keep 75% as in paper
        self.patch_size = 8         # More temporal patches for EEG dynamics
        self.embed_dim = 768        # Match SEED4 encoder width
        self.decoder_embed_dim = 384
        self.depth = 12
        self.num_heads = 12
        self.decoder_num_heads = 8
        self.decoder_depth = 4
        self.mlp_ratio = 4.0
        self.norm_pix_loss = True   # Stable MAE target normalization
        
        # ============================================
        # DEAP Dataset Parameters
        # ============================================
        self.data_path = '/DATA/EEG-MTP/DEAP'
        self.num_channels = 32      # DEAP EEG channels
        self.time_len = 8064        # Full trial length (or use chunks)
        
        # Data augmentation
        self.aug_times = 1          # ✓ Start conservative
        self.sparse_rate = 0.1      # ✓ Light augmentation (10% dropout)
        
        # ============================================
        # Optional: Focus mechanism (if needed)
        # ============================================
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6
        
        # ============================================
        # Paths
        # ============================================
        self.root_path = '/home/ab_students/EEG-MTP/'
        self.output_path = os.path.join(self.root_path, 'trial_mae_DEAP/results_kfold_subjectwise/')
        self.seed = 2024
        self.local_rank = 0