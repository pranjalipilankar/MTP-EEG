"""
Recommended configurations for DEAP EEG based on domain knowledge

Key insights for EEG vs Images:
1. EEG has lower SNR → need less aggressive masking
2. Temporal structure is critical → smaller patches better
3. Fewer parameters needed → lighter models work better  
4. Higher learning rates often work → faster convergence
"""

import os
import numpy as np


class Config_MAE_DEAP_Light():
    """Lightweight configuration - good starting point"""
    def __init__(self):
        super().__init__()
        
        # ============================================
        # Learning Rate (Higher for EEG)
        # ============================================
        self.lr = 2e-3              # Higher initial LR
        self.min_lr = 1e-5          # Non-zero minimum
        self.warmup_epochs = 10     # Shorter warmup
        self.num_epoch = 100        # Reasonable total
        
        # ============================================
        # Training Parameters
        # ============================================
        self.batch_size = 64        # Larger batch = more stable
        self.clip_grad = 1.0        
        self.weight_decay = 0.03    
        self.accum_iter = 1
        
        # ============================================
        # Model Parameters (LIGHT)
        # ============================================
        self.mask_ratio = 0.5       # Less aggressive masking for EEG
        self.patch_size = 16        # Good balance
        self.embed_dim = 512        # Lighter model
        self.decoder_embed_dim = 256
        self.depth = 12             # Much lighter (was 24!)
        self.decoder_depth = 4      
        self.num_heads = 8          # 512/64 = 8
        self.decoder_num_heads = 4
        self.mlp_ratio = 1.0
        
        # ============================================
        # DEAP Dataset Parameters
        # ============================================
        self.data_path = '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz'
        self.num_channels = 32      
        self.time_len = 8064        
        
        # Data augmentation
        self.aug_times = 1          
        self.sparse_rate = 0.1      
        
        # ============================================
        # Optional features (disabled)
        # ============================================
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6
        
        # ============================================
        # Paths
        # ============================================
        self.root_path = '/home/ab_students/EEG-MTP/'
        self.output_path = os.path.join(self.root_path, 'trial_mae_DEAP/results_light/')
        self.seed = 2024
        self.local_rank = 0


class Config_MAE_DEAP_Small():
    """Very small model - fastest training, good for debugging"""
    def __init__(self):
        super().__init__()
        
        # Learning parameters
        self.lr = 3e-3
        self.min_lr = 1e-5
        self.warmup_epochs = 5
        self.num_epoch = 80
        
        # Training
        self.batch_size = 64
        self.clip_grad = 1.0
        self.weight_decay = 0.01
        self.accum_iter = 1
        
        # Model (VERY LIGHT)
        self.mask_ratio = 0.4       # Even less masking
        self.patch_size = 32        # Larger patches = fewer tokens
        self.embed_dim = 256        
        self.decoder_embed_dim = 128
        self.depth = 8              
        self.decoder_depth = 4
        self.num_heads = 4
        self.decoder_num_heads = 2
        self.mlp_ratio = 1.0
        
        # DEAP Dataset
        self.data_path = '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz'
        self.num_channels = 32
        self.time_len = 8064
        self.aug_times = 1
        self.sparse_rate = 0.1
        
        # Optional
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6
        
        # Paths
        self.root_path = '/home/ab_students/EEG-MTP/'
        self.output_path = os.path.join(self.root_path, 'trial_mae_DEAP/results_small/')
        self.seed = 2024
        self.local_rank = 0


class Config_MAE_DEAP_Balanced():
    """Balanced configuration - recommended for production"""
    def __init__(self):
        super().__init__()
        
        # Learning parameters
        self.lr = 1.5e-3
        self.min_lr = 1e-5
        self.warmup_epochs = 10
        self.num_epoch = 120
        
        # Training
        self.batch_size = 64
        self.clip_grad = 1.0
        self.weight_decay = 0.03
        self.accum_iter = 1
        
        # Model (BALANCED)
        self.mask_ratio = 0.55      
        self.patch_size = 16        
        self.embed_dim = 384        
        self.decoder_embed_dim = 192
        self.depth = 10             
        self.decoder_depth = 4
        self.num_heads = 6
        self.decoder_num_heads = 3
        self.mlp_ratio = 1.0
        
        # DEAP Dataset
        self.data_path = '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz'
        self.num_channels = 32
        self.time_len = 8064
        self.aug_times = 1
        self.sparse_rate = 0.1
        
        # Optional
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6
        
        # Paths
        self.root_path = '/home/ab_students/EEG-MTP/'
        self.output_path = os.path.join(self.root_path, 'trial_mae_DEAP/results_balanced/')
        self.seed = 2024
        self.local_rank = 0


class Config_MAE_DEAP_FinePatch():
    """Fine-grained patches - better temporal resolution"""
    def __init__(self):
        super().__init__()
        
        # Learning parameters
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.warmup_epochs = 10
        self.num_epoch = 120
        
        # Training
        self.batch_size = 48        # Smaller batch for more patches
        self.clip_grad = 1.0
        self.weight_decay = 0.03
        self.accum_iter = 1
        
        # Model (FINE PATCHES)
        self.mask_ratio = 0.6       
        self.patch_size = 8         # Smaller patches = more detail
        self.embed_dim = 512        
        self.decoder_embed_dim = 256
        self.depth = 12             
        self.decoder_depth = 4
        self.num_heads = 8
        self.decoder_num_heads = 4
        self.mlp_ratio = 1.0
        
        # DEAP Dataset
        self.data_path = '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz'
        self.num_channels = 32
        self.time_len = 8064
        self.aug_times = 1
        self.sparse_rate = 0.1
        
        # Optional
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6
        
        # Paths
        self.root_path = '/home/ab_students/EEG-MTP/'
        self.output_path = os.path.join(self.root_path, 'trial_mae_DEAP/results_finepatch/')
        self.seed = 2024
        self.local_rank = 0
