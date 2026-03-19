import torch
import torch.nn as nn
import torch.nn.functional as F

class SEED_MAE(nn.Module):
    def __init__(self, embed_dim, additional_param=None):
        super(SEED_MAE, self).__init__()
        # ...existing code...
        
        # Add layer normalization for better stability
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Consider reducing mask ratio for better learning
        self.mask_ratio = 0.65  # Reduced from 0.75
        
        # Add gradient clipping value
        self.grad_clip_value = 1.0
        
    def forward(self, x):
        # ...existing code...
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # ...existing code...
        
        # Compute loss with better numerical stability
        loss = F.mse_loss(pred, target, reduction='mean')
        
        # Compute correlation with epsilon for stability
        eps = 1e-8
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_std = pred.std(dim=1, keepdim=True) + eps
        target_std = target.std(dim=1, keepdim=True) + eps
        
        correlation = ((pred - pred_mean) * (target - target_mean)).mean(dim=1) / (pred_std * target_std)
        
        return loss, pred, target, correlation.mean()
