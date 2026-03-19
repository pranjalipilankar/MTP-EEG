"""
Multi-scale Transformer Denoising Module (MTD)
Based on STAD paper Section III.C.3

This module implements:
1. Multi-scale 1D convolutions for temporal features at different scales
2. Diffusion Transformer blocks with cross-attention for conditional guidance
3. Proper noise prediction for diffusion denoising
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sinusoidal_embedding(timesteps, dim):
    """Sinusoidal positional encoding for diffusion timesteps"""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# -------------------------------------------------------------
# Multi-scale 1D Convolution Block (STAD paper Eq. 3)
# -------------------------------------------------------------
class MultiScale1DConvBlock(nn.Module):
    """
    Extract temporal features at multiple scales using different kernel sizes.
    STAD paper: "multi-scale 1D convolution blocks are introduced to capture 
    multi-scale temporal features in the reverse denoising process"
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.n_scales = len(kernel_sizes)
        
        # Create conv layers for each scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels // self.n_scales,
                    kernel_size=k, padding=k//2
                ),
                nn.BatchNorm1d(out_channels // self.n_scales),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        
        # Final projection to match dimensions
        self.proj = nn.Conv1d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D) - latent representation
        Returns:
            (B, C, D) - multi-scale features
        """
        # Transpose for conv1d: (B, C, D) -> (B, D, C)
        x = x.transpose(1, 2)
        
        # Apply each scale
        multi_scale_feats = []
        for conv in self.convs:
            feat = conv(x)
            multi_scale_feats.append(feat)
        
        # Concatenate along channel dimension
        x = torch.cat(multi_scale_feats, dim=1)  # (B, D, C)
        
        # Project
        x = self.proj(x)
        
        # Transpose back: (B, D, C) -> (B, C, D)
        x = x.transpose(1, 2)
        
        return x

# -------------------------------------------------------------
# Diffusion Transformer Block (STAD paper + DiT)
# -------------------------------------------------------------
class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block with:
    1. Self-attention (MSA)
    2. Cross-attention with conditioning
    3. Feed-forward network
    
    Based on STAD paper Figure 2 and Section III.C.3
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention with conditioning (STAD paper Eq. 5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, cond_tokens):
        """
        Args:
            x: (B, L, D) - input tokens
            cond_tokens: (B, C, D) - conditioning tokens from STC module
        Returns:
            (B, L, D) - refined tokens
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Cross-attention with conditioning
        x_norm = self.norm2(x)
        cond_norm = self.norm2(cond_tokens)  # Normalize conditioning too
        cross_attn_out, _ = self.cross_attn(
            query=x_norm,
            key=cond_norm,
            value=cond_norm
        )
        x = x + cross_attn_out
        
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        
        return x

# -------------------------------------------------------------
# Multi-scale Transformer Denoising Module (MTD)
# -------------------------------------------------------------
class MultiScaleTransformerDenoisingModule(nn.Module):
    """
    Complete denoising module as described in STAD paper.
    
    Architecture:
    1. Input: noisy latent zt, timestep t, conditioning c
    2. Multi-scale convolutions for temporal features
    3. Diffusion Transformer blocks with cross-attention
    4. Output: predicted noise ε_θ(zt, t, c)
    """
    def __init__(
        self,
        latent_channels=256,    # HR EEG channels
        embed_dim=128,          # Model dimension
        n_layers=6,             # Number of Transformer layers
        n_heads=16,             # Number of attention heads (paper uses 16)
        conv_kernel_sizes=[3, 5, 7, 9],  # Multi-scale kernels
        dropout=0.1,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # === Input Processing ===
        # Project latent to embedding dimension
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # Positional encoding for channels (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, latent_channels, embed_dim) * 0.02
        )
        
        # === Timestep Embedding ===
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # === Multi-scale Convolutions (STAD paper Eq. 3) ===
        self.multiscale_conv = MultiScale1DConvBlock(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_sizes=conv_kernel_sizes
        )
        
        # === Diffusion Transformer Blocks ===
        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # === Output ===
        self.norm_out = nn.LayerNorm(embed_dim)
        self.noise_pred = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
    
    def forward(self, zt, t_steps, cond_tokens, cond_pooled):
        """
        Args:
            zt: (B, C, D) - noisy latent at timestep t
            t_steps: (B,) - diffusion timesteps
            cond_tokens: (B, C_lr, D) - conditioning tokens from STC
            cond_pooled: (B, D) - pooled conditioning
        
        Returns:
            noise_pred: (B, C, D) - predicted noise
        """
        B, C, D = zt.shape
        
        # === 1. Timestep Embedding ===
        t_emb = sinusoidal_embedding(t_steps, D)
        t_emb = self.time_mlp(t_emb)  # (B, D)
        t_emb = t_emb.unsqueeze(1)  # (B, 1, D)
        
        # === 2. Input Processing ===
        x = self.input_proj(zt)
        x = x + self.pos_embed  # Add positional encoding
        x = x + t_emb  # Add time embedding
        x = x + cond_pooled.unsqueeze(1)  # Add global conditioning
        
        # === 3. Multi-scale Convolutions (STAD paper) ===
        x = self.multiscale_conv(x)  # (B, C, D)
        
        # === 4. Diffusion Transformer Blocks ===
        # Each block applies self-attention + cross-attention with conditioning
        for block in self.transformer_blocks:
            x = block(x, cond_tokens)
        
        # === 5. Predict Noise ===
        x = self.norm_out(x)
        noise_pred = self.noise_pred(x)  # (B, C, D)
        
        return noise_pred

# -------------------------------------------------------------
# Lightweight version for faster experimentation
# -------------------------------------------------------------
class LightweightMTD(nn.Module):
    """Simplified version with fewer parameters"""
    def __init__(
        self,
        latent_channels=256,
        embed_dim=128,
        n_layers=3,  # Reduced
        n_heads=8,   # Reduced
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Simpler input processing
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, latent_channels, embed_dim) * 0.02)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Single-scale conv instead of multi-scale
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        # Fewer transformer layers
        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(embed_dim, n_heads, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm_out = nn.LayerNorm(embed_dim)
        self.noise_pred = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, zt, t_steps, cond_tokens, cond_pooled):
        B, C, D = zt.shape
        
        # Time embedding
        t_emb = sinusoidal_embedding(t_steps, D)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        
        # Input processing
        x = self.input_proj(zt) + self.pos_embed + t_emb + cond_pooled.unsqueeze(1)
        
        # Convolution (single scale)
        x_t = x.transpose(1, 2)
        x_t = self.conv(x_t)
        x = x_t.transpose(1, 2)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, cond_tokens)
        
        # Predict noise
        x = self.norm_out(x)
        noise_pred = self.noise_pred(x)
        
        return noise_pred
