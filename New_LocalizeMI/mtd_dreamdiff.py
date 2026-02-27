#!/usr/bin/env python3
"""
Multi-Scale Transformer Denoising Module (STAD Paper Section III.C.3)

Implements the denoising module from STAD paper with:
- Multi-scale 1D convolutions (Eq. 3) with kernels [3,5,7,9]
- Diffusion Transformer blocks with MSA (Eq. 4) and cross-attention (Eq. 5)
- Patch-based latent processing (B, num_patches, latent_dim)
"""
import torch
import torch.nn as nn
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

class MultiScaleTransformerDenoisingModule(nn.Module):
    """
    Multi-scale transformer for denoising patch-based latent representations (STAD Paper).
    
    Implements Section III.C.3:
    - Multi-scale 1D convolutions (Eq. 3): h_i = BN(Conv(z_t, k_i)), H_t = Concat(h_1,...,h_n)
    - MSA layer (Eq. 4): o_t = H_t + MSA(LN(H_t))
    - Cross-attention (Eq. 5): Q_t = W_Q·o_t, K_t = W_K·c, V_t = W_V·c
    - Feed-forward networks with residual connections
    
    Args:
        num_patches: Number of patches (e.g., 175 for 2800pts with patch_size=16)
        latent_dim: Latent dimension (e.g., 1024 for MAE)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_multiscale_conv: Enable multi-scale 1D convolutions (STAD paper Eq. 3)
    """
    def __init__(
        self,
        num_patches=100,      # ✅ Changed from hr_channels
        latent_dim=256,       # ✅ Increased from 128
        n_layers=6,
        n_heads=16,
        dropout=0.1,
        use_multiscale_conv=True,  # STAD paper feature
    ):
        super().__init__()
        self.num_patches = num_patches
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.use_multiscale_conv = use_multiscale_conv
        
        # ✅ Multi-scale 1D convolutions (STAD paper Section III.C.3, Eq. 3)
        if use_multiscale_conv:
            self.kernel_sizes = [3, 5, 7, 9]
            self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(latent_dim, latent_dim, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(latent_dim)
                )
                for k in self.kernel_sizes
            ])
            # Project concatenated features back to latent_dim
            self.conv_proj = nn.Linear(latent_dim * len(self.kernel_sizes), latent_dim)
        
        # ✅ Positional embedding for PATCHES (not channels)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, latent_dim) * 0.02
        )
        
        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        
        # ✅ Cross-attention with LR conditioning
        # LR conditioning comes from STC as (B, lr_channels, embed_dim)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Self-attention transformer for denoising
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norms
        self.norm_cross = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(n_layers)
        ])
        self.norm_self = nn.LayerNorm(latent_dim)
        
        # ✅ FIX: Conditioning projection (defined in __init__, not forward!)
        # This will project STC output to match MTD latent_dim if needed
        self.cond_proj = None  # Will be initialized if needed
        
        # Final projection (optional, can help with stability)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, zt, t_steps, cond_tokens, cond_pooled):
        """
        Denoise latent representations using multi-scale transformers.
        
        Args:
            zt: (B, num_patches, latent_dim) - Noisy patch latents
            t_steps: (B,) - Diffusion timesteps
            cond_tokens: (B, lr_channels, embed_dim) - Per-channel conditioning from STC
            cond_pooled: (B, embed_dim) - Global conditioning from STC
        
        Returns:
            pred_noise: (B, num_patches, latent_dim) - Predicted noise
        """
        B, N, D = zt.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {D}"
        
        # ============================================================
        # 1. Multi-Scale 1D Convolutions (STAD Paper Eq. 3)
        # ============================================================
        if self.use_multiscale_conv:
            # Transpose for Conv1d: (B, latent_dim, num_patches)
            zt_conv = zt.transpose(1, 2)  # (B, D, N)
            
            # Apply convolutions with different kernel sizes
            # h_i = BN(Conv(zt, k_i)) from Eq. 3
            conv_outputs = []
            for conv_layer in self.conv_layers:
                h_i = conv_layer(zt_conv)  # (B, D, N)
                conv_outputs.append(h_i)
            
            # Concatenate along feature dimension: Concat(h_1, ..., h_n)
            h_concat = torch.cat(conv_outputs, dim=1)  # (B, D*4, N)
            h_concat = h_concat.transpose(1, 2)  # (B, N, D*4)
            
            # Project back to latent_dim
            x = self.conv_proj(h_concat)  # (B, N, D)
        else:
            x = zt
        
        # ============================================================
        # 2. Add Positional Encoding
        # ============================================================
        x = x + self.pos_embed  # (B, num_patches, latent_dim)
        
        # ============================================================
        # 3. Add Timestep Embedding
        # ============================================================
        t_emb = sinusoidal_embedding(t_steps, self.latent_dim)  # (B, latent_dim)
        t_emb = self.time_mlp(t_emb)  # (B, latent_dim)
        
        # Expand to all patches
        t_emb = t_emb.unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, num_patches, latent_dim)
        x = x + t_emb
        
        # ============================================================
        # 4. Cross-Attention with LR Conditioning (STAD Eq. 5)
        # ============================================================
        # cond_tokens: (B, lr_channels, embed_dim)
        # We need to match dimensions if embed_dim != latent_dim
        
        # If dimensions don't match, project conditioning
        if cond_tokens.shape[-1] != self.latent_dim:
            # Add a projection layer (define in __init__ if needed)
            if not hasattr(self, 'cond_proj'):
                self.cond_proj = nn.Linear(cond_tokens.shape[-1], self.latent_dim).to(cond_tokens.device)
            cond_tokens = self.cond_proj(cond_tokens)
        
        # Apply cross-attention layers
        for i, (cross_attn, norm) in enumerate(zip(self.cross_attn_layers, self.norm_cross)):
            # Cross-attention: query=latent patches, key/value=LR conditioning
            x_attended, _ = cross_attn(
                query=x,                # (B, num_patches, latent_dim)
                key=cond_tokens,        # (B, lr_channels, latent_dim)
                value=cond_tokens
            )
            x = norm(x + x_attended)  # Residual connection + norm
        
        # ============================================================
        # 5. Self-Attention Denoising (MSA from Eq. 4)
        # ============================================================
        x = self.transformer(x)  # (B, num_patches, latent_dim)
        x = self.norm_self(x)
        
        # ============================================================
        # 6. Output Projection
        # ============================================================
        pred_noise = self.output_proj(x)  # (B, num_patches, latent_dim)
        
        return pred_noise


# ============================================================
# Backward Compatibility: Wrapper for Old Interface
# ============================================================
class MultiScaleTransformerDenoisingModule_Legacy(nn.Module):
    """
    Legacy wrapper that converts channel-based interface to patch-based.
    Use this if you want to keep the old calling convention.
    """
    def __init__(self, hr_channels=32, latent_dim=256, n_layers=6, n_heads=16):
        super().__init__()
        # Calculate equivalent number of patches
        # Assuming 400 timepoints, patch_size=4 → 100 patches
        num_patches = 100
        
        self.core = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches,
            latent_dim=latent_dim,
            n_layers=n_layers,
            n_heads=n_heads
        )
        
        self.hr_channels = hr_channels
        self.num_patches = num_patches
    
    def forward(self, zt, t_steps, cond_tokens, cond_pooled):
        """
        Accepts both (B, num_patches, D) and (B, channels, D) formats.
        Automatically detects and converts.
        """
        B, N, D = zt.shape
        
        if N == self.hr_channels:
            # Old format: (B, 32, 256) - expand to patches
            # Simple strategy: repeat each channel to fill patches
            patches_per_channel = self.num_patches // self.hr_channels
            zt_patches = zt.unsqueeze(2).repeat(1, 1, patches_per_channel, 1)
            zt_patches = zt_patches.reshape(B, -1, D)
            
            # Pad if needed
            if zt_patches.shape[1] < self.num_patches:
                padding = self.num_patches - zt_patches.shape[1]
                zt_pad = zt_patches[:, -1:, :].repeat(1, padding, 1)
                zt_patches = torch.cat([zt_patches, zt_pad], dim=1)
        else:
            # New format: (B, num_patches, 256)
            zt_patches = zt
        
        # Call core module
        pred_noise_patches = self.core(zt_patches, t_steps, cond_tokens, cond_pooled)
        
        # Convert back if input was channel-based
        if N == self.hr_channels:
            # Average patches back to channels
            pred_noise = pred_noise_patches.reshape(B, self.hr_channels, -1, D).mean(dim=2)
        else:
            pred_noise = pred_noise_patches
        
        return pred_noise