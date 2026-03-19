# cond_diffusion_transformer.py
import torch
import torch.nn as nn
import math
import numpy as np

def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# -----------------------------------------------------
# Diffusion-Conditioned SpatioTemporal Transformer
# -----------------------------------------------------
class SpatioTemporalConditionedTransformer(nn.Module):
    def __init__(self, latent_dim=128, n_channels=32, n_layers=4, n_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        self.fourier_embed = nn.Linear(latent_dim, latent_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=n_heads,
                dim_feedforward=4 * latent_dim,
                dropout=0.1, batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)

    def forward(self, z_latent, cond_tokens, cond_pooled, t_steps):
        # === Spatio-Temporal Aware Diffusion (STAD-style) ===

        # Fourier temporal embedding (float32 FFT for numerical stability)
        with torch.amp.autocast('cuda', enabled=False):
            freqs = torch.fft.rfft(z_latent.float(), dim=1).real
            freqs = torch.log1p(freqs.abs())  # optional stability normalization

        freq_emb = self.fourier_embed(freqs)
        # interpolate from 32 -> 62 along sequence dimension
        freq_emb = torch.nn.functional.interpolate(
            freq_emb.transpose(1, 2),  # [B, D, F]
            size=z_latent.size(1),     # target length = 62
            mode="linear",
            align_corners=False
        ).transpose(1, 2)              # back to [B, T, D]


        # Diffusion time embedding
        t_emb = sinusoidal_embedding(t_steps, self.latent_dim).to(z_latent.device)
        t_emb = self.time_embed(t_emb).unsqueeze(1)

        # Fusion: temporal + spectral + diffusion embeddings
        x = z_latent + freq_emb + t_emb

        # Conditional fusion (cross-attention with conditioning tokens)
        attn_out, _ = self.cross_attn(x, cond_tokens, cond_tokens)
        x = x + attn_out + cond_pooled.unsqueeze(1)

        # Temporal transformer backbone
        for layer in self.layers:
            x = layer(x)

        return x