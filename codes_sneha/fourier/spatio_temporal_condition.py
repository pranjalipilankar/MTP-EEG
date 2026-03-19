import torch
import torch.nn as nn
import numpy as np

# -------------------------------------------------------------
# Helper: Fourier timestep embedding (diffusion-style)
# -------------------------------------------------------------
def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


# -------------------------------------------------------------
# Spatio-Temporal Conditioning Network (STC)
# -------------------------------------------------------------
class SpatioTemporalConditionNet(nn.Module):
    def __init__(self, n_channels=62, model_dim=128, n_conditions=8):
        super().__init__()

        # Temporal encoder across time (per channel)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(n_channels, model_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2)
        )

        # Spatial projection (2D electrode coords → model_dim)
        self.spatial_proj = nn.Linear(2, model_dim)

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim)
        )

        # Global pooling for pooled condition
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, chan_pos, t_steps, cond_c):
        B, C, T = x.shape

        # ---- Temporal encoding ----
        temporal_feat = self.temporal_cnn(x)        # (B, D, T)
        temporal_feat = temporal_feat.mean(dim=2)   # (B, D)
        temporal_feat = temporal_feat.unsqueeze(1).expand(B, C, -1)  # # (B, C, D)

        # ---- Spatial embedding ----
        spatial_feat = self.spatial_proj(chan_pos).unsqueeze(0).expand(B, -1, -1)  # (B, C, D)

        # ---- Time embedding ----
        t_sin = sinusoidal_embedding(t_steps, spatial_feat.shape[-1])  # (B, D)
        t_emb = self.time_embed(t_sin).unsqueeze(1).expand(B, C, -1)   # (B, C, D)

        # ---- Combine ----
        cond_tokens = temporal_feat + spatial_feat + t_emb

        # ---- Pooled condition ----
        cond_pooled = cond_tokens.mean(dim=1)  # (B, D)

        return cond_tokens, cond_pooled
