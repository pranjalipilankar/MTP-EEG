import torch
import torch.nn as nn
import numpy as np

# -------------------------------------------------------------
# Fourier timestep embedding (diffusion)
# -------------------------------------------------------------
def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

def fourier_positional_encoding(coords, num_freqs=4):
    """
    coords: (B, C, 2) → output (B, C, Df)
    """
    if coords.dim() == 2:  # (C, 2)
        coords = coords.unsqueeze(0)  # add batch dim
    
    B, C, _ = coords.shape
    freq_bands = 2 ** torch.arange(num_freqs, device=coords.device).float()  # (num_freqs,)
    coords_expanded = coords.unsqueeze(3) * freq_bands[None, None, None, :]  # (B, C, 2, num_freqs)

    sin_part = torch.sin(2 * np.pi * coords_expanded)
    cos_part = torch.cos(2 * np.pi * coords_expanded)
    embed = torch.cat([sin_part, cos_part], dim=2)  # (B, C, 4, num_freqs)
    embed = embed.reshape(B, C, -1)  # flatten → (B, C, Df)
    return embed

# -------------------------------------------------------------
# Spatio-Temporal Conditioning Network (FINAL)
# -------------------------------------------------------------
class FourierPositionConditionNet(nn.Module):
    def __init__(self, n_channels=32, model_dim=128):
        super().__init__()

        # Temporal encoder
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(n_channels, model_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2)
        )

        # Fourier spatial embedding
        self.num_freqs = 8
        self.fourier_proj = nn.Linear(self.num_freqs * 2 * 2, model_dim)

        # Time embedding (diffusion)
        self.time_embed = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim)
        )

    def forward(self, x, chan_pos, t_steps, cond_c):
        B, C, T = x.shape

        # ---- Temporal encoding ----
        temporal_feat = self.temporal_cnn(x)
        temporal_feat = temporal_feat.transpose(1, 2)
        temporal_feat = temporal_feat.mean(dim=1, keepdim=True).expand(B, C, -1)

        # ---- Fourier spatial encoding ----
        fourier_embed = fourier_positional_encoding(chan_pos, num_freqs=self.num_freqs)  # (B, C, Df)
        spatial_feat = self.fourier_proj(fourier_embed)  # ✅ (B, C, model_dim)

        # ---- Time embedding ----
        t_sin = sinusoidal_embedding(t_steps, spatial_feat.shape[-1])
        t_emb = self.time_embed(t_sin).unsqueeze(1).expand(B, C, -1)

        # ---- Combine ----
        cond_tokens = temporal_feat + spatial_feat + t_emb
        cond_pooled = cond_tokens.mean(dim=1)

        return cond_tokens, cond_pooled

