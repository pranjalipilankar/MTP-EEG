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

def fourier_positional_encoding(coords, num_freqs=8):
    """
    coords: (C, 2) tensor of normalized x,y coordinates
    Returns: (C, 2 * num_freqs * 2) positional encoding
    """
    C, _ = coords.shape
    device = coords.device

    freq_bands = 2 ** torch.linspace(0, num_freqs-1, num_freqs, device=device)  # (num_freqs,)
    # coords: (C, 2) → (C, 2, num_freqs)
    angles = coords.unsqueeze(-1) * freq_bands  # multiply each coord by each freq
    # apply sin & cos
    sin_enc = torch.sin(angles)
    cos_enc = torch.cos(angles)

    # final shape (C, 2 * num_freqs * 2)
    enc = torch.cat([sin_enc, cos_enc], dim=-1).reshape(C, -1)
    return enc

# -------------------------------------------------------------
# Spatio-Temporal Conditioning Network (FINAL)
# -------------------------------------------------------------
class SpatioTemporalConditionNet(nn.Module):
    def __init__(self, n_channels=62, model_dim=128):
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

    def forward(self, x, chan_pos, t_steps, cond_c=None):
        B, C, T = x.shape

        # Temporal features (B, C, model_dim)
        temporal_feat = self.temporal_cnn(x).mean(dim=2).unsqueeze(1).expand(B, C, -1)

        # Fourier spatial embedding
        fourier_embed = fourier_positional_encoding(chan_pos, num_freqs=self.num_freqs) # (C, Df)
        spatial_feat = self.fourier_proj(fourier_embed).unsqueeze(0).expand(B, C, -1)

        # Diffusion time embedding
        t_sin = sinusoidal_embedding(t_steps, spatial_feat.shape[-1])     # (B, model_dim)
        t_emb = self.time_embed(t_sin).unsqueeze(1).expand(B, C, -1)

        cond_tokens = temporal_feat + spatial_feat + t_emb
        cond_pooled = cond_tokens.mean(dim=1)
        return cond_tokens, cond_pooled
