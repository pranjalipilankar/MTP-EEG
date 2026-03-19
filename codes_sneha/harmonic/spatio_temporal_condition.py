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


# -------------------------------------------------------------
# Graph Harmonic Positional Embedding
# -------------------------------------------------------------
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def compute_graph_harmonics(chan_pos, k=8):
    if isinstance(chan_pos, torch.Tensor):
        chan_pos = chan_pos.cpu().numpy()

    A = kneighbors_graph(chan_pos, n_neighbors=4, mode='connectivity', include_self=False).toarray()
    L = csgraph.laplacian(A, normed=True)
    vals, vecs = np.linalg.eigh(L)

    # Skip constant eigenvector (index 0)
    basis = vecs[:, 1:k+1]
    return torch.tensor(basis, dtype=torch.float32)


# -------------------------------------------------------------
# Spatio-Temporal Conditioning Network (FINAL)
# -------------------------------------------------------------
class SpatioTemporalConditionNet(nn.Module):
    def __init__(self, n_channels=62, model_dim=128, n_conditions=8):
        super().__init__()

        # Temporal encoder (per channel)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(n_channels, model_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2)
        )

        # Harmonic spatial projection
        self.harm_proj = nn.Linear(n_conditions, model_dim)
        self.register_buffer("harmonics", None)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim)
        )

    def forward(self, x, chan_pos, t_steps, cond_c):
        """
        x: (B, C, T)
        chan_pos: (C, 2)
        cond_c is currently not used, but kept for future class-conditioning
        """
        B, C, T = x.shape

        # ---- Temporal encoding ----
        temporal_feat = self.temporal_cnn(x).mean(dim=2).unsqueeze(1).expand(B, C, -1)

        # ---- Harmonic spatial embedding ----
        if self.harmonics is None or self.harmonics.shape[0] != C:
            self.harmonics = compute_graph_harmonics(chan_pos, k=self.harm_proj.in_features).to(x.device)

        spatial_feat = self.harm_proj(self.harmonics).unsqueeze(0).expand(B, -1, -1)

        # ---- Time embedding ----
        t_sin = sinusoidal_embedding(t_steps, spatial_feat.shape[-1])
        t_emb = self.time_embed(t_sin).unsqueeze(1).expand(B, C, -1)

        # ---- Combine ----
        cond_tokens = temporal_feat + spatial_feat + t_emb
        cond_pooled = cond_tokens.mean(dim=1)

        return cond_tokens, cond_pooled
