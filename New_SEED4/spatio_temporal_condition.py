"""
Spatio-Temporal Condition Module with Graph Harmonic Innovation
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

# -------------------------------------------------------------
# Diffusion timestep embedding
# -------------------------------------------------------------
def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# -------------------------------------------------------------
# Graph Harmonic Spatial Embeddings
# -------------------------------------------------------------
def compute_graph_harmonics(chan_pos, k=32, n_neighbors=6, sigma=None):
    """
    Args:
        chan_pos : (C, 2) channel positions
        k        : number of eigenvectors to keep
        n_neighbors: kNN neighbors
        sigma    : RBF bandwidth; None = median heuristic
    Returns:
        basis: (C, k) normalized graph harmonic basis
    """
    if isinstance(chan_pos, torch.Tensor):
        pos = chan_pos.detach().cpu().numpy()
    else:
        pos = chan_pos.copy()
    if pos.ndim == 3:
        pos = pos[0]

    C = pos.shape[0]
    k = min(k, C - 2)  # need at least 2 spare

    # ── Weighted adjacency (RBF kernel on distances) ──────────────────────────
    n_nbrs = min(n_neighbors, C - 1)
    A_dist = kneighbors_graph(pos, n_nbrs, mode='distance',
                               include_self=False).toarray()

    # Symmetrise before computing sigma so both directions are counted
    A_dist = np.maximum(A_dist, A_dist.T)

    if sigma is None:
        nonzero = A_dist[A_dist > 0]
        sigma = np.median(nonzero) if len(nonzero) else 1.0

    A_weighted = np.where(A_dist > 0,
                          np.exp(-A_dist**2 / (2 * sigma**2)),
                          0.0).astype(np.float32)

    # ── Normalised Laplacian ──────────────────────────────────────────────────
    L = csgraph.laplacian(A_weighted, normed=True)

    # Small ridge to break degenerate eigenvalues (symmetric electrodes)
    L += 1e-4 * np.eye(C, dtype=np.float32)

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Drop DC component (eigval ≈ 0), keep next k
    basis = eigenvectors[:, 1:k + 1].astype(np.float32)   # (C, k)

    # L2-normalise each harmonic so scale is consistent across C
    norms = np.linalg.norm(basis, axis=0, keepdims=True) + 1e-8
    basis /= norms

    return torch.tensor(basis, dtype=torch.float32)
# -------------------------------------------------------------
# Simple X, Y Spatial Embedding (alternative, not used)
# -------------------------------------------------------------
def compute_xy_spatial_embedding(chan_pos, embed_dim=8):
    if isinstance(chan_pos, torch.Tensor):
        chan_pos = chan_pos.detach().cpu().numpy()
    if chan_pos.ndim == 3:
        chan_pos = chan_pos[0]
    half = embed_dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(0, half, dtype=np.float32) / half)
    x = chan_pos[:, 0:1]
    y = chan_pos[:, 1:2]
    x_enc = np.concatenate([np.sin(x * freqs), np.cos(x * freqs)], axis=-1)
    y_enc = np.concatenate([np.sin(y * freqs), np.cos(y * freqs)], axis=-1)
    return torch.tensor(x_enc + y_enc, dtype=torch.float32)


# -------------------------------------------------------------
# Fourier Spatial Embedding (alternative, not used)
# -------------------------------------------------------------
def compute_fourier_spatial_embedding(chan_pos, embed_dim=8):
    if isinstance(chan_pos, torch.Tensor):
        chan_pos = chan_pos.detach().cpu().numpy()
    if chan_pos.ndim == 3:
        chan_pos = chan_pos[0]
    half = embed_dim // 2
    np.random.seed(42)
    W = np.random.randn(2, half).astype(np.float32)
    proj = chan_pos @ W
    basis = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
    norms = np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8
    return torch.tensor(basis / norms, dtype=torch.float32)


# -------------------------------------------------------------
# 1D Convolution Block
# -------------------------------------------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatioTemporalConditionModule(nn.Module):
    def __init__(
        self,
        n_channels=64,
        seq_len=1000,
        embed_dim=256,
        n_harmonics=32,        # ← was 8, needs to be 32 for 62-ch
        patch_size=8,
        n_transformer_layers=4,
        n_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_harmonics = n_harmonics
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size

        # ── spatial embedding ──────────────────────────────────────────────
        # Keyed by C (number of channels) so LR/HR/SR each get correct harmonics.
        # Plain dict — NOT a register_buffer — because tensors are computed
        # lazily and may live on different devices.
        self._harmonic_cache: dict[int, torch.Tensor] = {}
        actual_harmonics = min(n_harmonics, n_channels - 2)  # mirrors the clip in compute_graph_harmonics
        self.actual_harmonics = actual_harmonics
        self.harmonic_proj = nn.Linear(actual_harmonics, embed_dim)

        # Temporal processing
        self.temporal_conv = TemporalConvBlock(n_channels, embed_dim, kernel_size=5)
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_channels, self.n_patches, embed_dim) * 0.02
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def _get_harmonics(self, chan_pos):
        pos = chan_pos[0] if chan_pos.ndim == 3 else chan_pos
        return compute_graph_harmonics(
            pos,
            k=self.actual_harmonics,
            n_neighbors=min(6, pos.shape[0] - 1)
        ).to(pos.device)

    def forward(self, x, chan_pos, t_steps):
        B, C, T = x.shape
        n_patches = T // self.patch_size
        usable_len = n_patches * self.patch_size

        if n_patches != self.n_patches:
            raise ValueError(
                f"Expected {self.n_patches} temporal patches from seq_len={self.seq_len}, "
                f"but got {n_patches} from T={T}."
            )
        if usable_len != T:
            x = x[:, :, :usable_len]

        # 1) Patch temporal embedding
        x_patches = x.reshape(B, C, n_patches, self.patch_size)
        temporal_tokens = self.patch_embed(x_patches)           # (B, C, N, embed_dim)

        # 2) Temporal conv context
        x_conv = self.temporal_conv(x)                          # (B, embed_dim, T)
        x_conv = x_conv.reshape(B, self.embed_dim, n_patches, self.patch_size).mean(dim=-1)
        x_conv = x_conv.permute(0, 2, 1).unsqueeze(1).expand(-1, C, -1, -1)

        temporal_feat = temporal_tokens + x_conv + self.pos_embed

        # 3) Graph harmonic spatial embedding  ← uses new per-C cache
        harmonics = self._get_harmonics(chan_pos)               # (C, n_harmonics)
        spatial_feat = self.harmonic_proj(harmonics)            # (C, embed_dim)
        spatial_feat = spatial_feat.unsqueeze(0).unsqueeze(2).expand(B, -1, n_patches, -1)

        # 4) Timestep embedding
        t_emb = self.time_mlp(sinusoidal_embedding(t_steps, self.embed_dim))
        t_emb = t_emb.unsqueeze(1).unsqueeze(2).expand(-1, C, n_patches, -1)

        # 5) Fuse + Transformer
        fused = temporal_feat + spatial_feat + t_emb
        fused_flat = fused.reshape(B, C * n_patches, self.embed_dim)
        cond_flat = self.transformer(fused_flat)
        cond_tokens = self.norm(cond_flat).reshape(B, C, n_patches, self.embed_dim).mean(dim=2)
        cond_pooled = cond_tokens.mean(dim=1)

        return cond_tokens, cond_pooled