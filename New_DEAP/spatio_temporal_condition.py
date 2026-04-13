"""
Spatio-Temporal Condition Module with Graph Harmonic Innovation
Based on STAD paper Section III.C.2 + Your graph-based embedding innovation

This module extracts spatio-temporal features from LR EEG to guide the diffusion process.
YOUR INNOVATION: Using graph harmonic (Laplacian eigenvectors) for spatial embeddings
instead of simple position embeddings.
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
    """Sinusoidal positional encoding for diffusion timesteps"""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# -------------------------------------------------------------
# Graph Harmonic Spatial Embeddings (YOUR INNOVATION)
# -------------------------------------------------------------
def compute_graph_harmonics(chan_pos, k=8, n_neighbors=4):
    """
    Compute graph Laplacian eigenvectors for spatial embeddings.
    
    YOUR INNOVATION: Instead of standard spatial position embeddings,
    use spectral graph theory to capture topological structure of electrode layout.
    
    Args:
        chan_pos: (C, 2) or (B, C, 2) - channel positions
        k: number of eigenvectors (harmonics) to use
        n_neighbors: number of neighbors for graph construction
    
    Returns:
        basis: (C, k) - graph harmonic basis functions
    """
    if isinstance(chan_pos, torch.Tensor):
        chan_pos = chan_pos.detach().cpu().numpy()
    
    # Handle batch dimension
    if chan_pos.ndim == 3:
        chan_pos = chan_pos[0]  # Use first sample (positions are same across batch)
    
    C = chan_pos.shape[0]
    
    # Build k-NN graph from electrode positions
    A = kneighbors_graph(
        chan_pos, 
        n_neighbors=min(n_neighbors, C-1), 
        mode='connectivity', 
        include_self=False
    ).toarray()
    
    # Compute normalized graph Laplacian
    # L = D^(-1/2) * (D - A) * D^(-1/2)
    L = csgraph.laplacian(A, normed=True)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Take first k+1 eigenvectors (skip the constant first one)
    # These are the "harmonic" basis functions on the electrode graph
    max_k = max(1, C - 1)
    use_k = min(k, max_k)
    basis = eigenvectors[:, 1:use_k + 1]  # (C, use_k)

    # Keep output width fixed at requested k so downstream Linear(k -> embed_dim) is stable.
    if use_k < k:
        pad = np.zeros((C, k - use_k), dtype=basis.dtype)
        basis = np.concatenate([basis, pad], axis=1)
    elif use_k > k:
        basis = basis[:, :k]
    
    return torch.tensor(basis, dtype=torch.float32)

# -------------------------------------------------------------
# 1D Convolution Block for Temporal Feature Extraction
# -------------------------------------------------------------
class TemporalConvBlock(nn.Module):
    """1D convolution for temporal feature extraction (STAD paper)"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# -------------------------------------------------------------
# Spatio-Temporal Condition Module (YOUR VERSION WITH INNOVATION)
# -------------------------------------------------------------
class SpatioTemporalConditionModule(nn.Module):
    """
    Extracts spatio-temporal features from LR EEG to condition the diffusion process.
    
    Based on STAD paper Section III.C.2:
    - Temporal processing: 1D CNN + Transformer
    - Spatial processing: YOUR INNOVATION - Graph harmonic embeddings
    - Output: Conditioning tokens for cross-attention in diffusion model
    """
    def __init__(
        self,
        n_channels=64,        # LR EEG channels
        seq_len=350,          # Temporal length (350ms as per paper)
        embed_dim=256,        # Model dimension
        n_harmonics=8,        # Number of graph harmonics (YOUR PARAMETER)
        patch_size=16,        # Temporal patch size
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
        
        # === YOUR INNOVATION: Graph Harmonic Spatial Embeddings ===
        self.register_buffer("graph_harmonics", None)  # Will be computed on first forward
        self.harmonic_proj = nn.Linear(n_harmonics, embed_dim)
        
        # === Temporal Processing (STAD paper) ===
        # 1D CNN for initial temporal feature extraction
        self.temporal_conv = TemporalConvBlock(n_channels, embed_dim, kernel_size=5)
        
        # Patch embedding for temporal patches
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        
        # Positional encoding for temporal patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_channels, self.n_patches, embed_dim) * 0.02
        )
        
        # === Transformer for Spatio-Temporal Integration ===
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
        
        # === Timestep Embedding (Diffusion conditioning) ===
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def _compute_harmonics_if_needed(self, chan_pos):
        """Compute graph harmonics on first forward pass"""
        if self.graph_harmonics is None:
            harmonics = compute_graph_harmonics(
                chan_pos, 
                k=self.n_harmonics,
                n_neighbors=4
            ).to(chan_pos.device)
            self.graph_harmonics = harmonics
        return self.graph_harmonics
    
    def forward(self, x, chan_pos, t_steps):
        B, C, T = x.shape
        
        # 1. Temporal CNN
        x_conv = self.temporal_conv(x)  # [32, 128, 350]
        x_conv = x_conv.transpose(1, 2)  # [32, 350, 128]
        
        # 2. SIMPLE temporal features
        n_patches = T // self.patch_size  # 21
        temporal_feat = x_conv.mean(dim=1)[:, None, None, :]  # [32, 1, 1, 128]
        temporal_feat = temporal_feat.expand(-1, C, n_patches, -1)  # [32, 16, 21, 128]
        temporal_feat = temporal_feat + self.pos_embed
        
        # 3. Spatial (unchanged)
        harmonics = self._compute_harmonics_if_needed(chan_pos)
        spatial_feat = self.harmonic_proj(harmonics).unsqueeze(0).unsqueeze(2).expand(B, -1, n_patches, -1)
        
        # 4. Time embedding (unchanged)
        t_emb = self.time_mlp(sinusoidal_embedding(t_steps, self.embed_dim)).unsqueeze(1).unsqueeze(2).expand(-1, C, n_patches, -1)
        
        # 5. Fusion + Transformer (unchanged)
        fused_feat = temporal_feat + spatial_feat + t_emb
        fused_flat = fused_feat.reshape(B, C * n_patches, self.embed_dim)
        cond_flat = self.transformer(fused_flat)
        cond_tokens = self.norm(cond_flat).reshape(B, C, n_patches, self.embed_dim).mean(dim=2)
        cond_pooled = cond_tokens.mean(dim=1)
        
        return cond_tokens, cond_pooled



# -------------------------------------------------------------
# Lightweight version for faster training (optional)
# -------------------------------------------------------------
class LightweightSpatioTemporalConditionModule(nn.Module):
    """
    Simplified version with fewer parameters for faster experimentation.
    Still maintains graph harmonic innovation.
    """
    def __init__(
        self,
        n_channels=64,
        seq_len=350,
        embed_dim=128,
        n_harmonics=8,
        n_transformer_layers=2,  # Reduced from 4
        n_heads=4,               # Reduced from 8
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.n_harmonics = n_harmonics
        
        # Graph harmonics
        self.register_buffer("graph_harmonics", None)
        self.harmonic_proj = nn.Linear(n_harmonics, embed_dim)
        
        # Simple temporal CNN
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(n_channels, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        
        # Simplified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=2 * embed_dim,  # Reduced
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x, chan_pos, t_steps):
        """Simplified forward pass"""
        B, C, T = x.shape
        
        # Temporal features
        temporal_feat = self.temporal_cnn(x)  # (B, embed_dim, T)
        temporal_feat = temporal_feat.mean(dim=2)  # (B, embed_dim)
        temporal_feat = temporal_feat.unsqueeze(1).expand(-1, C, -1)  # (B, C, embed_dim)
        
        # Spatial features (graph harmonics)
        if self.graph_harmonics is None:
            harmonics = compute_graph_harmonics(chan_pos, k=self.n_harmonics)
            self.graph_harmonics = harmonics.to(x.device)
        
        spatial_feat = self.harmonic_proj(self.graph_harmonics)  # (C, embed_dim)
        spatial_feat = spatial_feat.unsqueeze(0).expand(B, -1, -1)  # (B, C, embed_dim)
        
        # Time embedding
        t_emb = sinusoidal_embedding(t_steps, self.embed_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1).expand(-1, C, -1)  # (B, C, embed_dim)
        
        # Fusion
        fused = temporal_feat + spatial_feat + t_emb
        
        # Transformer
        cond_tokens = self.transformer(fused)
        cond_pooled = cond_tokens.mean(dim=1)
        
        return cond_tokens, cond_pooled
