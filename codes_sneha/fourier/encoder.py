import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint_sequential

# -------------------------------------------------------------
# Fourier Time Embedding
# -------------------------------------------------------------
class FourierTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, max_freq=10.0, num_bands=6):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, num_bands)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(num_bands * 2, embed_dim)

    def forward(self, t_steps):
        """
        t_steps: int or tensor length T
        Returns: (T, D)
        """
        if isinstance(t_steps, int):
            t = torch.linspace(-1, 1, t_steps, device=self.freqs.device)
        else:
            t = t_steps.to(self.freqs.device)
        freqs = self.freqs[None, :] * math.pi * t[:, None]
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.proj(emb)  # (T, D)


# -------------------------------------------------------------
# Transformer-based EEG Encoder (STAD-style, memory efficient)
# -------------------------------------------------------------
class EEGEncoder(nn.Module):
    def __init__(
        self,
        in_channels=62,
        latent_dim=128,
        seq_len=400,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        pool_factor=4,         # temporal downsampling factor
        checkpoint_segments=4, # gradient checkpoint segments
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.pool_factor = pool_factor
        self.checkpoint_segments = checkpoint_segments

        # Project temporal signals to embedding dimension
        self.input_projection = nn.Linear(1, latent_dim)

        # Fourier positional/time embeddings
        self.time_embed = FourierTimeEmbedding(embed_dim=latent_dim)

        # Transformer for temporal encoding (per channel)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Channel-level aggregation
        self.channel_projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        """
        x: (B, C, T) - raw EEG
        Returns:
          z: (B, C, D) - latent representation per channel
          feats: (B, C, T_down, D) - temporal features per channel
        """
        B, C, T = x.shape
        x = x.unsqueeze(-1)  # (B, C, T, 1)
        x = self.input_projection(x)  # (B, C, T, D)

        # Add Fourier time embeddings
        time_emb = self.time_embed(T)  # (T, D)
        x = x + time_emb.unsqueeze(0).unsqueeze(0)  # broadcast -> (B, C, T, D)

        # Temporal downsampling (STAD-style pooling)
        if self.pool_factor > 1:
            B_, C_, T_, D_ = x.shape
            # reshape to (B*C, D, T) for avg_pool1d
            x = x.reshape(B_ * C_, D_, T_)                # (B*C, D, T)
            x = F.avg_pool1d(x, kernel_size=self.pool_factor, stride=self.pool_factor)  # (B*C, D, T_down)
            x = x.reshape(B_, C_, D_, -1).transpose(2, 3)  # (B, C, T_down, D)
            T = x.size(2)  # update new temporal length

        # Merge batch & channel for transformer processing
        x = x.view(B * C, T, self.latent_dim)  # (B*C, T, D)

        # Memory-efficient transformer (checkpointing)
        # Build module list from transformer layers (works for different torch versions)
        layers = getattr(self.temporal_transformer, "layers", None)
        if layers is not None:
            modules = list(layers)
        else:
            # fallback to children if .layers not present
            modules = list(self.temporal_transformer.children())

        # If checkpoint_segments <= 1 just run modules directly to avoid checkpoint overhead
        if self.checkpoint_segments is None or self.checkpoint_segments <= 1:
            out = x
            for m in modules:
                out = m(out)
            feats = out
        else:
            # IMPORTANT: pass use_reentrant=False so autograd can compute gradients reliably
            feats = checkpoint_sequential(modules, self.checkpoint_segments, x, use_reentrant=False)

        feats = feats.view(B, C, T, self.latent_dim)  # (B, C, T, D)

        # Mean-pool across time -> per-channel latent
        z = feats.mean(dim=2)  # (B, C, D)
        z = self.channel_projection(z)

        return z, feats
