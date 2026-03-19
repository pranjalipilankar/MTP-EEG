import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

# -------------------------------------------------------------
# Fourier Time Embedding
# -------------------------------------------------------------
class FourierTimeEmbedding(nn.Module):
    """Fourier positional embedding for temporal decoding"""
    def __init__(self, embed_dim, max_freq=10.0, num_bands=6):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, num_bands)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(num_bands * 2, embed_dim)

    def forward(self, T):
        t = torch.linspace(-1, 1, T, device=self.freqs.device)
        freqs = self.freqs[None, :] * math.pi * t[:, None]
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.proj(emb)

# -------------------------------------------------------------
# EEGDecoder — with downsampled decoding + activation checkpointing
# -------------------------------------------------------------
class EEGDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        out_channels=32,
        seq_len=400,
        pool_factor=4,
        seq_len_down=None,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        num_bands=6,
        max_freq=10.0,
        checkpoint_segments=2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.pool_factor = pool_factor
        self.seq_len_down = seq_len_down or (seq_len // pool_factor)
        self.checkpoint_segments = checkpoint_segments

        # Learned temporal queries at downsampled resolution
        self.temporal_queries = nn.Parameter(torch.randn(1, self.seq_len_down, latent_dim))

        # Fourier time embedding
        self.time_embed = FourierTimeEmbedding(embed_dim=latent_dim, max_freq=max_freq, num_bands=num_bands)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.layers = nn.ModuleList([decoder_layer for _ in range(n_layers)])
        self.norm = nn.LayerNorm(latent_dim)

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Final projection
        self.out_proj = nn.Linear(latent_dim, 1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


        # Optional timestep conditioning
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z, t_embed=None):
        """
        Args:
            z: (B, C, D)
            t_embed: (B, D), optional timestep embedding for diffusion
        Returns:
            x_hat: (B, C, T_full)
        """
        B, C, D = z.shape
        T_full = self.seq_len
        T_down = self.seq_len_down

        # Prepare temporal queries
        queries = self.temporal_queries.expand(B * C, T_down, D)
        time_emb = self.time_embed(T_down).unsqueeze(0).expand(B * C, -1, -1)
        queries = queries + time_emb

        if t_embed is not None:
            t_cond = t_embed.unsqueeze(1).expand(B, 1, D).repeat(1, C, 1).reshape(B * C, 1, D)
            t_cond = t_cond.expand(-1, T_down, -1)
            queries = queries + self.gamma * t_cond

        memory = z.reshape(B * C, 1, D)

        # ---------------------------------------------------------
        # ✅ Correct checkpointed decoding
        # ---------------------------------------------------------
        def run_layers(layers, x):
            for layer in layers:
                x = layer(x, memory)
            return x

        if self.checkpoint_segments > 1:
            segment_size = len(self.layers) // self.checkpoint_segments
            out_seq = queries
            for i in range(self.checkpoint_segments):
                start = i * segment_size
                end = len(self.layers) if i == self.checkpoint_segments - 1 else (start + segment_size)
                # capture subset of layers
                segment_layers = self.layers[start:end]
                # use_reentrant=False ensures stable gradient flow
                out_seq = checkpoint.checkpoint(
                    run_layers, segment_layers, out_seq, use_reentrant=False
                )
        else:
            out_seq = run_layers(self.layers, queries)
        # ---------------------------------------------------------

        out_seq = self.norm(out_seq)
        out_seq = self.fusion(out_seq)

        # Project to amplitude
        x_down = self.out_proj(out_seq).squeeze(-1)  # (B*C, T_down)
        x_down = x_down.unsqueeze(1)
        x_full = F.interpolate(x_down, size=T_full, mode="linear", align_corners=False)
        x_full = x_full.squeeze(1).view(B, C, T_full)
        return x_full
