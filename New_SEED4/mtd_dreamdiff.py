#!/usr/bin/env python3
"""
Multi-Scale Transformer Denoising Module
"""
import torch
import torch.nn as nn
import numpy as np


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half, dtype=torch.float32,
                                       device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MultiScaleTransformerDenoisingModule(nn.Module):
    """
    Args:
        num_patches : number of MAE latent patches (e.g. 125)
        latent_dim  : MAE latent dimension (e.g. 768)
        cond_dim    : STC embed_dim (e.g. 256)
        n_layers    : transformer depth
        n_heads     : attention heads
    """
    def __init__(
        self,
        num_patches=125,
        latent_dim=768,
        cond_dim=256,
        n_layers=6,
        n_heads=16,
        dropout=0.1,
        use_multiscale_conv=True,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.latent_dim  = latent_dim
        self.cond_dim    = cond_dim

        # ── Multi-scale 1-D convolutions ──────────────────────────────────
        if use_multiscale_conv:
            self.kernel_sizes = [3, 5, 7, 9]
            self.conv_layers  = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(latent_dim, latent_dim, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(latent_dim),
                )
                for k in self.kernel_sizes
            ])
            self.conv_proj = nn.Linear(latent_dim * len(self.kernel_sizes), latent_dim)
        self.use_multiscale_conv = use_multiscale_conv

        # ── Positional embedding ──────────────────────────────────────────
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, latent_dim) * 0.02
        )

        # ── Timestep MLP ──────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

        # ── Global conditioning gate (uses cond_pooled) ───────────────────
        # Projects STC global token → scale+shift for the timestep embedding.
        self.global_cond_proj = nn.Sequential(
            nn.Linear(cond_dim, latent_dim * 2),   # → (scale, shift)
            nn.SiLU(),
        )

        # ── Conditioning projection (cond_tokens dim → latent_dim) ────────
        self.cond_proj = (nn.Linear(cond_dim, latent_dim)
                          if cond_dim != latent_dim else nn.Identity())

        # ── Cross-attention: latent patches attend to LR channel tokens ───
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm_cross = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(n_layers)
        ])

        # ── Self-attention transformer ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_self    = nn.LayerNorm(latent_dim)

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    # ------------------------------------------------------------------
    def forward(self, zt, t_steps, cond_tokens, cond_pooled):
        """
        Args:
            zt          : (B, num_patches, latent_dim)  noisy latents
            t_steps     : (B,)
            cond_tokens : (B, C, cond_dim)              per-channel LR tokens
            cond_pooled : (B, cond_dim)                 global LR token
        Returns:
            pred_noise  : (B, num_patches, latent_dim)
        """
        B, N, D = zt.shape

        # 1) Multi-scale convolutions
        if self.use_multiscale_conv:
            zt_t   = zt.transpose(1, 2)                             # (B, D, N)
            outs   = [conv(zt_t) for conv in self.conv_layers]      # each (B, D, N)
            x = self.conv_proj(
                torch.cat(outs, dim=1).transpose(1, 2)              # (B, N, D*4)
            )                                                        # (B, N, D)
        else:
            x = zt

        # 2) Positional encoding
        x = x + self.pos_embed

        # 3) Timestep embedding modulated by global LR conditioning
        t_emb   = self.time_mlp(sinusoidal_embedding(t_steps, self.latent_dim))  # (B, D)
        gs      = self.global_cond_proj(cond_pooled)                              # (B, D*2)
        scale, shift = gs.chunk(2, dim=-1)                                        # each (B, D)
        t_emb   = t_emb * (1 + scale) + shift                                    # FiLM modulation

        x = x + t_emb.unsqueeze(1).expand(-1, N, -1)

        # 4) Project cond_tokens to latent_dim
        kv = self.cond_proj(cond_tokens)                            # (B, C, D)

        # 5) Cross-attention: latent patches ← LR channel tokens
        for cross_attn, norm in zip(self.cross_attn_layers, self.norm_cross):
            attended, _ = cross_attn(query=x, key=kv, value=kv)
            x = norm(x + attended)

        # 6) Self-attention denoising
        x = self.transformer(x)
        x = self.norm_self(x)

        # 7) Predict noise
        return self.output_proj(x)