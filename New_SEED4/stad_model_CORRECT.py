import torch
import torch.nn as nn
from spatio_temporal_condition import SpatioTemporalConditionModule
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule
from diffusion_scheduler import DDPMScheduler
import numpy as np
import warnings

class STADModel(nn.Module):
    """
    CORRECT STAD Implementation (Following LocalizeMI Architecture)
    
    Super-resolution happens through:
    1. SR (62ch) → project to 31ch → MAE encode → latents
    2. Add noise to latents (diffusion forward)
    3. MTD denoises using LR (16ch) conditioning ← SUPER-RESOLUTION HAPPENS HERE
    4. MAE decode → 31ch → upsample to 62ch
    """
    def __init__(
        self,
        mae_encoder,
        lr_channels=16,
        hr_channels=31,  # MAE was trained on this
        sr_channels=62,  # Target output
        latent_dim=768,
        num_patches=125,
        stc_embed_dim=256,
        mtd_layers=6,
        mtd_heads=16,
        diffusion_steps=1000,
        diffusion_schedule='cosine',
        lr_channel_indices=None,
        device='cuda'
    ):
        super().__init__()
        
        self.mae_encoder = mae_encoder
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.device = device
        self.diffusion_schedule = diffusion_schedule
        self.lr_channel_indices = lr_channel_indices
        
        # ✅ 1. STC: Extract conditioning from LR (16 channels)
        self.stc = SpatioTemporalConditionModule(
            n_channels=lr_channels,
            seq_len=1000,
            embed_dim=stc_embed_dim,
            n_harmonics=8,
            patch_size=8,
            n_transformer_layers=4,
            n_heads=8,
            dropout=0.1
        )
        
        # ✅ 2. MTD: Denoise SR latents using LR conditioning
        # THIS IS WHERE SUPER-RESOLUTION HAPPENS!
        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches,
            latent_dim=latent_dim,
            cond_dim=stc_embed_dim,
            n_layers=mtd_layers,
            n_heads=mtd_heads,
            dropout=0.1,
            use_multiscale_conv=True
        )
        
        # ✅ 3. Diffusion scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule=diffusion_schedule,
            beta_start=0.0001,
            beta_end=0.02
        )

        # Real SEED-IV electrode positions (downsampled to LR channel count to match dataset indices)
        lr_positions = self._create_seed4_positions(lr_channels, channel_indices=lr_channel_indices)
        self.register_buffer('lr_positions', torch.tensor(lr_positions, dtype=torch.float32))
        
        # ✅ 4. Channel projection: SR (62ch) → HR (31ch) for MAE encoding
        self.sr_to_hr_projection = nn.Conv1d(
            sr_channels, hr_channels, kernel_size=1, bias=False
        )
        
        # ✅ 5. Channel upsampling: HR (31ch) → SR (62ch) after MAE decoding
        # Following LocalizeMI implementation
        self.hr_to_sr_upsampler = nn.Sequential(
            nn.Identity(),
            nn.Conv1d(hr_channels, sr_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(sr_channels),
            nn.GELU(),
            nn.Conv1d(sr_channels, sr_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(sr_channels * 2),
            nn.GELU(),
            nn.Conv1d(sr_channels * 2, sr_channels, kernel_size=1),
        )
        
        print(f"\n✅ STAD Model (Correct Architecture):")
        print(f"   LR channels: {lr_channels} → STC → conditioning")
        print(f"   SR channels: {sr_channels} → project → {hr_channels} → MAE → latents")
        print(f"   MTD: Denoises SR latents using LR conditioning")
        print(f"   Diffusion schedule: {diffusion_schedule}")
        print(f"   Output: {hr_channels} → upsample → {sr_channels}")
        print(f"\n   STC params: {sum(p.numel() for p in self.stc.parameters()):,}")
        print(f"   MTD params: {sum(p.numel() for p in self.mtd.parameters()):,}")
        print(f"   MAE (frozen): {sum(p.numel() for p in self.mae_encoder.parameters() if not p.requires_grad):,}")
    
    def _create_standard_positions(self, n_channels):
        """Create standard 2D electrode positions for graph harmonics"""
        positions = []
        rows = int(np.ceil(np.sqrt(n_channels)))
        for i in range(n_channels):
            x = (i % rows) / (rows - 1) if rows > 1 else 0.5
            y = (i // rows) / (rows - 1) if rows > 1 else 0.5
            positions.append([x, y])
        return np.array(positions)

    def _create_seed4_positions(self, n_channels, channel_indices=None):
        """
        Build SEED-IV channel coordinates from MNE standard_1005 montage.
        Uses SEED-IV channel ordering and maps CB1/CB2 -> I1/I2.
        """
        seed4_mne_names = [
            'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'I1', 'O1', 'Oz',
            'O2', 'I2'
        ]

        try:
            import mne
            montage = mne.channels.make_standard_montage('standard_1005')
            ch_pos = montage.get_positions()['ch_pos']

            full_positions = []
            for name in seed4_mne_names:
                if name not in ch_pos:
                    raise KeyError(f"Channel {name} not found in standard_1005 montage")
                xyz = ch_pos[name]
                full_positions.append([xyz[0], xyz[1]])

            full_positions = np.array(full_positions, dtype=np.float32)
            scale = np.max(np.abs(full_positions))
            if scale > 0:
                full_positions = full_positions / scale

            if channel_indices is not None:
                channel_indices = np.asarray(channel_indices, dtype=int)
                if len(channel_indices) != n_channels:
                    raise ValueError(
                        f"Expected {n_channels} channel indices, got {len(channel_indices)}"
                    )
                return full_positions[channel_indices]

            if n_channels == len(full_positions):
                return full_positions

            # Match dataset channel downsampling strategy (linspace index selection).
            indices = np.linspace(0, len(full_positions) - 1, n_channels, dtype=int)
            return full_positions[indices]

        except Exception as exc:
            warnings.warn(
                f"Falling back to synthetic grid positions because real SEED-IV coordinates "
                f"could not be loaded: {exc}"
            )
            return self._create_standard_positions(n_channels)
    
    def encode_sr(self, sr_eeg):
        """
        Encode SR EEG to latent space.
        
        Args:
            sr_eeg: (B, 62, 1000) - Super-resolution EEG
        Returns:
            latents: (B, 125, 768) - Latent representations
        """
        # Project SR to HR space for MAE compatibility
        hr_projected = self.sr_to_hr_projection(sr_eeg)  # (B, 31, 1000)
        
        # Encode using pretrained MAE
        self.mae_encoder.eval()
        with torch.no_grad():
            latents, _, _ = self.mae_encoder.forward_encoder(hr_projected, mask_ratio=0.0)
            latents = latents[:, 1:, :]  # Remove CLS token → (B, 125, 768)
        
        return latents
    
    def decode_latent_to_sr(self, latents, lr_eeg=None):
        """
        Decode latents back to SR EEG (Following LocalizeMI architecture).
        
        Args:
            latents: (B, 125, 768) - Clean latents from diffusion
            lr_eeg: (B, 16, 1000) - For target length (optional)
        Returns:
            sr_eeg: (B, 62, 1000) - Super-resolution EEG
        """
        B = latents.size(0)
        target_len = lr_eeg.size(-1) if lr_eeg is not None else 1000
        
        # Add CLS token for MAE decoder
        cls_token = self.mae_encoder.cls_token.expand(B, -1, -1)
        latents_with_cls = torch.cat([cls_token, latents], dim=1)
        
        # Decode through MAE
        hr_patches = self.mae_encoder.forward_decoder(
            latents_with_cls,
            torch.zeros(B, self.num_patches, dtype=torch.long, device=latents.device)
        )  # (B, 125, C*patch_size)
        
        # Unpatchify to spatial domain
        hr_eeg = self.mae_encoder.unpatchify(hr_patches)  # (B, 31, 1000)
        
        # Resize if needed
        if hr_eeg.size(-1) != target_len:
            hr_eeg = torch.nn.functional.interpolate(
                hr_eeg, size=target_len, mode='linear', align_corners=False
            )
        
        # Upsample channels: 31 → 62
        sr_eeg = self.hr_to_sr_upsampler(hr_eeg)  # (B, 62, 1000)
        
        return sr_eeg
    
    def forward(self, lr_eeg, hr_eeg, sr_eeg):
        """
        Training forward pass.
        
        Args:
            lr_eeg: (B, 16, 1000) - Low-resolution input for conditioning
            hr_eeg: (B, 31, 1000) - Not used (kept for compatibility)
            sr_eeg: (B, 62, 1000) - Super-resolution TARGET
        
        Returns:
            loss: Diffusion denoising loss
            pred_sr: Predicted SR reconstruction (for monitoring)
        """
        B = lr_eeg.shape[0]
        
        # ============================================================
        # 1. Encode SR to latent space
        # ============================================================
        sr_latents = self.encode_sr(sr_eeg)  # (B, 125, 768)
        
        # ============================================================
        # 2. Get LR conditioning from STC
        # ============================================================
        chan_pos = self.lr_positions.unsqueeze(0).expand(B, -1, -1).to(self.device)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,),
            device=self.device
        ).long()
        
        # Extract conditioning
        cond_tokens, cond_pooled = self.stc(
            x=lr_eeg,
            chan_pos=chan_pos,
            t_steps=timesteps
        )
        
        # ============================================================
        # 3. Diffusion forward process: Add noise to SR latents
        # ============================================================
        noise = torch.randn_like(sr_latents)
        noisy_latents = self.scheduler.add_noise(sr_latents, noise, timesteps)
        
        # ============================================================
        # 4. MTD: Predict noise (THIS IS WHERE SUPER-RESOLUTION HAPPENS!)
        # ============================================================
        pred_noise = self.mtd(
            zt=noisy_latents,
            t_steps=timesteps,
            cond_tokens=cond_tokens,
            cond_pooled=cond_pooled
        )
        
        # ============================================================
        # 5. Compute diffusion loss
        # ============================================================
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        # ============================================================
        # 6. Reconstruct SR from denoised latents
        # ============================================================
        # Predict clean latent x0 from xt and predicted noise (DDPM parameterization).
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        alpha_bar_t = alphas_cumprod[timesteps].view(B, 1, 1)
        alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-4, max=1.0)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        pred_x0_latents = (
            noisy_latents - sqrt_one_minus_alpha_bar_t * pred_noise
        ) / (sqrt_alpha_bar_t + 1e-8)
        pred_x0_latents = torch.nan_to_num(pred_x0_latents, nan=0.0, posinf=10.0, neginf=-10.0)
        pred_x0_latents = torch.clamp(pred_x0_latents, min=-10.0, max=10.0)

        # Decode in FP32 to avoid AMP overflow in MAE decoder/upsampler path.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pred_sr = self.decode_latent_to_sr(pred_x0_latents.float(), lr_eeg.float())

        pred_sr = torch.nan_to_num(pred_sr, nan=0.0, posinf=1e3, neginf=-1e3)
        
        return loss, pred_sr
    
    @torch.no_grad()
    def sample_sr(self, lr_eeg, num_inference_steps=50):
        """
        Generate SR from LR using full DDPM/DDIM sampling.
        
        Args:
            lr_eeg: (B, 16, 1000)
            num_inference_steps: Number of denoising steps
        Returns:
            sr_eeg: (B, 62, 1000)
        """
        B = lr_eeg.shape[0]
        
        # Get conditioning
        chan_pos = self.lr_positions.unsqueeze(0).expand(B, -1, -1).to(self.device)
        timesteps_cond = torch.zeros(B, device=self.device).long()
        
        cond_tokens, cond_pooled = self.stc(lr_eeg, chan_pos, timesteps_cond)
        
        # Start from pure noise in latent space
        latents = torch.randn(B, self.num_patches, self.latent_dim, device=self.device)
        
        # Iterative denoising
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            t_batch = t.unsqueeze(0).repeat(B).to(self.device)
            
            # Predict noise
            pred_noise = self.mtd(
                zt=latents,
                t_steps=t_batch,
                cond_tokens=cond_tokens,
                cond_pooled=cond_pooled
            )
            
            # Denoise step
            latents = self.scheduler.step(pred_noise, t, latents)
        
        # Decode clean latents to SR
        sr_eeg = self.decode_latent_to_sr(latents, lr_eeg)
        
        return sr_eeg
