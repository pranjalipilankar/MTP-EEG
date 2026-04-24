#!/usr/bin/env python3
"""Evaluate STAD checkpoint on DEAP dataset."""

import argparse
import json
import os
import sys
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/ab_students/EEG-MTP/trial_mae_DEAP')
from mae_for_eeg import MAEforEEG
from spatio_temporal_condition import SpatioTemporalConditionModule
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule


def get_deap_channel_indices(target_channels, total_channels=32):
    """Fixed channel subsets used in DEAP training."""
    if target_channels == 32:
        return np.arange(32, dtype=int)
    if target_channels == 16:
        # Even-channel protocol: 0,2,...,30
        return np.arange(0, 32, 2, dtype=int)
    if target_channels == 8:
        # Every 4th channel: 0,4,...,28
        return np.arange(0, 32, 4, dtype=int)
    return np.linspace(0, total_channels - 1, target_channels, dtype=int)


def compute_sr_metrics(pred_sr, target_sr, eps=1e-8):
    """Compute PCC, NMSE, and SNR in dB."""
    pred = pred_sr.reshape(pred_sr.shape[0], -1)
    target = target_sr.reshape(target_sr.shape[0], -1)

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered.pow(2).sum(dim=1) + eps) *
        (target_centered.pow(2).sum(dim=1) + eps)
    )
    pcc = (numerator / denominator).mean().item()

    mse = (pred - target).pow(2).mean(dim=1)
    signal_power = target.pow(2).mean(dim=1)
    nmse = (mse / (signal_power + eps)).mean().item()
    snr = (10.0 * torch.log10((signal_power + eps) / (mse + eps))).mean().item()

    return {
        'pcc': pcc,
        'nmse': nmse,
        'snr': snr,
    }


def _infer_mae_kwargs_from_checkpoint(checkpoint):
    """Infer MAE constructor kwargs from a MAE or STAD checkpoint."""
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    # If this is a full STAD checkpoint, the MAE weights are stored with a 'mae.' prefix.
    if any(key.startswith('mae.') for key in state.keys()):
        state = {
            key[len('mae.'):]: value
            for key, value in state.items()
            if key.startswith('mae.')
        }

    config = dict(checkpoint.get('config', {}) or {})

    patch_weight = state.get('patch_embed.proj.weight')
    decoder_embed_weight = state.get('decoder_embed.weight')
    decoder_pred_weight = state.get('decoder_pred.weight')
    qkv_weight = state.get('blocks.0.attn.qkv.weight')

    if patch_weight is not None:
        config.setdefault('embed_dim', int(patch_weight.shape[0]))
        config.setdefault('in_chans', int(patch_weight.shape[1]))
        config.setdefault('patch_size', int(patch_weight.shape[2]))

    if decoder_embed_weight is not None:
        config.setdefault('decoder_embed_dim', int(decoder_embed_weight.shape[0]))

    if decoder_pred_weight is not None and 'patch_size' in config and 'in_chans' not in config:
        out_dim = int(decoder_pred_weight.shape[0])
        patch_size = int(config['patch_size'])
        if out_dim % patch_size == 0:
            config.setdefault('in_chans', out_dim // patch_size)

    if qkv_weight is not None and 'embed_dim' in config:
        embed_dim = int(config['embed_dim'])
        num_heads = config.get('num_heads')
        if num_heads is None:
            preferred = [12, 16, 8, 6, 4, 3, 2]
            config['num_heads'] = next((h for h in preferred if h != 0 and embed_dim % h == 0), 1)

    config.setdefault('time_len', 1000)
    config.setdefault('patch_size', 8)
    config.setdefault('embed_dim', 768)
    config.setdefault('in_chans', 16)
    config.setdefault('depth', 12)
    config.setdefault('num_heads', 12)
    config.setdefault('decoder_embed_dim', 384)
    config.setdefault('decoder_depth', 4)
    config.setdefault('decoder_num_heads', 8)
    config.setdefault('mlp_ratio', 4.0)
    config.setdefault('norm_pix_loss', True)

    return config


def _infer_fold_from_path(path_text):
    match = re.search(r'fold[_-](\d+)', str(path_text))
    return int(match.group(1)) if match else None


def _resolve_fold_subject_split(results_dir, preferred_fold=None):
    """Resolve train/val subject lists from fold_splits.json for a selected fold."""
    splits_path = os.path.join(results_dir, 'fold_splits.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"fold_splits.json not found in {results_dir}")

    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    if not splits:
        raise ValueError(f"No fold splits available in {splits_path}")

    if preferred_fold is None:
        preferred_fold = max(splits, key=lambda x: x.get('val_size', 0)).get('fold')

    selected = None
    for row in splits:
        if int(row['fold']) == int(preferred_fold):
            selected = row
            break
    if selected is None:
        raise ValueError(f"Fold {preferred_fold} not found in {splits_path}")

    train_subjects = [str(s) for s in selected.get('train_subjects', [])]
    val_subjects = [str(s) for s in selected.get('val_subjects', [])]
    overlap = sorted(set(train_subjects).intersection(set(val_subjects)))
    if overlap:
        raise ValueError(f"Data leakage detected in fold {preferred_fold}: shared subjects {overlap}")

    return int(preferred_fold), train_subjects, val_subjects


def get_channel_positions(n_channels, device='cpu', batch_size=1):
    """Get channel positions for spatial conditioning."""
    if n_channels == 32:
        positions = np.array([
            [-0.35, 0.93], [-0.71, 0.71], [-0.53, 0.84], [-0.88, 0.47],
            [-1.00, 0.00], [-0.88,-0.47], [-0.53,-0.84], [ 0.71,-0.71],
            [ 0.88, 0.47], [ 0.71, 0.71], [ 0.53, 0.84], [ 0.35, 0.93],
            [ 0.71,-0.71], [ 0.00, 0.00], [ 0.35,-0.93], [-0.35,-0.93],
            [ 0.00, 1.00], [-0.25, 0.25], [ 0.25, 0.25], [ 0.00,-1.00],
            [-0.50,-0.25], [ 0.50,-0.25], [-0.25,-0.50], [ 0.25,-0.50],
            [-0.50, 0.25], [ 0.50, 0.25], [-0.25, 0.50], [ 0.25, 0.50],
            [ 0.00, 0.50], [-0.75, 0.00], [ 0.75, 0.00], [ 0.00, 0.00]
        ], dtype=np.float32)
    elif n_channels == 16:
        positions = get_channel_positions(32, 'cpu', 1).squeeze(0).numpy()[::2]
    elif n_channels == 8:
        positions = get_channel_positions(32, 'cpu', 1).squeeze(0).numpy()[::4]
    else:
        raise ValueError(f"Unsupported n_channels={n_channels}")
    return torch.tensor(positions, device=device).unsqueeze(0).expand(batch_size, -1, -1)


class DEAPTestDataset(Dataset):
    """Loader for DEAP test data from NPZ format or preprocessed folder structure."""

    def __init__(self, data_path, test_indices=None, lr_channels=8, hr_channels=16, subject_ids=None):
        self.lr_indices = get_deap_channel_indices(lr_channels, 32)
        self.hr_indices = get_deap_channel_indices(hr_channels, 32)
        self.allowed_subjects = None if subject_ids is None else {str(s) for s in subject_ids}

        data_path = Path(data_path)

        if data_path.is_file() and data_path.suffix == '.npz':
            self._load_from_npz(data_path, test_indices)
        elif data_path.is_dir():
            self._load_from_folder(data_path)
        else:
            raise FileNotFoundError(f"Not found or unsupported format: {data_path}")

    def _load_from_npz(self, npz_path, test_indices=None):
        """Load from NPZ file."""
        payload = np.load(npz_path, allow_pickle=True)

        # Support multiple key layouts
        if 'X_test' in payload:
            X = payload['X_test'].astype(np.float32)
            key = 'X_test'
        elif 'X' in payload:
            X = payload['X'].astype(np.float32)
            key = 'X'
        else:
            raise KeyError(f"Unsupported npz format in {npz_path}. Expected X_test or X key.")

        if test_indices is not None:
            X = X[test_indices]

        self.sr_samples = X
        self.hr_samples = X[:, self.hr_indices, :]
        self.lr_samples = X[:, self.lr_indices, :]

        # Load metadata if available
        n = len(X)
        if 'subject_ids' in payload:
            self.subject_ids = np.asarray(payload['subject_ids']).astype(str)[:n]
        else:
            self.subject_ids = np.array([f'unknown_{i:04d}' for i in range(n)])

        if 'video_ids' in payload:
            self.video_ids = np.asarray(payload['video_ids']).astype(int)[:n]
        else:
            self.video_ids = np.arange(n, dtype=int)

        if 'trial_ids' in payload:
            self.trial_ids = np.asarray(payload['trial_ids']).astype(int)[:n]
        else:
            self.trial_ids = np.arange(n, dtype=int)

        print(f"Loaded {n} test windows from {key} in {npz_path}")

    def _load_from_folder(self, folder_path):
        """Load from preprocessed folder structure (per-subject layout)."""
        all_windows = []
        all_subject_ids = []
        all_video_ids = []
        all_trial_ids = []

        # Get all subject folders
        subject_folders = sorted([
            d for d in folder_path.iterdir()
            if d.is_dir() and re.fullmatch(r's\d{2}', d.name)
        ])

        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            if self.allowed_subjects is not None and subject_id not in self.allowed_subjects:
                continue

            x_file = subject_folder / 'X_prc1.npy'

            if x_file.exists():
                x_data = np.load(x_file, mmap_mode='r').astype(np.float32)
                all_windows.append(x_data)
                all_subject_ids.extend([subject_id] * len(x_data))

                # Load trial metadata if available
                trial_labels_file = subject_folder / 'trial_labels.json'
                if trial_labels_file.exists():
                    with open(trial_labels_file, 'r', encoding='utf-8') as f:
                        trial_meta = json.load(f)
                    windows_per_trial = trial_meta.get('windows_per_trial', None)
                    if windows_per_trial is not None:
                        trial_seq = []
                        for tid, n_w in enumerate(windows_per_trial):
                            trial_seq.extend([tid] * int(n_w))
                        if len(trial_seq) == len(x_data):
                            all_trial_ids.extend(trial_seq)
                        else:
                            all_trial_ids.extend(np.arange(len(x_data), dtype=int).tolist())
                    else:
                        all_trial_ids.extend(np.arange(len(x_data), dtype=int).tolist())
                else:
                    all_trial_ids.extend(np.arange(len(x_data), dtype=int).tolist())

                # DEAP has video_ids (0-39 per subject)
                all_video_ids.extend(np.arange(len(x_data), dtype=int).tolist())

        if not all_windows:
            raise ValueError(f"No preprocessed data found in {folder_path}")

        all_windows = np.concatenate(all_windows, axis=0).astype(np.float32)
        self.sr_samples = all_windows
        self.hr_samples = all_windows[:, self.hr_indices, :]
        self.lr_samples = all_windows[:, self.lr_indices, :]
        self.subject_ids = np.array(all_subject_ids)
        self.video_ids = np.array(all_video_ids, dtype=int)
        self.trial_ids = np.array(all_trial_ids, dtype=int)

        print(
            f"Loaded {len(self.sr_samples)} preprocessed windows from "
            f"{len(np.unique(self.subject_ids))} subjects in {folder_path}"
        )

    def __len__(self):
        return len(self.sr_samples)

    def __getitem__(self, idx):
        return {
            'lr': torch.from_numpy(self.lr_samples[idx]).float(),
            'hr': torch.from_numpy(self.hr_samples[idx]).float(),
            'sr': torch.from_numpy(self.sr_samples[idx]).float(),
            'subject_id': self.subject_ids[idx],
            'video_id': self.video_ids[idx],
            'trial_id': self.trial_ids[idx],
        }


class STAD(torch.nn.Module):
    """STAD model for DEAP (mirrors training architecture)."""

    def __init__(
        self,
        mae_encoder,
        lr_channels=8,
        hr_channels=16,
        sr_channels=32,
        latent_dim=None,
        num_patches=None,
        device='cuda',
    ):
        super().__init__()
        self.mae = mae_encoder
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels
        self.latent_dim = int(latent_dim if latent_dim is not None else getattr(mae_encoder, 'embed_dim', 256))
        self.num_patches = int(num_patches if num_patches is not None else getattr(mae_encoder, 'num_patches', 125))
        self.device = device

        # Spatial-temporal conditioning
        self.stc = SpatioTemporalConditionModule(
            lr_channels,
            seq_len=1000,
            embed_dim=self.latent_dim,
            n_harmonics=8,
            patch_size=8,
            n_transformer_layers=4,
            n_heads=8,
        )

        # Diffusion module
        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=self.num_patches,
            latent_dim=self.latent_dim,
            n_layers=6,
            n_heads=16,
        )

        # SR head
        self.sr_head = torch.nn.Conv1d(hr_channels, sr_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.sr_head.weight.zero_()
            self.sr_head.bias.zero_()
            for i in range(min(hr_channels, sr_channels // 2)):
                self.sr_head.weight[2 * i, i, 0] = 1.0

        # Diffusion schedule
        betas = self._get_beta_schedule(1000)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _get_beta_schedule(self, timesteps=1000):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def mae_encode_no_mask(self, eeg):
        """Encode to latent space."""
        x = self.mae.patch_embed(eeg)
        x = x + self.mae.pos_embed[:, 1:, :]
        cls_token = self.mae.cls_token + self.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.mae.blocks:
            x = blk(x)
        x = self.mae.norm(x)
        return x[:, 1:, :]

    def latent_to_sr(self, latent_no_cls):
        """Decode latent to SR space."""
        cls_token = self.mae.cls_token.expand(latent_no_cls.shape[0], -1, -1)
        latent_with_cls = torch.cat([cls_token, latent_no_cls], dim=1)
        x = self.mae.decoder_embed(latent_with_cls)
        x = x + self.mae.decoder_pos_embed
        for blk in self.mae.decoder_blocks:
            x = blk(x)
        x = self.mae.decoder_norm(x)
        x = self.mae.decoder_pred(x)
        hr_eeg = self.mae.unpatchify(x[:, 1:, :])
        hr_eeg = torch.nan_to_num(hr_eeg, nan=0.0, posinf=0.0, neginf=0.0)
        sr_eeg = self.sr_head(hr_eeg)
        return torch.nan_to_num(sr_eeg, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def sample_sr(self, lr_eeg, num_inference_steps=50):
        """Sample SR EEG using DDIM."""
        B = lr_eeg.shape[0]
        device = lr_eeg.device

        # Start from noise
        zt = torch.randn(B, self.num_patches, self.latent_dim, device=device)

        timesteps = torch.linspace(999, 0, num_inference_steps, dtype=torch.long, device=device)

        lr_pos = get_channel_positions(self.lr_channels, device, B)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)

            # Get conditioning
            cond_tokens, cond_pooled = self.stc(lr_eeg, lr_pos, t_batch)

            # Predict noise
            pred_noise = self.mtd(zt, t_batch, cond_tokens, cond_pooled)

            # DDIM step
            alpha_t = self.sqrt_alphas_cumprod[t] ** 2

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.sqrt_alphas_cumprod[t_prev] ** 2
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            pred_x0 = (zt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

            if i < len(timesteps) - 1:
                zt = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                zt = pred_x0

        sr_eeg = self.latent_to_sr(zt)
        return sr_eeg

    def forward(self, lr_eeg, hr_eeg, sr_eeg, t=None):
        """Training forward (used in evaluation for diffusion loss)."""
        B = lr_eeg.shape[0]
        device = lr_eeg.device

        with torch.no_grad():
            z0 = self.mae_encode_no_mask(hr_eeg)

        if t is None:
            t = torch.randint(0, 1000, (B,), device=device)

        epsilon = torch.randn_like(z0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon

        # Conditioning
        lr_pos = get_channel_positions(self.lr_channels, device, B)

        # Predict noise
        pred_epsilon = self.mtd(zt, t, *self.stc(lr_eeg, lr_pos, t))

        # MSE loss
        diff_loss = F.mse_loss(pred_epsilon, epsilon)

        # Decode to SR
        pred_z0 = (zt - sqrt_one_minus * pred_epsilon) / torch.clamp(sqrt_alpha, min=1e-3)
        pred_z0 = torch.clamp(pred_z0, min=-10.0, max=10.0)
        pred_sr = self.latent_to_sr(pred_z0)

        return diff_loss, pred_sr


def build_mae_encoder(mae_checkpoint_path, device):
    """Load 16-channel MAE model used by STAD (DEAP)."""
    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu', weights_only=False)

    cfg = _infer_mae_kwargs_from_checkpoint(checkpoint)

    mae_model = MAEforEEG(
        time_len=int(cfg['time_len']),
        patch_size=int(cfg['patch_size']),
        embed_dim=int(cfg['embed_dim']),
        in_chans=int(cfg['in_chans']),
        depth=int(cfg['depth']),
        num_heads=int(cfg['num_heads']),
        decoder_embed_dim=int(cfg['decoder_embed_dim']),
        decoder_depth=int(cfg['decoder_depth']),
        decoder_num_heads=int(cfg['decoder_num_heads']),
        mlp_ratio=float(cfg['mlp_ratio']),
        norm_pix_loss=bool(cfg['norm_pix_loss']),
    )

    if 'model_state_dict' in checkpoint:
        mae_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        state = checkpoint['model']
        if any(key.startswith('mae.') for key in state.keys()):
            state = {
                key[len('mae.'):]: value
                for key, value in state.items()
                if key.startswith('mae.')
            }
        mae_model.load_state_dict(state)
    else:
        mae_model.load_state_dict(checkpoint)

    print(
        f"Loaded MAE checkpoint {mae_checkpoint_path} | "
        f"T={cfg['time_len']}, patch={cfg['patch_size']}, embed={cfg['embed_dim']}, "
        f"in_chans={cfg['in_chans']}, heads={cfg['num_heads']}"
    )

    mae_model.to(device)
    mae_model.eval()
    for p in mae_model.parameters():
        p.requires_grad = False

    return mae_model


def save_eeg_signal_figure(pred_sr, target_sr, lr_eeg, out_path, sample_idx, channels_to_plot):
    """Save overlay plots of predicted vs target EEG signals for selected channels."""
    pred_np = pred_sr[sample_idx].detach().cpu().numpy()
    target_np = target_sr[sample_idx].detach().cpu().numpy()
    lr_np = lr_eeg[sample_idx].detach().cpu().numpy()

    n_rows = len(channels_to_plot)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for row, ch_idx in enumerate(channels_to_plot):
        ax = axes[row]
        ax.plot(target_np[ch_idx], color='black', linewidth=1.0, label='Target SR (32ch)')
        ax.plot(pred_np[ch_idx], color='tab:blue', linewidth=0.9, alpha=0.85, label='Predicted SR')

        if ch_idx < lr_np.shape[0]:
            ax.plot(lr_np[ch_idx], color='tab:orange', linewidth=0.8, alpha=0.65, label='LR input (if mapped)')

        ax.set_ylabel(f"Ch {ch_idx}")
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=8)

    axes[-1].set_xlabel('Time samples')
    fig.suptitle(f'EEG Signals: sample {sample_idx}', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mae_checkpoint_path = Path(args.mae_checkpoint).resolve()
    mae_results_dir = mae_checkpoint_path.parent.parent
    preferred_fold = _infer_fold_from_path(mae_checkpoint_path)
    fold_used, train_subjects, val_subjects = _resolve_fold_subject_split(
        str(mae_results_dir),
        preferred_fold=preferred_fold,
    )
    print(f"Fold {fold_used} held-out subjects: {val_subjects}")

    # Load test data
    dataset = DEAPTestDataset(
        data_path=args.data_path,
        test_indices=None,
        lr_channels=8,
        hr_channels=16,
        subject_ids=val_subjects,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Build MAE encoder
    mae_encoder = build_mae_encoder(args.mae_checkpoint, device)

    # Build STAD model
    model = STAD(
        mae_encoder=mae_encoder,
        lr_channels=8,
        hr_channels=16,
        sr_channels=32,
        latent_dim=getattr(mae_encoder, 'embed_dim', 256),
        num_patches=getattr(mae_encoder, 'num_patches', 125),
        device=device,
    ).to(device)

    # Load STAD checkpoint
    ckpt = torch.load(args.stad_checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading STAD checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading STAD checkpoint: {len(unexpected)}")

    model.eval()

    # Setup output directories
    fig_dir = None
    saved_figures = 0
    if args.save_fig_dir:
        fig_dir = Path(args.save_fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

    channels_to_plot = [0, 4, 8, 12, 16, 24, 31]

    diff_losses = []
    sr_losses = []
    pcc_scores = []
    nmse_scores = []
    snr_scores = []
    saved_pred_sr = []
    saved_target_sr = []
    saved_subject_ids = []
    saved_video_ids = []
    saved_trial_ids = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Testing')):
            lr_eeg = batch['lr'].to(device)
            hr_eeg = batch['hr'].to(device)
            sr_eeg = batch['sr'].to(device)
            subject_ids = list(batch['subject_id'])
            video_ids = batch['video_id'].cpu().numpy().astype(int).tolist()
            trial_ids = batch['trial_id'].cpu().numpy().astype(int).tolist()

            if args.use_sampling:
                pred_sr = model.sample_sr(lr_eeg, num_inference_steps=args.num_inference_steps)
                diff_loss = torch.tensor(float('nan'), device=device)
            else:
                diff_loss, pred_sr = model(lr_eeg, hr_eeg, sr_eeg)

            sr_loss = F.l1_loss(pred_sr.float(), sr_eeg.float())
            metrics = compute_sr_metrics(pred_sr.float(), sr_eeg.float())

            saved_pred_sr.append(pred_sr.detach().cpu().numpy())
            saved_target_sr.append(sr_eeg.detach().cpu().numpy())
            saved_subject_ids.extend(subject_ids)
            saved_video_ids.extend(video_ids)
            saved_trial_ids.extend(trial_ids)

            diff_losses.append(float(diff_loss.item()))
            sr_losses.append(float(sr_loss.item()))
            pcc_scores.append(metrics['pcc'])
            nmse_scores.append(metrics['nmse'])
            snr_scores.append(metrics['snr'])

            if fig_dir is not None and saved_figures < args.num_fig_samples:
                batch_size = pred_sr.shape[0]
                for b in range(batch_size):
                    if saved_figures >= args.num_fig_samples:
                        break
                    fig_path = fig_dir / f"eeg_signal_sample_{saved_figures:03d}.png"
                    save_eeg_signal_figure(
                        pred_sr=pred_sr,
                        target_sr=sr_eeg,
                        lr_eeg=lr_eeg,
                        out_path=fig_path,
                        sample_idx=b,
                        channels_to_plot=channels_to_plot,
                    )
                    saved_figures += 1

            if args.max_batches > 0 and (i + 1) >= args.max_batches:
                break

    mean_diff = np.nanmean(diff_losses) if diff_losses else float('nan')
    mean_sr = np.mean(sr_losses) if sr_losses else float('nan')
    mean_pcc = np.mean(pcc_scores) if pcc_scores else float('nan')
    mean_nmse = np.mean(nmse_scores) if nmse_scores else float('nan')
    mean_snr = np.mean(snr_scores) if snr_scores else float('nan')

    print('\n' + '=' * 80)
    print('DEAP Test Results')
    print('=' * 80)
    print(f"Samples tested: {len(sr_losses) * args.batch_size} (approx)")
    print(f"Diff Loss: {mean_diff:.6f}")
    print(f"SR L1 Loss: {mean_sr:.6f}")
    print(f"PCC: {mean_pcc:.4f}")
    print(f"NMSE: {mean_nmse:.4f}")
    print(f"SNR: {mean_snr:.2f} dB")
    if fig_dir is not None:
        print(f"Saved EEG figures: {saved_figures} -> {fig_dir}")

    if args.save_sr_output_path:
        sr_out = np.concatenate(saved_pred_sr, axis=0)
        save_path = Path(args.save_sr_output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, sr_out)
        print(f"Saved predicted SR EEG: {sr_out.shape} -> {save_path}")

        if args.save_target_output_path:
            target_out = np.concatenate(saved_target_sr, axis=0)
            target_path = Path(args.save_target_output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, target_out)
            print(f"Saved target SR EEG: {target_out.shape} -> {target_path}")

        if args.save_test_metadata_path:
            meta_path = Path(args.save_test_metadata_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                meta_path,
                subject_ids=np.array(saved_subject_ids),
                video_ids=np.array(saved_video_ids, dtype=int),
                trial_ids=np.array(saved_trial_ids, dtype=int),
            )
            print(f"Saved test metadata: {meta_path}")

        if args.save_subject_wise_dir:
            subject_wise_dir = Path(args.save_subject_wise_dir)
            subject_wise_dir.mkdir(parents=True, exist_ok=True)

            # Group outputs by subject
            sr_all = np.concatenate(saved_pred_sr, axis=0)
            target_all = np.concatenate(saved_target_sr, axis=0)
            subject_ids_array = np.array(saved_subject_ids)
            video_ids_array = np.array(saved_video_ids, dtype=int)
            trial_ids_array = np.array(saved_trial_ids, dtype=int)

            unique_subjects = np.unique(subject_ids_array)
            print(f"\nSaving subject-wise outputs to {subject_wise_dir}:")

            for subject_id in unique_subjects:
                mask = subject_ids_array == subject_id
                sr_subject = sr_all[mask]
                target_subject = target_all[mask]
                video_subject = video_ids_array[mask]
                trial_subject = trial_ids_array[mask]

                subject_dir = subject_wise_dir / str(subject_id)
                subject_dir.mkdir(parents=True, exist_ok=True)

                # Save predicted SR
                sr_path = subject_dir / 'pred_sr.npy'
                np.save(sr_path, sr_subject)

                # Save target SR
                target_path = subject_dir / 'target_sr.npy'
                np.save(target_path, target_subject)

                # Save metadata
                meta_path = subject_dir / 'metadata.npz'
                np.savez(
                    meta_path,
                    video_ids=video_subject,
                    trial_ids=trial_subject,
                )

                # Compute and save metrics for this subject
                sr_subject_flat = sr_subject.reshape(sr_subject.shape[0], -1)
                target_subject_flat = target_subject.reshape(target_subject.shape[0], -1)

                pred_centered = sr_subject_flat - sr_subject_flat.mean(axis=1, keepdims=True)
                target_centered = target_subject_flat - target_subject_flat.mean(axis=1, keepdims=True)
                numerator = (pred_centered * target_centered).sum(axis=1)
                denominator = np.sqrt(
                    (pred_centered**2).sum(axis=1) * (target_centered**2).sum(axis=1) + 1e-8
                )
                pcc = np.mean(numerator / denominator)

                mse = np.mean((sr_subject_flat - target_subject_flat) ** 2)
                signal_power = np.mean(target_subject_flat ** 2)
                nmse = mse / (signal_power + 1e-8)
                snr = 10 * np.log10((signal_power + 1e-8) / (mse + 1e-8))

                metrics_path = subject_dir / 'metrics.json'
                metrics_dict = {
                    'pcc': float(pcc),
                    'nmse': float(nmse),
                    'snr': float(snr),
                    'n_samples': int(len(sr_subject)),
                }
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)

                print(f"  {subject_id}: {len(sr_subject)} samples | PCC={pcc:.4f}, NMSE={nmse:.4f}, SNR={snr:.2f}dB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test STAD on DEAP dataset')
    parser.add_argument('--data_path', type=str,
                        default='/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
                        help='Path to DEAP test data NPZ file')
    parser.add_argument('--mae_checkpoint', type=str, required=True,
                        help='Path to pretrained MAE checkpoint (fold-specific)')
    parser.add_argument('--stad_checkpoint', type=str,
                        default='/home/ab_students/EEG-MTP/New_DEAP/best_stad_prcoutput_fold5.pt',
                        help='Path to trained STAD checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--use_sampling', action='store_true',
                        help='Use iterative sampling (slow, more realistic inference)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Sampling steps when --use_sampling is enabled')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='If >0, stop early after this many batches (debug)')
    parser.add_argument('--save_fig_dir', type=str, default='test_figures_deap',
                        help='Directory to save EEG signal figures (empty to disable)')
    parser.add_argument('--num_fig_samples', type=int, default=3,
                        help='How many samples to plot as EEG figures')
    parser.add_argument('--save_sr_output_path', type=str, default='',
                        help='Optional path to save predicted SR EEG windows (.npy)')
    parser.add_argument('--save_target_output_path', type=str, default='',
                        help='Optional path to save target SR EEG windows (.npy)')
    parser.add_argument('--save_test_metadata_path', type=str, default='',
                        help='Optional path to save test metadata (.npz with subject_ids/video_ids)')
    parser.add_argument('--save_subject_wise_dir', type=str, default='',
                        help='Optional directory to save subject-wise outputs (pred_sr.npy, target_sr.npy, metadata.npz, metrics.json per subject)')

    evaluate(parser.parse_args())
