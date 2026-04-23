#!/usr/bin/env python3
"""
CRITICAL FIX: Dimension and Statistics Alignment
Main fixes:
1. STC embed_dim matches MAE latent_dim (256, not 128)
2. Proper latent normalization
3. Correct reconstruction with statistics matching
"""
import os
import numpy as np
import torch
import torch.nn as nn
import json
import sys
import shutil
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# Use the exact MAE implementation used in k-fold pretraining for full latent compatibility.
sys.path.insert(0, '/home/ab_students/EEG-MTP/trial_mae_DEAP')
from mae_for_eeg import MAEforEEG
from spatio_temporal_condition import SpatioTemporalConditionModule  
from mtd_dreamdiff import MultiScaleTransformerDenoisingModule


def _safe_save_checkpoint(payload, save_path, lite_exclude_prefix='mae.'):
    """Save checkpoint robustly; fall back to lite model payload on write failures."""
    full_payload = dict(payload)
    full_payload['checkpoint_type'] = 'full'

    try:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(full_payload, save_path)
        return True, save_path, 'full'
    except Exception as e:
        print(f"⚠️  Full checkpoint save failed at {save_path}: {e}")

    lite_payload = dict(full_payload)
    model_state = lite_payload.get('model', {})
    if isinstance(model_state, dict):
        lite_payload['model'] = {
            k: v for k, v in model_state.items()
            if not k.startswith(lite_exclude_prefix)
        }
    lite_payload['checkpoint_type'] = 'lite_no_mae'

    fallback_dir = '/tmp/EEG-MTP/New_DEAP_checkpoints'
    os.makedirs(fallback_dir, exist_ok=True)
    lite_name = os.path.basename(save_path)
    if lite_name.endswith('.pt'):
        lite_name = lite_name.replace('.pt', '_lite.pt')
    elif lite_name.endswith('.pth'):
        lite_name = lite_name.replace('.pth', '_lite.pth')
    else:
        lite_name = f"{lite_name}_lite.pt"
    fallback_path = os.path.join(fallback_dir, lite_name)

    try:
        # Best-effort free-space report for debugging storage issues.
        free_gb = shutil.disk_usage('/tmp').free / (1024 ** 3)
        print(f"ℹ️  Retrying with lightweight checkpoint in /tmp (free={free_gb:.1f}GB)")
        torch.save(lite_payload, fallback_path)
        return True, fallback_path, 'lite_no_mae'
    except Exception as e:
        print(f"❌ Fallback checkpoint save also failed at {fallback_path}: {e}")
        return False, '', 'none'


def _resolve_best_kfold_checkpoint(results_dir, preferred_fold=None):
    """Return checkpoint path and metadata from a DEAP k-fold results directory."""
    summary_path = os.path.join(results_dir, 'kfold_summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"kfold_summary.json not found in {results_dir}")

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    fold_results = summary.get('fold_results', [])
    if not fold_results:
        raise ValueError(f"No fold_results found in {summary_path}")

    if preferred_fold is not None:
        selected = None
        for row in fold_results:
            if int(row['fold']) == int(preferred_fold):
                selected = row
                break
        if selected is None:
            raise ValueError(f"Requested fold {preferred_fold} not found in {summary_path}")
    else:
        selected = max(fold_results, key=lambda x: x.get('best_val_corr', float('-inf')))

    fold = int(selected['fold'])
    ckpt_path = os.path.join(results_dir, f'fold_{fold}', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path, selected


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
        _, selected_meta = _resolve_best_kfold_checkpoint(results_dir, preferred_fold=None)
        target_fold = int(selected_meta['fold'])
    else:
        target_fold = int(preferred_fold)

    selected = None
    for row in splits:
        if int(row['fold']) == target_fold:
            selected = row
            break
    if selected is None:
        raise ValueError(f"Fold {target_fold} not found in {splits_path}")

    train_subjects = [str(s) for s in selected.get('train_subjects', [])]
    val_subjects = [str(s) for s in selected.get('val_subjects', [])]
    overlap = sorted(set(train_subjects).intersection(set(val_subjects)))
    if overlap:
        raise ValueError(f"Data leakage detected in fold {target_fold}: shared subjects {overlap}")

    return target_fold, train_subjects, val_subjects


def _load_mae_from_kfold_results(mae_module, results_dir, device, preferred_fold=None, strict=True):
    """Load MAE weights from best (or chosen) k-fold checkpoint."""
    ckpt_path, selected = _resolve_best_kfold_checkpoint(results_dir, preferred_fold=preferred_fold)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    source_state = ckpt.get('model_state_dict', {})

    source_state = {
        k: (v.float() if torch.is_floating_point(v) else v)
        for k, v in source_state.items()
    }

    if strict:
        mae_module.load_state_dict(source_state, strict=True)
        loaded_msg = f"Strict load complete with {len(source_state)} tensors."
    else:
        target_state = mae_module.state_dict()
        compatible = {
            k: v for k, v in source_state.items()
            if k in target_state and target_state[k].shape == v.shape
        }
        missing, unexpected = mae_module.load_state_dict(compatible, strict=False)
        loaded_msg = (
            f"Transfer load complete: compatible={len(compatible)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    print(
        f"✅ Loaded MAE from {ckpt_path} | fold={selected['fold']} "
        f"(val_corr={selected.get('best_val_corr', 0.0):.4f})"
    )
    print(f"   {loaded_msg}")

    return ckpt.get('config', {}), ckpt_path

def get_beta_schedule(timesteps=1000):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_diffusion_params(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
    }

def get_channel_positions(n_channels, device='cpu', batch_size=1):
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
        raise ValueError(f"Unsupported n_channels={n_channels} for channel positions")
    return torch.tensor(positions, device=device).unsqueeze(0).expand(batch_size, -1, -1)


def compute_sr_metrics_torch(pred_sr, target_sr, eps=1e-8):
    """Batch-level SR metrics (PCC, NMSE, SNR dB) in torch."""
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
    return {'PCC': pcc, 'NMSE': nmse, 'SNR': snr}

def reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50):
    """FIXED reconstruction with proper dimensions"""
    model.eval()
    with torch.no_grad():
        B = x_lr.shape[0]
        lr_pos = get_channel_positions(model.lr_channels, device, B)
        
        # Start from noise in latent shape (B, num_patches, latent_dim)
        zt = torch.randn(B, model.num_patches, model.latent_dim, device=device)
        
        timesteps = torch.linspace(999, 0, steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            
            # Get conditioning
            cond_tokens, cond_pooled = model.stc(x_lr, lr_pos, t_batch)
            
            # Predict noise
            pred_noise = model.mtd(zt, t_batch, cond_tokens, cond_pooled)
            
            # DDIM step
            alpha_t = diff_params['sqrt_alphas_cumprod'][t] ** 2
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = diff_params['sqrt_alphas_cumprod'][t_prev] ** 2
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            pred_x0 = (zt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            if i < len(timesteps) - 1:
                zt = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                zt = pred_x0
        
        # Decode latent to SR through the model SR head.
        sr_eeg = model.latent_to_sr(zt)
        
        return sr_eeg

def validate_reconstruction(model, val_loader, diff_params, device):
    """Validate with proper metrics - now comparing HR channels since MAE is trained on HR"""
    metrics = {'PCC': [], 'SNR': [], 'NMSE': [], 'MAE': []}
    
    with torch.no_grad():
        for i, (x_lr, y_hr, y_sr) in enumerate(val_loader):
            if i >= 10:  # More samples for better stats
                break
                
            x_lr, y_hr, y_sr = x_lr.to(device), y_hr.to(device), y_sr.to(device)
            sr_eeg = reconstruct_eeg_fixed(model, x_lr, diff_params, device, steps=50)
            
            sr_np = sr_eeg.cpu().numpy()
            sr_gt = y_sr.cpu().numpy()
            
            # Align length
            min_len = min(sr_np.shape[2], sr_gt.shape[2])
            sr_np = sr_np[:, :, :min_len]
            sr_gt = sr_gt[:, :, :min_len]
            
            # Channel-wise PCC (most important metric)
            for b in range(sr_np.shape[0]):
                for ch in range(sr_np.shape[1]):
                    sr_sig = sr_np[b, ch]
                    sr_sig_gt = sr_gt[b, ch]
                    
                    if np.std(sr_sig) > 1e-6 and np.std(sr_sig_gt) > 1e-6:
                        pcc = np.corrcoef(sr_sig, sr_sig_gt)[0, 1]
                        if not np.isnan(pcc):
                            metrics['PCC'].append(pcc)
            
            # Other metrics
            mse = np.mean((sr_np - sr_gt) ** 2)
            nmse = mse / (np.mean(sr_gt ** 2) + 1e-10)
            snr = 10 * np.log10((np.mean(sr_gt ** 2) + 1e-10) / (mse + 1e-10))
            mae = np.mean(np.abs(sr_np - sr_gt))
            
            metrics['NMSE'].append(nmse)
            metrics['SNR'].append(snr)
            metrics['MAE'].append(mae)
    
    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}

class STADDataset(Dataset):
    def __init__(self, npz_path, split='train', lr_channels=8, hr_channels=16, sr_channels=32, window_size=400, fs=128):
        self.lr_channels, self.hr_channels, self.sr_channels = lr_channels, hr_channels, sr_channels
        self.window_size, self.fs = window_size, fs
        
        data = np.load(npz_path)
        X = data[f"X_{split}"]
        
        self.hr_samples = self.prepare_segments(X, hr_channels)
        self.lr_samples = self.prepare_segments(X, lr_channels)
        self.sr_samples = self.prepare_segments(X, sr_channels)
    
    def prepare_segments(self, X, target_channels):
        n_trials, n_channels, _ = X.shape
        if target_channels == n_channels:
            indices = np.arange(n_channels, dtype=int)
        elif target_channels == 8 and n_channels >= 32:
            indices = np.arange(0, 32, 4, dtype=int)
        elif target_channels == 16 and n_channels >= 32:
            # Match DEAP even-channel protocol used in MAE runs: 0,2,...,30
            indices = np.arange(0, 32, 2, dtype=int)
        else:
            indices = np.linspace(0, n_channels - 1, target_channels, dtype=int)
        X_sub = X[:, indices, :]
        
        def bandpass_filter(data):
            nyquist = 0.5 * self.fs
            b, a = butter(4, [max(1.0/nyquist, 0.01), min(40.0/nyquist, 0.99)], 'band')
            return filtfilt(b, a, data, axis=-1)
        
        X_filtered = np.array([bandpass_filter(trial) for trial in X_sub])
        
        segments = []
        for trial in X_filtered:
            for start in range(0, trial.shape[-1] - self.window_size + 1, self.window_size):
                segments.append(trial[:, start:start + self.window_size])
        
        X_seg = np.stack(segments)
        for ch in range(X_seg.shape[1]):
            mean, std = X_seg[:, ch].mean(axis=1, keepdims=True), X_seg[:, ch].std(axis=1, keepdims=True) + 1e-6
            X_seg[:, ch] = (X_seg[:, ch] - mean) / std
        
        return X_seg.astype(np.float32)
    
    def __len__(self): 
        return len(self.hr_samples)
    
    def __getitem__(self, idx): 
        return (
            torch.tensor(self.lr_samples[idx]),
            torch.tensor(self.hr_samples[idx]),
            torch.tensor(self.sr_samples[idx]),
        )


class STADPRCSubjectDataset(Dataset):
    """Fold-aware PRC dataset built from per-subject 32ch files, leakage-safe by subject split."""

    def __init__(self, prc_root_dir, subject_ids, lr_channels=8, hr_channels=16, sr_channels=32):
        self.prc_root_dir = prc_root_dir
        self.subject_ids = [str(s) for s in subject_ids]
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels

        self.lr_idx = np.arange(0, 32, 4, dtype=int)
        self.hr_idx = np.arange(0, 32, 2, dtype=int)
        self.sr_idx = np.arange(0, 32, dtype=int)

        self.subject_arrays = {}
        self.index_map = []
        for sid in self.subject_ids:
            x_path = os.path.join(self.prc_root_dir, sid, 'X_prc1.npy')
            if not os.path.exists(x_path):
                raise FileNotFoundError(f"Missing PRC file for subject {sid}: {x_path}")
            arr = np.load(x_path, mmap_mode='r')
            if arr.ndim != 3 or arr.shape[1] < 32:
                raise ValueError(f"Unexpected PRC shape for {sid}: {arr.shape}")
            self.subject_arrays[sid] = arr
            for i in range(arr.shape[0]):
                self.index_map.append((sid, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sid, sample_idx = self.index_map[idx]
        x32 = np.asarray(self.subject_arrays[sid][sample_idx], dtype=np.float32)

        x_lr = x32[self.lr_idx]
        x_hr = x32[self.hr_idx]
        x_sr = x32[self.sr_idx]

        return torch.tensor(x_lr), torch.tensor(x_hr), torch.tensor(x_sr)

# ✅ CRITICAL FIX: Proper dimension alignment
class STAD(nn.Module):
    def __init__(
        self,
        lr_channels=8,
        hr_channels=16,
        sr_channels=32,
        seq_len=400,
        latent_dim=256,
        n_harmonics=8,
        mae_patch_size=4,
        mae_depth=6,
        mae_num_heads=8,
        mae_decoder_embed_dim=128,
        mae_decoder_depth=4,
        mae_decoder_num_heads=8,
        mae_mlp_ratio=4.0,
        mae_norm_pix_loss=True,
    ):
        super().__init__()
        
        patch_size = mae_patch_size
        num_patches = seq_len // patch_size
        
        # ✅ FIX: MAE in_chans must match pretrained checkpoint (hr_channels=16, not sr_channels=32)
        self.mae = MAEforEEG(
            time_len=seq_len,
            patch_size=patch_size,
            embed_dim=latent_dim,
            in_chans=hr_channels,  # ✅ 16 to match pretrained MAE
            depth=mae_depth,
            num_heads=mae_num_heads,
            decoder_embed_dim=mae_decoder_embed_dim,
            decoder_depth=mae_decoder_depth,
            decoder_num_heads=mae_decoder_num_heads,
            mlp_ratio=mae_mlp_ratio,
            norm_pix_loss=mae_norm_pix_loss,
        )
        
        # ✅ FIX: STC with embed_dim=256 (not 128!)
        self.stc = SpatioTemporalConditionModule(
            lr_channels, 
            seq_len, 
            embed_dim=latent_dim,  # ✅ 256 to match MAE
            n_harmonics=n_harmonics, 
            patch_size=patch_size,  # ✅ Use same patch_size as MAE (8), not hard-coded 16
            n_transformer_layers=4,
            n_heads=8
        )
        
        # MTD (latent_dim=256, num_patches=100)
        self.mtd = MultiScaleTransformerDenoisingModule(
            num_patches=num_patches,
            latent_dim=latent_dim,
            n_layers=6,
            n_heads=16
        )
        # SEED4-style direct SR supervision head from decoded HR waveform.
        self.sr_head = nn.Conv1d(hr_channels, sr_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.sr_head.weight.zero_()
            self.sr_head.bias.zero_()
            for i in range(min(hr_channels, sr_channels // 2)):
                self.sr_head.weight[2 * i, i, 0] = 1.0

        self.use_mae_checkpointing = True
        
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.lr_channels = lr_channels
        self.hr_channels = hr_channels
        self.sr_channels = sr_channels

    def mae_encode_no_mask(self, eeg):
        """Exact no-mask encoder forward for trial_mae_DEAP MAE."""
        x = self.mae.patch_embed(eeg)
        x = x + self.mae.pos_embed[:, 1:, :]
        cls_token = self.mae.cls_token + self.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.mae.blocks:
            x = blk(x)
        x = self.mae.norm(x)
        return x

    def mae_decode_full(self, encoded_with_cls):
        """Exact full-sequence decoder forward for trial_mae_DEAP MAE."""
        x = self.mae.decoder_embed(encoded_with_cls)
        x = x + self.mae.decoder_pos_embed
        for blk in self.mae.decoder_blocks:
            if self.use_mae_checkpointing and self.training and x.requires_grad:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.mae.decoder_norm(x)
        x = self.mae.decoder_pred(x)
        return x[:, 1:, :]
    
    def encode_hr(self, hr_eeg):
        """Encode HR to latent (unnormalized for diffusion)"""
        latent = self.mae_encode_no_mask(hr_eeg)
        latent = latent[:, 1:, :]  # Remove CLS → (B, 100, 256)
        
        # Don't normalize - diffusion models work better with unnormalized latents
        # The noise epsilon is sampled from N(0, 1) and should match the latent scale
        return latent

    def latent_to_hr(self, latent_no_cls):
        cls_token = self.mae.cls_token.expand(latent_no_cls.shape[0], -1, -1)
        latent_with_cls = torch.cat([cls_token, latent_no_cls], dim=1)
        pred_patches = self.mae_decode_full(latent_with_cls)
        hr_eeg = self.mae.unpatchify(pred_patches)
        return torch.nan_to_num(hr_eeg, nan=0.0, posinf=0.0, neginf=0.0)

    def latent_to_sr(self, latent_no_cls):
        hr_eeg = self.latent_to_hr(latent_no_cls)
        sr_eeg = self.sr_head(hr_eeg)
        return torch.nan_to_num(sr_eeg, nan=0.0, posinf=0.0, neginf=0.0)
    
    def forward(self, lr_eeg, zt, t_steps, lr_chan_pos):
        cond_tokens, cond_pooled = self.stc(lr_eeg, lr_chan_pos, t_steps)
        return self.mtd(zt, t_steps, cond_tokens, cond_pooled)

def train_stad_fixed(
    dataset_path,
    num_epochs=300,
    batch_size=32,
    lr=2e-4,
    mae_results_dir=None,
    mae_fold=None,
    require_full_mae_latent=True,
    prc_root_dir=None,
    lambda_eps=1.0,
    lambda_recon=0.05,
    lambda_signal=1.0,  # ✅ Default increased; properly normalized signal loss
    adaptive_gradnorm=True,
    gradnorm_eta=0.005,
    sr_loss_weight=1.0,
    resume_stad_checkpoint='',
    resume_optimizer=False,
    save_optimizer_state=False,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 STAD Training (DIMENSION FIXED)")
    selected_fold = None

    # Paper-style setup: LR=8, HR=16, SR=32.
    hr_channels = 16
    lr_channels = 8
    sr_channels = 32
    seq_len = 1000
    latent_dim = 256
    mae_patch_size = 8
    mae_depth = 12
    mae_num_heads = 12
    mae_decoder_embed_dim = 384
    mae_decoder_depth = 4
    mae_decoder_num_heads = 8
    mae_mlp_ratio = 4.0
    mae_norm_pix_loss = True

    if mae_results_dir is not None and os.path.isdir(mae_results_dir):
        ckpt_path, selected = _resolve_best_kfold_checkpoint(mae_results_dir, preferred_fold=mae_fold)
        selected_fold = selected.get('fold')
        ckpt_preview = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt_preview.get('config', {})

        checkpoint_in_chans = int(cfg.get('in_chans', hr_channels))
        if checkpoint_in_chans != hr_channels and require_full_mae_latent:
            raise ValueError(
                f"Expected MAE checkpoint with in_chans={hr_channels} for HR latent training, "
                f"but got in_chans={checkpoint_in_chans}."
            )

        lr_channels = min(8, hr_channels)
        seq_len = int(cfg.get('time_len', seq_len))
        latent_dim = int(cfg.get('embed_dim', latent_dim))
        mae_patch_size = int(cfg.get('patch_size', mae_patch_size))
        mae_depth = int(cfg.get('depth', mae_depth))
        mae_num_heads = int(cfg.get('num_heads', mae_num_heads))
        mae_decoder_embed_dim = int(cfg.get('decoder_embed_dim', mae_decoder_embed_dim))
        mae_decoder_depth = int(cfg.get('decoder_depth', mae_decoder_depth))
        mae_decoder_num_heads = int(cfg.get('decoder_num_heads', mae_decoder_num_heads))
        mae_mlp_ratio = float(cfg.get('mlp_ratio', mae_mlp_ratio))
        mae_norm_pix_loss = bool(cfg.get('norm_pix_loss', mae_norm_pix_loss))

        print(
            f"📌 Using MAE config from fold {selected['fold']} in {mae_results_dir}: "
            f"HR={hr_channels}, SR={sr_channels}, T={seq_len}, patch={mae_patch_size}, latent={latent_dim}"
        )

    if seq_len % mae_patch_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by mae_patch_size={mae_patch_size}")
    
    if prc_root_dir is not None:
        if mae_results_dir is None:
            raise ValueError("mae_results_dir is required when using prc_root_dir for fold-aware subject split.")
        fold_used, train_subjects, val_subjects = _resolve_fold_subject_split(mae_results_dir, preferred_fold=mae_fold)
        selected_fold = fold_used
        print(f"📌 Fold {fold_used} subject split from pretrained results")
        print(f"   Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"   Val subjects ({len(val_subjects)}): {val_subjects}")

        train_dataset = STADPRCSubjectDataset(
            prc_root_dir=prc_root_dir,
            subject_ids=train_subjects,
            lr_channels=lr_channels,
            hr_channels=hr_channels,
            sr_channels=sr_channels,
        )
        val_dataset = STADPRCSubjectDataset(
            prc_root_dir=prc_root_dir,
            subject_ids=val_subjects,
            lr_channels=lr_channels,
            hr_channels=hr_channels,
            sr_channels=sr_channels,
        )
    else:
        train_dataset = STADDataset(
            dataset_path,
            'train',
            lr_channels=lr_channels,
            hr_channels=hr_channels,
            sr_channels=sr_channels,
            window_size=seq_len,
        )
        val_dataset = STADDataset(
            dataset_path,
            'val',
            lr_channels=lr_channels,
            hr_channels=hr_channels,
            sr_channels=sr_channels,
            window_size=seq_len,
        )
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # ✅ Model with correct dimensions
    model = STAD(
        lr_channels=lr_channels,
        hr_channels=hr_channels,
        sr_channels=sr_channels,
        seq_len=seq_len,
        latent_dim=latent_dim,
        n_harmonics=8,
        mae_patch_size=mae_patch_size,
        mae_depth=mae_depth,
        mae_num_heads=mae_num_heads,
        mae_decoder_embed_dim=mae_decoder_embed_dim,
        mae_decoder_depth=mae_decoder_depth,
        mae_decoder_num_heads=mae_decoder_num_heads,
        mae_mlp_ratio=mae_mlp_ratio,
        mae_norm_pix_loss=mae_norm_pix_loss,
    ).to(device)
    
    # Load pretrained MAE
    if mae_results_dir is not None and os.path.isdir(mae_results_dir):
        _cfg, _path = _load_mae_from_kfold_results(
            model.mae,
            mae_results_dir,
            device,
            preferred_fold=mae_fold,
            strict=False,
        )
    else:
        mae_checkpoint = 'mae_deap_FIXED.pt'
        if os.path.exists(mae_checkpoint):
            mae_state = torch.load(mae_checkpoint, map_location=device, weights_only=False)
            if isinstance(mae_state, dict) and 'model_state_dict' in mae_state:
                mae_state = mae_state['model_state_dict']
            model.mae.load_state_dict(mae_state, strict=True)
            print(f"✅ Loaded MAE from {mae_checkpoint}")
        else:
            print("⚠️  No MAE checkpoint found. Training STAD from randomly initialized MAE.")
        
    # Freeze encoder initially
    for p in model.mae.parameters():
        p.requires_grad = False
    print("🔒 MAE frozen")
    
    # Diffusion
    T = 1000
    betas = get_beta_schedule(T).to(device)
    diff_params = get_diffusion_params(betas)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()
    signal_criterion = nn.L1Loss()  # L1 is more stable than MSE for signal-level
    scaler = GradScaler('cuda')

    dataset_tag = 'prcoutput' if prc_root_dir is not None else 'deap'
    fold_tag = f"_fold{selected_fold}" if selected_fold is not None else ""
    best_ckpt_name = f"best_stad_{dataset_tag}{fold_tag}.pt"
    latest_ckpt_name = f"latest_stad_{dataset_tag}{fold_tag}.pt"
    preferred_save_root = os.environ.get('STAD_SAVE_DIR', '').strip()
    if preferred_save_root:
        ckpt_root = preferred_save_root
    else:
        cwd_free_gb = shutil.disk_usage(os.getcwd()).free / (1024 ** 3)
        ckpt_root = os.getcwd() if cwd_free_gb >= 3.0 else '/tmp/EEG-MTP/New_DEAP_checkpoints'
    os.makedirs(ckpt_root, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_root, best_ckpt_name)
    latest_ckpt_path = os.path.join(ckpt_root, latest_ckpt_name)
    print(f"💾 Best checkpoint filename: {best_ckpt_name}")
    print(f"💾 Latest checkpoint filename: {latest_ckpt_name}")
    print(f"📁 Checkpoint directory: {ckpt_root}")
    print(f"⚖️  Loss weights: lambda_eps={lambda_eps}, lambda_recon={lambda_recon}, lambda_signal={lambda_signal}")
    print(f"⚖️  SR supervision weight: {sr_loss_weight}")
    print(f"🧭 Adaptive GradNorm: {adaptive_gradnorm} (eta={gradnorm_eta})")

    ref_param = next((p for p in model.mtd.parameters() if p.requires_grad), None)
    ema_g_eps = None
    ema_g_rec = None
    ema_g_sig = None
    gradnorm_beta = 0.9
    lambda_min = 1e-4
    lambda_max_aux = 0.2
    lambda_aux_budget = 0.2
    
    best_val_loss = float('inf')
    start_epoch = 0
    mae_unfrozen = False

    if resume_stad_checkpoint:
        resume_path = resume_stad_checkpoint
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model_state = payload.get('model')
        if model_state is None:
            raise KeyError(f"Resume checkpoint missing 'model': {resume_path}")
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"⚠️  Missing keys while resuming: {len(missing)}")
        if unexpected:
            print(f"⚠️  Unexpected keys while resuming: {len(unexpected)}")

        start_epoch = int(payload.get('epoch', 0))
        best_val_loss = float(payload.get('best_val_loss', payload.get('val_loss', float('inf'))))
        lambda_recon = float(payload.get('lambda_recon', lambda_recon))
        lambda_signal = float(payload.get('lambda_signal', lambda_signal))

        if resume_optimizer and 'optimizer_state_dict' in payload:
            try:
                optimizer.load_state_dict(payload['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️  Could not load optimizer state: {e}")
        if resume_optimizer and 'scheduler_state_dict' in payload:
            try:
                scheduler.load_state_dict(payload['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️  Could not load scheduler state: {e}")
        if resume_optimizer and 'scaler_state_dict' in payload:
            try:
                scaler.load_state_dict(payload['scaler_state_dict'])
            except Exception as e:
                print(f"⚠️  Could not load scaler state: {e}")

        print(f"🔁 Resumed STAD from: {resume_path}")
        print(f"   Starting epoch: {start_epoch + 1}/{num_epochs}")
        print(f"   Best val loss so far: {best_val_loss:.6f}")

    if start_epoch >= num_epochs:
        print(f"✓ Resume checkpoint already at epoch {start_epoch}; nothing to train.")
        return
    
    for epoch in range(start_epoch, num_epochs):
        # Unfreeze MAE after 50 epochs (also works for resumed runs after epoch 50).
        if (not mae_unfrozen) and (epoch >= 50):
            print("\n🔓 Unfreezing MAE for fine-tuning")
            for p in model.mae.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr / 10, 
                weight_decay=0.05
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - epoch)
            mae_unfrozen = True
        
        # Train
        model.train()
        train_loss = 0.0
        train_eps_loss = 0.0
        train_recon_loss = 0.0
        train_signal_loss = 0.0
        
        for x_lr, y_hr, y_sr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_lr, y_hr, y_sr = x_lr.to(device), y_hr.to(device), y_sr.to(device)
            B = x_lr.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', dtype=torch.float16):
                # ✅ FIX: Encode HR not SR - MAE was trained on HR (16ch) not SR (32ch)
                with torch.no_grad():
                    z0 = model.encode_hr(y_hr)
                
                # Sample timestep and noise
                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)  # N(0,1) noise
                
                # Forward diffusion
                sqrt_alpha = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon
                
                # Conditioning
                lr_pos = get_channel_positions(model.lr_channels, device, B)
                
                # Predict noise
                pred_epsilon = model(x_lr, zt, t, lr_pos)

                # Primary diffusion loss
                eps_loss = criterion(pred_epsilon, epsilon)

                # Direct latent reconstruction target from predicted epsilon
                safe_sqrt_alpha = torch.clamp(sqrt_alpha, min=1e-3)
                pred_z0 = (zt - sqrt_one_minus * pred_epsilon) / safe_sqrt_alpha
                recon_loss = recon_criterion(pred_z0, z0)
                # Normalize by latent dimension to match epsilon scale (~1-2)
                recon_loss = recon_loss / (model.latent_dim + 1e-8)

                # SEED4-style direct SR supervision.
                pred_z0_decode = torch.clamp(pred_z0, min=-10.0, max=10.0)
                pred_sr_eeg = model.latent_to_sr(pred_z0_decode)
                signal_loss = signal_criterion(pred_sr_eeg, y_sr)
                signal_loss = signal_loss / (model.sr_channels * x_lr.shape[2] + 1e-8)
                signal_loss = torch.nan_to_num(signal_loss, nan=0.0, posinf=0.0, neginf=0.0)

                if adaptive_gradnorm and ref_param is not None:
                    g_eps = torch.autograd.grad(eps_loss, ref_param, retain_graph=True, allow_unused=True)[0]
                    g_rec = torch.autograd.grad(recon_loss, ref_param, retain_graph=True, allow_unused=True)[0]
                    g_sig = torch.autograd.grad(signal_loss, ref_param, retain_graph=True, allow_unused=True)[0]

                    g_eps_n = float(g_eps.norm().item()) if g_eps is not None else 0.0
                    g_rec_n = float(g_rec.norm().item()) if g_rec is not None else 0.0
                    g_sig_n = float(g_sig.norm().item()) if g_sig is not None else 0.0

                    if np.isfinite(g_eps_n) and g_eps_n > 0.0:
                        if ema_g_eps is None:
                            ema_g_eps, ema_g_rec, ema_g_sig = g_eps_n, g_rec_n, g_sig_n
                        else:
                            ema_g_eps = (gradnorm_beta * ema_g_eps) + ((1.0 - gradnorm_beta) * g_eps_n)
                            ema_g_rec = (gradnorm_beta * ema_g_rec) + ((1.0 - gradnorm_beta) * g_rec_n)
                            ema_g_sig = (gradnorm_beta * ema_g_sig) + ((1.0 - gradnorm_beta) * g_sig_n)

                        if ema_g_rec is not None and ema_g_rec > 0.0 and np.isfinite(ema_g_rec):
                            ratio_rec = ema_g_eps / (ema_g_rec + 1e-8)
                            lambda_recon = float(np.clip(lambda_recon * (ratio_rec ** gradnorm_eta), lambda_min, lambda_max_aux))
                        if ema_g_sig is not None and ema_g_sig > 0.0 and np.isfinite(ema_g_sig):
                            ratio_sig = ema_g_eps / (ema_g_sig + 1e-8)
                            lambda_signal = float(np.clip(lambda_signal * (ratio_sig ** gradnorm_eta), lambda_min, lambda_max_aux))

                        # Keep epsilon dominant by constraining total auxiliary weight.
                        aux_sum = lambda_recon + lambda_signal
                        if aux_sum > lambda_aux_budget:
                            scale = lambda_aux_budget / (aux_sum + 1e-8)
                            lambda_recon = max(lambda_min, lambda_recon * scale)
                            lambda_signal = max(lambda_min, lambda_signal * scale)

                # Balanced multi-objective
                loss = (lambda_eps * eps_loss) + (lambda_recon * recon_loss) + (sr_loss_weight * lambda_signal * signal_loss)
                loss = torch.nan_to_num(loss, nan=eps_loss.detach(), posinf=eps_loss.detach(), neginf=eps_loss.detach())
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_eps_loss += eps_loss.item()
            train_recon_loss += recon_loss.item()
            train_signal_loss += signal_loss.item()
        
        train_loss /= len(train_loader)
        train_eps_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_signal_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_eps_loss = 0.0
        val_recon_loss = 0.0
        val_signal_loss = 0.0
        val_pcc = []
        val_nmse = []
        val_snr = []
        with torch.no_grad():
            for x_lr, y_hr, y_sr in val_loader:
                x_lr, y_hr, y_sr = x_lr.to(device), y_hr.to(device), y_sr.to(device)
                B = x_lr.size(0)
                
                z0 = model.encode_hr(y_hr)  # ✅ Encode HR (16ch) to match MAE training
                t = torch.randint(0, T, (B,), device=device)
                epsilon = torch.randn_like(z0)
                
                sqrt_alpha = diff_params['sqrt_alphas_cumprod'][t].view(B, 1, 1)
                sqrt_one_minus = diff_params['sqrt_one_minus_alphas_cumprod'][t].view(B, 1, 1)
                zt = sqrt_alpha * z0 + sqrt_one_minus * epsilon
                
                lr_pos = get_channel_positions(model.lr_channels, device, B)
                pred_epsilon = model(x_lr, zt, t, lr_pos)

                eps_loss = criterion(pred_epsilon, epsilon)
                safe_sqrt_alpha = torch.clamp(sqrt_alpha, min=1e-3)
                pred_z0 = (zt - sqrt_one_minus * pred_epsilon) / safe_sqrt_alpha
                recon_loss = recon_criterion(pred_z0, z0)
                # Normalize by latent dimension to match epsilon scale (~1-2)
                recon_loss = recon_loss / (model.latent_dim + 1e-8)
                
                pred_z0_decode = torch.clamp(pred_z0, min=-10.0, max=10.0)
                pred_sr_eeg = model.latent_to_sr(pred_z0_decode)
                signal_loss = signal_criterion(pred_sr_eeg, y_sr)
                signal_loss = signal_loss / (model.sr_channels * x_lr.shape[2] + 1e-8)
                signal_loss = torch.nan_to_num(signal_loss, nan=0.0, posinf=0.0, neginf=0.0)
                
                total_loss = (lambda_eps * eps_loss) + (lambda_recon * recon_loss) + (sr_loss_weight * lambda_signal * signal_loss)
                total_loss = torch.nan_to_num(total_loss, nan=eps_loss.detach(), posinf=eps_loss.detach(), neginf=eps_loss.detach())

                sr_metrics = compute_sr_metrics_torch(pred_sr_eeg.float(), y_sr.float())
                val_pcc.append(sr_metrics['PCC'])
                val_nmse.append(sr_metrics['NMSE'])
                val_snr.append(sr_metrics['SNR'])

                val_loss += total_loss.item()
                val_eps_loss += eps_loss.item()
                val_recon_loss += recon_loss.item()
                val_signal_loss += signal_loss.item()
        
        val_loss /= len(val_loader)
        val_eps_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_signal_loss /= len(val_loader)
        mean_pcc = float(np.mean(val_pcc)) if val_pcc else 0.0
        mean_nmse = float(np.mean(val_nmse)) if val_nmse else float('inf')
        mean_snr = float(np.mean(val_snr)) if val_snr else -float('inf')
        scheduler.step()
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: {train_loss:.6f} (eps={train_eps_loss:.6f}, rec={train_recon_loss:.6f}, sig={train_signal_loss:.6f}) | "
            f"Val: {val_loss:.6f} (eps={val_eps_loss:.6f}, rec={val_recon_loss:.6f}, sig={val_signal_loss:.6f}) | "
            f"PCC={mean_pcc:.4f}, NMSE={mean_nmse:.4f}, SNR={mean_snr:.2f}dB"
        )
        print(f"   lambdas -> eps={lambda_eps:.4f}, rec={lambda_recon:.4f}, sig={lambda_signal:.4f}")
        
        # Check if loss is in expected range
        if epoch > 10:
            if train_loss < 0.01:
                print("⚠️  WARNING: Loss suspiciously low - possible collapse!")
            elif train_loss > 2.0:
                print("⚠️  WARNING: Loss very high - possible gradient issues!")
        
        def _build_ckpt_payload(epoch_idx):
            payload = {
                'model': model.state_dict(),
                'epoch': epoch_idx + 1,
                'best_val_loss': best_val_loss,
                'val_loss': val_loss,
                'val_eps_loss': val_eps_loss,
                'val_recon_loss': val_recon_loss,
                'val_signal_loss': val_signal_loss,
                'val_pcc': mean_pcc,
                'val_nmse': mean_nmse,
                'val_snr_db': mean_snr,
                'train_loss': train_loss,
                'train_eps_loss': train_eps_loss,
                'train_recon_loss': train_recon_loss,
                'train_signal_loss': train_signal_loss,
                'lambda_eps': lambda_eps,
                'lambda_recon': lambda_recon,
                'lambda_signal': lambda_signal,
                'sr_loss_weight': sr_loss_weight,
                'diff_params': diff_params,
            }
            if save_optimizer_state:
                payload['optimizer_state_dict'] = optimizer.state_dict()
                payload['scheduler_state_dict'] = scheduler.state_dict()
                payload['scaler_state_dict'] = scaler.state_dict()
            return payload
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            saved_ok, saved_path, saved_kind = _safe_save_checkpoint(_build_ckpt_payload(epoch), best_ckpt_path)
            if saved_ok:
                print(f"✅ SAVED [{saved_kind}] (val={val_loss:.6f}) -> {saved_path}")
            else:
                print(f"⚠️  Could not save checkpoint this epoch (val={val_loss:.6f}); training will continue.")

        latest_ok, latest_path, latest_kind = _safe_save_checkpoint(_build_ckpt_payload(epoch), latest_ckpt_path)
        if not latest_ok:
            print(f"⚠️  Could not save latest checkpoint at epoch {epoch+1}.")
        else:
            print(f"📝 LATEST [{latest_kind}] -> {latest_path}")

    print(f"\n🎉 Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"💾 Final checkpoint target: {best_ckpt_path}")
    print(f"💾 Final latest checkpoint: {latest_ckpt_path}")


if __name__ == '__main__':
    train_stad_fixed(
        '/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
        num_epochs=300,
        batch_size=32,
        lr=2e-4,
        mae_results_dir='/home/ab_students/EEG-MTP/trial_mae_DEAP/results_kfold_prcoutput_mask75_val75_even16',
        require_full_mae_latent=False,
        prc_root_dir='/DATA/EEG-MTP/DEAP-PrC_final',
        lambda_signal=1.0,
        sr_loss_weight=0.5,
        resume_stad_checkpoint='',
        resume_optimizer=False,
        save_optimizer_state=False,
    )