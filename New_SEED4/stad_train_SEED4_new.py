#!/usr/bin/env python3
"""
STAD Training for SEED-IV Dataset (62 channels, 250Hz)
Runs on preprocessed_data.npz with keys: LR, HR, SR, labels, train_indices, val_indices, test_indices
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from mae_for_eeg import MAEforEEG
from stad_model_CORRECT import STADModel


# -------------------------------------------------------------
# Channel index helpers
# -------------------------------------------------------------
def get_seed4_channel_indices(target_channels):
    if target_channels == 62:
        return np.arange(62, dtype=int)
    if target_channels == 31:
        return np.array([
            0, 2, 4, 5, 7, 9, 11, 13,
            15, 17, 19, 21, 23, 25, 27, 29,
            31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 58, 60,
        ], dtype=int)
    if target_channels == 16:
        return np.array([
            0, 2, 5, 7, 9, 13, 17, 21,
            23, 27, 31, 35, 39, 45, 53, 60,
        ], dtype=int)
    return np.linspace(0, 61, target_channels, dtype=int)


# -------------------------------------------------------------
# Metrics
# -------------------------------------------------------------
def compute_sr_metrics(pred_sr, target_sr, eps=1e-8):
    pred   = pred_sr.reshape(pred_sr.shape[0], -1)
    target = target_sr.reshape(target_sr.shape[0], -1)

    pred_c   = pred   - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt((pred_c.pow(2).sum(dim=1) + eps) * (target_c.pow(2).sum(dim=1) + eps))
    pcc  = (num / den).mean().item()

    mse  = (pred - target).pow(2).mean(dim=1)
    sig  = target.pow(2).mean(dim=1)
    nmse = (mse / (sig + eps)).mean().item()
    snr  = (10.0 * torch.log10((sig + eps) / (mse + eps))).mean().item()

    return {'pcc': pcc, 'nmse': nmse, 'snr': snr}


# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class PreprocessedSEED4Dataset(Dataset):
    """
    Loads from preprocessed_data.npz which contains:
        SR            : (N, 62, 1000)
        (Optional: train_indices, val_indices, test_indices)

    If indices not present, auto-splits by subject_id if available, else random split (70/15/15).
    LR (16ch) and HR (31ch) derived on-the-fly by channel subsampling.
    """
    def __init__(self, npz_path, split='train', lr_channels=16, hr_channels=31, sr_channels=62):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        payload = np.load(npz_path, allow_pickle=True)

        sr_all = payload['SR'].astype(np.float32)  # (N, 62, 1000)
        N = len(sr_all)

        # Try to load precomputed indices
        if f'{split}_indices' in payload.files:
            indices = payload[f'{split}_indices'].astype(int)
        else:
            # Auto-generate split from subject_ids if available
            if 'subject_ids' in payload.files:
                subject_ids = payload['subject_ids']
                from sklearn.model_selection import train_test_split
                all_idx = np.arange(N)
                train_idx, rest_idx = train_test_split(
                    all_idx, test_size=0.3, random_state=2024,
                    stratify=subject_ids if len(np.unique(subject_ids)) > 1 else None
                )
                val_idx, test_idx = train_test_split(
                    rest_idx, test_size=0.5, random_state=2024,
                    stratify=subject_ids[rest_idx] if len(np.unique(subject_ids[rest_idx])) > 1 else None
                )
                splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
                indices = splits[split]
            else:
                # Fallback: random split (70/15/15)
                all_idx = np.arange(N)
                train_idx, rest_idx = train_test_split(all_idx, test_size=0.3, random_state=2024)
                val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=2024)
                splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
                indices = splits[split]

        self.sr_samples = sr_all[indices]

        lr_idx = get_seed4_channel_indices(lr_channels)
        hr_idx = get_seed4_channel_indices(hr_channels)

        self.lr_samples = self.sr_samples[:, lr_idx, :]
        self.hr_samples = self.sr_samples[:, hr_idx, :]
        self.lr_indices = lr_idx

        print(f"  [{split}] {len(self.sr_samples)} windows | "
              f"LR={self.lr_samples.shape} HR={self.hr_samples.shape} SR={self.sr_samples.shape}")

    def __len__(self):
        return len(self.sr_samples)

    def __getitem__(self, idx):
        return {
            'lr': torch.from_numpy(self.lr_samples[idx]),
            'hr': torch.from_numpy(self.hr_samples[idx]),
            'sr': torch.from_numpy(self.sr_samples[idx]),
        }


# -------------------------------------------------------------
# MAE loading
# -------------------------------------------------------------
def load_mae(checkpoint_path, device, freeze_encoder=True):
    print(f"\n{'='*70}")
    print(f"Loading MAE checkpoint: {checkpoint_path}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    mae_config = {
        'time_len': 1000, 'patch_size': 8, 'embed_dim': 768,
        'in_chans': 31,   'depth': 12,     'num_heads': 12,
        'decoder_embed_dim': 384, 'decoder_depth': 4,
        'decoder_num_heads': 8,   'mlp_ratio': 4.0,
    }

    mae_model = MAEforEEG(**mae_config)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        state = (checkpoint.get('model_state_dict')
                 or checkpoint.get('model')
                 or checkpoint)
    else:
        state = checkpoint

    mae_model.load_state_dict(state)
    print(f"✅ MAE weights loaded")

    if freeze_encoder:
        for name, param in mae_model.named_parameters():
            if 'decoder' not in name:
                param.requires_grad = False
        frozen = sum(p.numel() for p in mae_model.parameters() if not p.requires_grad)
        print(f"🔒 Encoder frozen ({frozen:,} params)")

    mae_model = mae_model.to(device)
    print(f"{'='*70}\n")
    return mae_model, mae_config


# -------------------------------------------------------------
# Training loop
# -------------------------------------------------------------
def train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir):
    trainable = [p for p in stad_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    history       = []
    start_epoch   = 0
    mae_unfrozen  = False

    if args.resume_stad_checkpoint:
        p = Path(args.resume_stad_checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {p}")
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        stad_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch   = int(ckpt.get('epoch', 0))
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        print(f"🔁 Resumed from epoch {start_epoch}, best val loss {best_val_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):

        # Optionally unfreeze MAE encoder mid-training
        if (args.freeze_mae and not mae_unfrozen
                and args.unfreeze_mae_epoch >= 0
                and epoch >= args.unfreeze_mae_epoch):
            new_params = []
            for p in stad_model.mae_encoder.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)
            if new_params:
                optimizer.add_param_group({'params': new_params, 'lr': args.mae_finetune_lr})
                print(f"🔓 Unfroze MAE encoder at epoch {epoch+1}")
            mae_unfrozen = True

        # ---- Train ----
        stad_model.train()
        t_total, t_diff, t_sr = [], [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            sr = batch['sr'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                diff_loss, pred_sr = stad_model(lr, hr, sr)
                sr_loss = F.mse_loss(pred_sr.float(), sr.float())
                total   = diff_loss + args.sr_loss_weight * sr_loss

            if not torch.isfinite(total):
                continue

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in stad_model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            t_total.append(total.item())
            t_diff.append(diff_loss.item())
            t_sr.append(sr_loss.item())

        # ---- Validate ----
        stad_model.eval()
        v_total, v_diff, v_sr = [], [], []
        v_pcc, v_nmse, v_snr  = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                sr = batch['sr'].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    diff_loss, pred_sr = stad_model(lr, hr, sr)
                    sr_loss = F.mse_loss(pred_sr.float(), sr.float())
                    total   = diff_loss + args.sr_loss_weight * sr_loss

                if not torch.isfinite(total):
                    continue

                metrics = compute_sr_metrics(pred_sr.float(), sr.float())
                v_total.append(total.item())
                v_diff.append(diff_loss.item())
                v_sr.append(sr_loss.item())
                v_pcc.append(metrics['pcc'])
                v_nmse.append(metrics['nmse'])
                v_snr.append(metrics['snr'])

        scheduler.step()

        tr   = float(np.mean(t_total)) if t_total else float('inf')
        td   = float(np.mean(t_diff))  if t_diff  else float('inf')
        ts   = float(np.mean(t_sr))    if t_sr    else float('inf')
        vl   = float(np.mean(v_total)) if v_total else float('inf')
        vd   = float(np.mean(v_diff))  if v_diff  else float('inf')
        vs   = float(np.mean(v_sr))    if v_sr    else float('inf')
        pcc  = float(np.mean(v_pcc))   if v_pcc   else 0.0
        nmse = float(np.mean(v_nmse))  if v_nmse  else float('inf')
        snr  = float(np.mean(v_snr))   if v_snr   else -float('inf')

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Total: {tr:.6f} (Diff {td:.6f}, SR {ts:.6f}) | "
            f"Val Total: {vl:.6f} (Diff {vd:.6f}, SR {vs:.6f}) | "
            f"PCC: {pcc:.4f} | NMSE: {nmse:.4f} | SNR: {snr:.2f} dB"
        )

        history.append({
            'epoch': epoch+1,
            'train_total_loss': tr, 'train_diff_loss': td, 'train_sr_loss': ts,
            'val_total_loss': vl,   'val_diff_loss': vd,   'val_sr_loss': vs,
            'val_pcc': pcc, 'val_nmse': nmse, 'val_snr_db': snr,
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        payload = {
            'epoch': epoch+1,
            'model_state_dict': stad_model.state_dict(),
            'best_val_loss': best_val_loss,
            'val_total_loss': vl,
            'val_pcc': pcc,
        }

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(payload, output_dir / 'best_stad_model.pth')
            print(f"  ✅ Best model saved (val_loss={vl:.6f})")

        torch.save(payload, output_dir / 'latest_stad_model.pth')

    np.save(output_dir / 'training_history.npy', history, allow_pickle=True)
    print(f"\n📈 Training history saved.")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser('SEED-IV STAD Training')

    # Paths
    parser.add_argument('--data_path',      type=str,
                        default='/home/arnav-a5000/MTP-EEG/DATA/preprocessed_data.npz')
    parser.add_argument('--mae_checkpoint', type=str,
                        default='/home/arnav-a5000/MTP-EEG/trial_mae_SEED4/results_31ch_kfold_fixed/best_model.pth')
    parser.add_argument('--output_dir',     type=str, default='results_stad')

    # Training
    parser.add_argument('--epochs',             type=int,   default=100)
    parser.add_argument('--batch_size',         type=int,   default=32)
    parser.add_argument('--lr',                 type=float, default=1e-4)
    parser.add_argument('--weight_decay',       type=float, default=0.05)
    parser.add_argument('--min_lr',             type=float, default=1e-6)
    parser.add_argument('--sr_loss_weight',     type=float, default=0.1)
    parser.add_argument('--diffusion_schedule', type=str,   default='cosine',
                        choices=['linear', 'cosine'])

    # MAE freeze / unfreeze
    parser.add_argument('--freeze_mae',         action='store_true')
    parser.add_argument('--unfreeze_mae_epoch', type=int,   default=50)
    parser.add_argument('--mae_finetune_lr',    type=float, default=2e-5)

    # Resume
    parser.add_argument('--resume_stad_checkpoint', type=str, default='')

    # Misc
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    print("\nLoading datasets from NPZ...")
    train_ds = PreprocessedSEED4Dataset(args.data_path, split='train')
    val_ds   = PreprocessedSEED4Dataset(args.data_path, split='val')
    test_ds  = PreprocessedSEED4Dataset(args.data_path, split='test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # MAE
    mae_model, mae_config = load_mae(args.mae_checkpoint, device, freeze_encoder=args.freeze_mae)

    # Verify latent shape
    with torch.no_grad():
        sample = next(iter(train_loader))['hr'].to(device)
        latents, _, _ = mae_model.forward_encoder(sample, mask_ratio=0.0)
        latents = latents[:, 1:, :]
    num_patches = latents.shape[1]
    print(f"MAE latent shape: {latents.shape}  (num_patches={num_patches})")

    # STAD
    stad_model = STADModel(
        mae_encoder=mae_model,
        lr_channels=16,
        hr_channels=mae_config['in_chans'],
        sr_channels=62,
        latent_dim=mae_config['embed_dim'],
        num_patches=num_patches,
        diffusion_schedule=args.diffusion_schedule,
        lr_channel_indices=train_ds.lr_indices,
        device=device,
    ).to(device)

    # Train
    print("\n" + "="*70)
    print("Training STAD")
    print("="*70)
    train_stad_model(stad_model, train_loader, val_loader, args, device, output_dir)
    print("\n✅ Done.")


if __name__ == '__main__':
    main()

    