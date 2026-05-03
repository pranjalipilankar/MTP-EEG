#!/usr/bin/env python3
"""
STAD Training for SEED-IV Dataset (62 channels, 250Hz)
Runs on preprocessed_data.npz with keys: LR, HR, SR, labels, train_indices, val_indices, test_indices
Loss: MSE + PSD + PCC
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
# GradNorm (Chen et al. 2018)
# Adaptively reweights losses by normalizing gradient magnitudes.
# Ensures each loss contributes equally to the shared backbone.
# -------------------------------------------------------------
class GradNormWeights(nn.Module):
    """
    Learnable loss weights updated via GradNorm algorithm.
    Maintains weights for: [diff, mse, psd, pcc]
    """
    def __init__(self, n_losses=4, alpha=1.5):
        super().__init__()
        # Initialise weights to 1.0
        self.weights  = nn.Parameter(torch.ones(n_losses))
        self.alpha    = alpha       # restoring force strength (1.0-2.0 typical)
        self.n_losses = n_losses
        self.L0       = None        # initial loss values (set on first call)

    def forward(self, losses):
        """
        Weighted sum of losses for the main backward pass.
        GradNorm weight update is done separately in update_weights().

        Args:
            losses: list of n_losses scalar tensors
        Returns:
            total:   weighted sum
            weights: current weights for logging
        """
        w = torch.abs(self.weights)  # keep positive
        total = sum(w[i] * losses[i] for i in range(self.n_losses))
        return total, [w[i].item() for i in range(self.n_losses)]

    @torch.no_grad()
    def update_weights(self, losses, shared_layer, optimizer_weights):
        """
        Compute GradNorm loss and update weights.
        Call AFTER the main backward pass, BEFORE optimizer.step().

        Args:
            losses:           list of current scalar loss tensors (detached)
            shared_layer:     last shared layer whose grad norms are measured
                              (e.g. stad_model.mtd's final linear layer weight)
            optimizer_weights: separate optimizer for self.weights
        """
        # Initialise L0 on first call
        if self.L0 is None:
            self.L0 = torch.tensor(
                [l.item() for l in losses], dtype=torch.float32
            ).to(losses[0].device)

        w = torch.abs(self.weights)

        # Compute per-loss gradient norms w.r.t. shared layer
        G_norms = []
        for i, loss in enumerate(losses):
            # Grad of weighted loss w.r.t. shared params
            g = torch.autograd.grad(
                w[i] * loss,
                shared_layer,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            if g is not None:
                G_norms.append(g.norm())
            else:
                G_norms.append(torch.tensor(0.0, device=w.device))

        G_norms  = torch.stack(G_norms)           # (n_losses,)
        G_mean   = G_norms.mean().detach()         # target norm

        # Relative inverse training rates
        L_cur  = torch.tensor([l.item() for l in losses], device=w.device)
        L_hat  = L_cur / (self.L0 + 1e-8)         # loss ratio
        r_hat  = L_hat / (L_hat.mean() + 1e-8)    # relative training rate

        # GradNorm target
        G_target = (G_mean * r_hat ** self.alpha).detach()

        # GradNorm loss
        gradnorm_loss = torch.abs(G_norms - G_target).sum()

        optimizer_weights.zero_grad()
        gradnorm_loss.backward()
        optimizer_weights.step()

        # Renormalize weights so they sum to n_losses
        with torch.no_grad():
            self.weights.data = (
                torch.abs(self.weights) / torch.abs(self.weights).sum() * self.n_losses
            )


# -------------------------------------------------------------
# Combined Loss: MSE + PSD + PCC
# -------------------------------------------------------------
def psd_loss(pred, target, fs=250, eps=1e-8):
    """
    Power Spectral Density loss.
    Computes FFT of pred and target, compares magnitude spectra.

    Args:
        pred:   (B, C, T)
        target: (B, C, T)
        fs:     sampling frequency (Hz)
    Returns:
        scalar loss
    """
    pred_fft   = torch.fft.rfft(pred,   dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    pred_psd   = pred_fft.abs().pow(2)
    target_psd = target_fft.abs().pow(2)

    # Log-scale comparison for better dynamic range
    pred_psd_log   = torch.log(pred_psd   + eps)
    target_psd_log = torch.log(target_psd + eps)

    return F.mse_loss(pred_psd_log, target_psd_log)


def pcc_loss(pred, target, eps=1e-8):
    """
    Pearson Correlation Coefficient loss (1 - PCC).
    Encourages high linear correlation between pred and target.

    Args:
        pred:   (B, C, T)
        target: (B, C, T)
    Returns:
        scalar loss in [0, 2]
    """
    B, C, T = pred.shape
    pred_flat   = pred.reshape(B * C, T)
    target_flat = target.reshape(B * C, T)

    pred_c   = pred_flat   - pred_flat.mean(dim=1, keepdim=True)
    target_c = target_flat - target_flat.mean(dim=1, keepdim=True)

    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt(pred_c.pow(2).sum(dim=1) * target_c.pow(2).sum(dim=1) + eps)
    pcc = (num / den).mean()

    return 1.0 - pcc  # minimise → maximise PCC


def combined_loss(pred_sr, sr, diff_loss, gradnorm):
    """
    Weighted loss using GradNorm weights.

    Args:
        pred_sr:  (B, 62, T)
        sr:       (B, 62, T)
        diff_loss: scalar
        gradnorm: GradNormWeights module
    Returns:
        total, mse, psd, pcc_l, weights
    """
    pred_f = pred_sr.float()
    sr_f   = sr.float()

    mse   = F.mse_loss(pred_f, sr_f)
    psd   = psd_loss(pred_f, sr_f)
    pcc_l = pcc_loss(pred_f, sr_f)

    total, weights = gradnorm([diff_loss, mse, psd, pcc_l])
    return total, mse, psd, pcc_l, weights


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
    def __init__(self, npz_path, split='train', lr_channels=16, hr_channels=31, sr_channels=62):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        payload = np.load(npz_path, allow_pickle=True)
        sr_all  = payload['SR'].astype(np.float32)
        N       = len(sr_all)

        if f'{split}_indices' in payload.files:
            indices = payload[f'{split}_indices'].astype(int)
        else:
            from sklearn.model_selection import train_test_split
            all_idx = np.arange(N)
            train_idx, rest_idx = train_test_split(all_idx, test_size=0.3, random_state=2024)
            val_idx, test_idx   = train_test_split(rest_idx, test_size=0.5, random_state=2024)
            indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}[split]

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
    # GradNorm adaptive loss weighting
    gradnorm = GradNormWeights(n_losses=4, alpha=args.gradnorm_alpha).to(device)

    # Shared layer for GradNorm — last linear layer of MTD
    shared_layer = stad_model.mtd.layers[-1].ff[-1].weight  # adjust if MTD structure differs

    trainable = [p for p in stad_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Separate optimizer for GradNorm weights (higher lr, no weight decay)
    optimizer_weights = torch.optim.Adam(gradnorm.parameters(), lr=args.gradnorm_lr)
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
        t_total, t_diff, t_mse, t_psd, t_pcc_l = [], [], [], [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            sr = batch['sr'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                diff_loss, pred_sr                   = stad_model(lr, hr, sr)
                total, mse, psd, pcc_l, eff_weights  = combined_loss(pred_sr, sr, diff_loss, gradnorm)

            if not torch.isfinite(total):
                continue

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in stad_model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            # GradNorm weight update (after main backward)
            with torch.amp.autocast('cuda', enabled=False):
                diff_loss2, pred_sr2 = stad_model(lr, hr, sr)
                mse2   = F.mse_loss(pred_sr2.float(), sr.float())
                psd2   = psd_loss(pred_sr2.float(), sr.float())
                pcc_l2 = pcc_loss(pred_sr2.float(), sr.float())
                gradnorm.update_weights(
                    [diff_loss2, mse2, psd2, pcc_l2],
                    shared_layer,
                    optimizer_weights
                )

            t_total.append(total.item())
            t_diff.append(diff_loss.item())
            t_mse.append(mse.item())
            t_psd.append(psd.item())
            t_pcc_l.append(pcc_l.item())

        # ---- Validate ----
        stad_model.eval()
        v_total, v_diff, v_mse, v_psd, v_pcc_l = [], [], [], [], []
        v_pcc, v_nmse, v_snr = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                sr = batch['sr'].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    diff_loss, pred_sr                    = stad_model(lr, hr, sr)
                    total, mse, psd, pcc_l, eff_weights   = combined_loss(pred_sr, sr, diff_loss, adaptive_weights)

                if not torch.isfinite(total):
                    continue

                metrics = compute_sr_metrics(pred_sr.float(), sr.float())
                v_total.append(total.item())
                v_diff.append(diff_loss.item())
                v_mse.append(mse.item())
                v_psd.append(psd.item())
                v_pcc_l.append(pcc_l.item())
                v_pcc.append(metrics['pcc'])
                v_nmse.append(metrics['nmse'])
                v_snr.append(metrics['snr'])

        scheduler.step()

        def _m(lst, default=float('inf')):
            return float(np.mean(lst)) if lst else default

        tr   = _m(t_total); td = _m(t_diff); tm = _m(t_mse)
        tp   = _m(t_psd);   tpc = _m(t_pcc_l)
        vl   = _m(v_total); vd = _m(v_diff); vm = _m(v_mse)
        vp   = _m(v_psd);   vpc = _m(v_pcc_l)
        pcc  = _m(v_pcc,  0.0); nmse = _m(v_nmse); snr = _m(v_snr, -float('inf'))

        # Current GradNorm weights
        w = [torch.abs(gradnorm.weights[i]).item() for i in range(4)]

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: {tr:.4f} (Diff {td:.4f} MSE {tm:.4f} PSD {tp:.4f} PCC {tpc:.4f}) | "
            f"Val:   {vl:.4f} (Diff {vd:.4f} MSE {vm:.4f} PSD {vp:.4f} PCC {vpc:.4f}) | "
            f"PCC: {pcc:.4f} | NMSE: {nmse:.4f} | SNR: {snr:.2f} dB | "
            f"GradNorm W [diff={w[0]:.3f} mse={w[1]:.3f} psd={w[2]:.3f} pcc={w[3]:.3f}]"
        )

        history.append({
            'epoch': epoch+1,
            'train_total': tr, 'train_diff': td, 'train_mse': tm,
            'train_psd': tp,   'train_pcc_loss': tpc,
            'val_total': vl,   'val_diff': vd,   'val_mse': vm,
            'val_psd': vp,     'val_pcc_loss': vpc,
            'val_pcc': pcc, 'val_nmse': nmse, 'val_snr_db': snr,
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        payload = {
            'epoch': epoch+1,
            'model_state_dict': stad_model.state_dict(),
            'best_val_loss': best_val_loss,
            'val_total_loss': vl,
            'val_pcc': pcc,
            'val_nmse': nmse,
            'val_snr_db': snr,
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
    parser.add_argument('--diffusion_schedule', type=str,   default='cosine',
                        choices=['linear', 'cosine'])

    # Loss weights — removed fixed weights, replaced by GradNorm
    parser.add_argument('--gradnorm_alpha', type=float, default=1.5,
                        help='GradNorm restoring force (0.0=equal weights, higher=more aggressive)')
    parser.add_argument('--gradnorm_lr',    type=float, default=1e-3,
                        help='Learning rate for GradNorm weight optimizer')

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
    print(f"GradNorm alpha: {args.gradnorm_alpha} | GradNorm lr: {args.gradnorm_lr}")

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