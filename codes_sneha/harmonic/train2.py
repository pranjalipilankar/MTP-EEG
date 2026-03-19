#!/usr/bin/env python3
"""
train_cond_transformer.py (diffusion-conditioned + metrics)

Now includes:
 - AMP (mixed precision)
 - gradient clipping
 - OneCycleLR scheduler
 - early stopping
 - full RMSE, MAE, nRMSE(range), nRMSE(std)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from encoder import EEGEncoder
from decoder import EEGDecoder
from cond_diffusion_transformer import SpatioTemporalConditionedTransformer
from channel_ops import downsample_channels_average, upsample_channels_linear
from spatio_temporal_condition import SpatioTemporalConditionNet

# -------------------------- utilities --------------------------

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def compute_graph_harmonics(chan_pos, k=8):
    """
    chan_pos: (C, 2) torch or numpy
    returns: (C, k) harmonic embedding
    """
    if isinstance(chan_pos, torch.Tensor):
        chan_pos = chan_pos.cpu().numpy()

    # 1) Build adjacency via k-NN
    A = kneighbors_graph(chan_pos, n_neighbors=4, mode='connectivity', include_self=False)
    A = A.toarray()

    # 2) Graph Laplacian
    L = csgraph.laplacian(A, normed=True)

    # 3) Eigen-decompose
    vals, vecs = np.linalg.eigh(L)

    # 4) Take smallest non-constant eigenvectors (skip eigenvector 0)
    harmonic_basis = vecs[:, 1:k+1]   # shape (C, k)

    return torch.tensor(harmonic_basis, dtype=torch.float32)


def load_eeg_channel_positions(locs_path):
    positions = []
    with open(locs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # parts: index, x, y, label
            x = float(parts[1])
            y = float(parts[2])
            positions.append([x, y])
    positions = np.array(positions, dtype=np.float32)

    # normalize to unit circle scale (optional, recommended)
    positions = positions / np.max(np.abs(positions))
    return torch.tensor(positions, dtype=torch.float32)

def bandpass_filter(data, low=1.0, high=40.0, fs=200.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)

def upsample_latents_to_channels(z, target_channels=62):
    if z.dim() != 3:
        raise ValueError("z must be (B, N, D)")
    B, N, D = z.shape
    if N == target_channels:
        return z
    z_t = z.permute(0, 2, 1)
    z_up = F.interpolate(z_t, size=target_channels, mode='linear', align_corners=True)
    z_up = z_up.permute(0, 2, 1)
    return z_up

# -------------------------- Dataset --------------------------
class EEGDataset(Dataset):
    def __init__(self, data_root, window_size=400, fs=200, mode="denoise"):
        self.mode = mode
        self.window_size = window_size
        self.fs = fs
        self.samples, self.subject_ids = self.load_all_subjects(data_root, window_size, fs)


    def segment_eeg_blocks(self, eeg_data, window_size=400):
        n_blocks, n_channels, n_points = eeg_data.shape
        windows = []
        for b in range(n_blocks):
            for start in range(0, n_points - window_size + 1, window_size):
                seg = eeg_data[b, :, start:start + window_size]
                windows.append(seg)
        return np.stack(windows)

    def normalize_per_channel(self, X):
        n_segments, n_channels, n_time = X.shape
        Xn = np.zeros_like(X)
        for ch in range(n_channels):
            scaler = StandardScaler()
            Xn[:, ch, :] = scaler.fit_transform(X[:, ch, :])
        return Xn

    def preprocess_subject(self, eeg, fs):
        n_blocks, n_channels, n_points = eeg.shape
        filtered = np.zeros_like(eeg)
        for b in range(n_blocks):
            filtered[b] = bandpass_filter(eeg[b], low=1, high=40, fs=fs)
        return filtered

    def load_all_subjects(self, data_root, window_size, fs):
        all_segments = []
        all_subject_ids = []
        subj_id = 0
        for fname in sorted(os.listdir(data_root)):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(data_root, fname)
            eeg = np.load(path)
            print(f"📂 Loading {fname}  shape={eeg.shape}")

            # -------- KEEP ONLY LAST 50% OF DATA --------
            half = eeg.shape[-1] // 2
            eeg = eeg[..., half:]
            print(f"✂️  Using only last 50% → new shape={eeg.shape}")
            # --------------------------------------------

            eeg = self.preprocess_subject(eeg, fs)
            eeg_segments = self.segment_eeg_blocks(eeg, window_size)
            eeg_segments = self.normalize_per_channel(eeg_segments)

            all_segments.append(eeg_segments)
            # record subject id for every segment from this file
            all_subject_ids.extend([subj_id] * len(eeg_segments))
            subj_id += 1

        all_segments = np.concatenate(all_segments, axis=0)
        all_subject_ids = np.array(all_subject_ids, dtype=np.int64)
        print(f"✅ Loaded {all_segments.shape[0]} segments from all subjects.")
        return all_segments.astype(np.float32), all_subject_ids
    


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        seg = self.samples[idx]
        seg_t = torch.tensor(seg, dtype=torch.float32)
        seg_31 = downsample_channels_average(seg_t)
        seg_interp = upsample_channels_linear(seg_31)

        if self.mode == "denoise":
            inp = seg_interp
            target = seg_t
        elif self.mode == "completion":
            inp = seg_31
            target = seg_t
        else:
            raise ValueError("mode must be 'denoise' or 'completion'")

        subj_id = int(self.subject_ids[idx])
        return inp, target, subj_id

# -------------------------- Training procedure --------------------------
def train_model(data_root="/home/ab_students/EEG-MTP/DATA/SEED",
                mode="denoise",
                num_epochs=10,
                batch_size=16,
                lr=1e-4,
                weight_decay=1e-4,
                max_lr=3e-4,
                patience=6,
                save_path=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device} | mode={mode}")

    dataset = EEGDataset(data_root, window_size=400, fs=200, mode=mode)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training samples: {len(train_ds)}, Validation: {len(val_ds)}")

    encoder_in_channels = 62
    decoder_out_channels = 62
    cond_channels = 62     # condition on full noisy EEG
    
    n_time = 400
    latent_dim = 128
    n_conditions = 8
    n_channels_transformer = 62

    encoder = EEGEncoder(in_channels=encoder_in_channels,latent_dim=latent_dim,seq_len=n_time,n_layers=4,n_heads=8,dropout=0.1,pool_factor=4,checkpoint_segments=4,).to(device)

    condition_net = SpatioTemporalConditionNet(n_channels=cond_channels,model_dim=latent_dim,n_conditions=n_conditions).to(device)

    transformer = SpatioTemporalConditionedTransformer(latent_dim=latent_dim,n_channels=decoder_out_channels,n_layers=4,n_heads=8).to(device)

    decoder = EEGDecoder(latent_dim=latent_dim,out_channels=decoder_out_channels,seq_len=n_time).to(device)
    
    criterion = nn.MSELoss()
    params = list(encoder.parameters()) + list(condition_net.parameters()) + list(transformer.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=num_epochs, anneal_strategy='cos')

    scaler = torch.amp.GradScaler('cuda',enabled=(device != "cpu"))

    train_losses, val_losses, val_rmses, val_maes, val_nrmse_ranges, val_nrmse_stds = [], [], [], [], [], []
    best_val_loss = float("inf")
    patience_counter = 0
    if save_path is None:
        save_path = f"eeg_conditioned_model_diff_{mode}.pt"

    locs_path = "/home/ab_students/EEG-MTP/DATA/SEED/channel_62_pos.locs"
    chan_pos = load_eeg_channel_positions(locs_path).to(device)   # shape (62, 2)
    num_diffusion_steps = 1000
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print("🚀 Starting training...")

    for epoch in range(num_epochs):
        encoder.train(); transformer.train(); decoder.train(); condition_net.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inp, target, subj_id = batch
            inp, target = inp.to(device), target.to(device)
            # subj_id is already a tensor of shape (B,) or a list -- ensure tensor
            if not isinstance(subj_id, torch.Tensor):
                subj_id = torch.tensor(subj_id, dtype=torch.long, device=device)
            else:
                subj_id = subj_id.to(device).long()

            optimizer.zero_grad()

            # --- Create per-batch conditioning vars ---
            B = target.size(0)
            t_steps = torch.randint(0, num_diffusion_steps, (B,), device=device)
            cond_c = subj_id  # (B,) long tensor


            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
                # Single encoder pass (avoid double encoding)
                z, _ = encoder(inp)
                if z.shape[1] != n_channels_transformer:
                    z = upsample_latents_to_channels(z, target_channels=n_channels_transformer)
                z.requires_grad_(True)

                # Construct noisy target for diffusion training
                alpha_t = alphas_cumprod[t_steps].view(B, 1, 1)
                #noise = torch.randn_like(target)

                # Conditioning on noisy target (standard denoising objective)
                # Step 1: Construct noisy input (DDPM-style)
                alpha_t = alphas_cumprod[t_steps].view(B, 1, 1)
                #noise = torch.randn_like(target)                # target noise
                #x_t = alpha_t.sqrt() * target + (1 - alpha_t).sqrt() * noise

                # Step 2: Conditioning
                cond_tokens, cond_pooled = condition_net(inp, chan_pos, t_steps, cond_c)

                # Step 3: Transformer + Decoder
                z_refined = transformer(z, cond_tokens, cond_pooled, t_steps)
                pred_noise = decoder(z_refined)

                # Step 4: Compute loss on noise, NOT target
                loss = F.mse_loss(pred_noise, target)




            # Backprop with AMP + gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Ensure scheduler.step is called after optimizer.step
            try:
                scheduler.step()
            except Exception:
                pass

            # sync and accumulate
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # free transient memory before validation
        torch.cuda.empty_cache()

        # ---------------- VALIDATION ----------------
        encoder.eval(); transformer.eval(); decoder.eval(); condition_net.eval()
        val_loss, val_rmse, val_mae, val_nrmse_range, val_nrmse_std = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inp, target, subj_id = batch
                inp, target = inp.to(device), target.to(device)
                if not isinstance(subj_id, torch.Tensor):
                    subj_id = torch.tensor(subj_id, dtype=torch.long, device=device)
                else:
                    subj_id = subj_id.to(device).long()

                # per-batch conditioning vars (handles last smaller batch)
                B = target.size(0)
                t_steps = torch.randint(0, num_diffusion_steps, (B,), device=device)
                cond_c = subj_id


                with torch.amp.autocast('cuda', enabled=(device != "cpu")):
                    z, _ = encoder(inp)
                    if z.shape[1] != n_channels_transformer:
                        z = upsample_latents_to_channels(z, target_channels=n_channels_transformer)
                    
                    cond_tokens, cond_pooled = condition_net(inp, chan_pos, t_steps, cond_c)

                    # Use same noisy x_t as in training
                    #alpha_t = alphas_cumprod[t_steps].view(B, 1, 1)
                    #noise = torch.randn_like(target)
                    #x_t = alpha_t.sqrt() * target + (1 - alpha_t).sqrt() * noise

                    cond_tokens, cond_pooled = condition_net(inp, chan_pos, t_steps, cond_c)
                    #z_refined = transformer(z, cond_tokens, cond_pooled, t_steps)
                    pred_noise = decoder(transformer(z, cond_tokens, cond_pooled, t_steps))

                    # Reconstruct clean signal
                    recon = (target - pred_noise * (1 - alphas_cumprod[t_steps].view(B, 1, 1))**0.5)
                    loss = criterion(recon, target)


                val_loss += loss.item()

                # metrics per-sample
                diff = target - recon
                mse = torch.mean(diff ** 2, dim=(1, 2))
                mae = torch.mean(torch.abs(diff), dim=(1, 2))
                rmse = torch.sqrt(mse)

                target_range = (torch.max(target, dim=2).values - torch.min(target, dim=2).values).mean(dim=1) + 1e-8
                target_std = torch.std(target, dim=(1, 2)) + 1e-8
                nrmse_range = rmse / target_range
                nrmse_std = rmse / target_std

                val_rmse += rmse.sum().item()
                val_mae += mae.sum().item()
                val_nrmse_range += nrmse_range.sum().item()
                val_nrmse_std += nrmse_std.sum().item()

        # finalize validation stats
        n_val_samples = len(val_ds)
        val_loss /= len(val_loader)
        val_rmse /= n_val_samples
        val_mae /= n_val_samples
        val_nrmse_range /= n_val_samples
        val_nrmse_std /= n_val_samples

        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        val_nrmse_ranges.append(val_nrmse_range)
        val_nrmse_stds.append(val_nrmse_std)

        # cleanup large tensors and free GPU memory
        try:
            del inp, target, z, cond_tokens, cond_pooled, recon, pred_noise, noise
        except Exception:
            pass
        torch.cuda.empty_cache()

        # epoch summary & checkpointing
        print(f"Epoch {epoch+1:03d}: TrainLoss={train_loss:.6f} | ValLoss_eps={val_loss:.6f} "
              f"| ValReconRMSE={val_rmse:.6f} | ValMAE={val_mae:.6f} "
              f"| nRMSE(range)={val_nrmse_range:.6f} | nRMSE(std)={val_nrmse_std:.6f}")

        # small GPU memory diagnostic
        if torch.cuda.is_available():
            print(f"[GPU] allocated={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

        # early stopping / save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "condition_net": condition_net.state_dict(),
                "transformer": transformer.state_dict(),
                "decoder": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "mode": mode
            }, save_path)
            print(f"✅ Saved best model (val_loss={val_loss:.6f}) → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    print("🎉 Training complete.")
    print(f"🏆 Best Validation Loss: {best_val_loss:.6f}")

    # Save metrics & plots (unchanged)
    os.makedirs("results", exist_ok=True)
    epochs_arr = np.arange(1, len(train_losses) + 1)
    np.savez(f"results/training_metrics_{mode}.npz",
             epochs=epochs_arr,
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             val_rmses=np.array(val_rmses),
             val_maes=np.array(val_maes),
             val_nrmse_range=np.array(val_nrmse_ranges),
             val_nrmse_std=np.array(val_nrmse_stds))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_arr, train_losses, 'b-o', label="Train Loss")
    plt.plot(epochs_arr, val_losses, 'r-o', label="Val Loss")
    plt.title(f"Training & Validation Loss (MSE) [{mode}]")
    plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs_arr, val_rmses, 'g-o', label="RMSE")
    plt.plot(epochs_arr, val_maes, 'm-o', label="MAE")
    plt.plot(epochs_arr, val_nrmse_ranges, 'c-o', label="nRMSE(range)")
    plt.plot(epochs_arr, val_nrmse_stds, 'y-o', label="nRMSE(std)")
    plt.title("Validation Metrics per Epoch")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/training_metrics_{mode}.png", dpi=300)
    plt.close()
    print(f"📈 Saved training plot → results/training_metrics_{mode}.png")

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conditioned EEG transformer (with full metrics)")
    parser.add_argument("--data_root", type=str, default="/home/ab_students/EEG-MTP/DATA/SEED_processed/train")
    parser.add_argument("--mode", type=str, choices=["denoise", "completion"], default="denoise")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    train_model(data_root=args.data_root,
                mode=args.mode,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_lr=args.max_lr,
                patience=args.patience)