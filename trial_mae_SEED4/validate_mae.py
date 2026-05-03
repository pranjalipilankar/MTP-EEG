# validate_mae.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from mae_for_eeg import MAEforEEG

def compute_metrics(pred_sr, target_sr, eps=1e-8):
    pred   = pred_sr.reshape(pred_sr.shape[0], -1)
    target = target_sr.reshape(target_sr.shape[0], -1)

    pred_c   = pred   - pred.mean(axis=1, keepdims=True)
    target_c = target - target.mean(axis=1, keepdims=True)

    num = (pred_c * target_c).sum(axis=1)
    den = np.sqrt((pred_c**2).sum(axis=1) * (target_c**2).sum(axis=1) + eps)
    pcc  = (num / den).mean()

    mse  = ((pred - target)**2).mean(axis=1)
    sig  = (target**2).mean(axis=1)
    nmse = (mse / (sig + eps)).mean()
    snr  = (10 * np.log10((sig + eps) / (mse + eps))).mean()

    return pcc, nmse, snr

class HRDataset(Dataset):
    def __init__(self, npz_path, split='val'):
        data = np.load(npz_path)
        idx = data[f'{split}_indices'].astype(int)
        self.hr = data['HR'][idx].astype('float32')
    def __len__(self): return len(self.hr)
    def __getitem__(self, i):
        return {'eeg': torch.from_numpy(self.hr[i])}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
ds = HRDataset('/home/arnav-a5000/MTP-EEG/DATA/preprocessed_data.npz', split='val')
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
print(f"Val samples: {len(ds)}")

# Load model
model = MAEforEEG(
    time_len=1000, patch_size=8, embed_dim=768, in_chans=31,
    depth=12, num_heads=12, decoder_embed_dim=384,
    decoder_depth=4, decoder_num_heads=8, mlp_ratio=4.0
).to(device)

ckpt = torch.load(
    '/home/arnav-a5000/MTP-EEG/trial_mae_SEED4/results_31ch_kfold_fixed/best_model.pth',
    map_location='cpu', weights_only=False
)
state = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt
model.load_state_dict(state)
model.eval()
print(f"✅ Loaded checkpoint (epoch {ckpt.get('epoch')}, val_cor={ckpt.get('val_cor'):.4f})")

# Validate
all_pcc, all_nmse, all_snr, all_loss = [], [], [], []

with torch.no_grad():
    for batch in loader:
        eeg = batch['eeg'].to(device)

        loss, pred, mask = model(eeg, mask_ratio=0.75)
        target = model.patchify(eeg)  # (B, num_patches, C*patch_size)

        # Only masked patches
        mask_np   = mask.cpu().numpy().astype(bool)   # (B, num_patches)
        pred_np   = pred.cpu().numpy()                # (B, num_patches, C*p)
        target_np = target.cpu().numpy()

        # Reconstruct full signals for signal-level metrics
        # Use pred for masked patches, target for unmasked
        full_pred = target_np.copy()
        for i in range(len(eeg)):
            full_pred[i, mask_np[i]] = pred_np[i, mask_np[i]]

        # Unpatchify both
        B, N, Cp = full_pred.shape
        p = 8   # patch_size
        C = Cp // p
        full_pred_sig   = full_pred.reshape(B, N, C, p).transpose(0,2,1,3).reshape(B, C, N*p)
        target_sig      = target_np.reshape(B, N, C, p).transpose(0,2,1,3).reshape(B, C, N*p)

        pcc, nmse, snr = compute_metrics(full_pred_sig, target_sig)
        all_pcc.append(pcc)
        all_nmse.append(nmse)
        all_snr.append(snr)
        all_loss.append(loss.item())

print(f"\n{'='*50}")
print(f"MAE Validation Metrics (masked patches only reconstructed)")
print(f"{'='*50}")
print(f"  Loss : {np.mean(all_loss):.6f}")
print(f"  PCC  : {np.mean(all_pcc):.4f}")
print(f"  NMSE : {np.mean(all_nmse):.4f}")
print(f"  SNR  : {np.mean(all_snr):.2f} dB")