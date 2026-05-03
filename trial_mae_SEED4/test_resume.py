# test_resume.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from mae_for_eeg import MAEforEEG

# ---- Load checkpoint ----
ckpt_path = '/home/arnav-a5000/MTP-EEG/trial_mae_SEED4/results_31ch_kfold_fixed/best_model.pth'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print("Checkpoint keys:", ckpt.keys())
print("Saved epoch:", ckpt.get('epoch'))
print("Val cor:", ckpt.get('val_cor'))

# ---- Dataset ----
class HRDataset(Dataset):
    def __init__(self, npz_path):
        import torch
        data = np.load(npz_path)
        print("NPZ keys:", data.files)
        idx = data['train_indices'].astype(int)
        self.hr = data['HR'][idx].astype('float32')
    def __len__(self): return len(self.hr)
    def __getitem__(self, i):
        return {'eeg': torch.from_numpy(self.hr[i])}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = HRDataset('/home/arnav-a5000/MTP-EEG/DATA/preprocessed_data.npz')
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

# ---- Model ----
model = MAEforEEG(
    time_len=1000, patch_size=8, embed_dim=768, in_chans=31,
    depth=12, num_heads=12, decoder_embed_dim=384,
    decoder_depth=4, decoder_num_heads=8, mlp_ratio=4.0
).to(device)

state = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt
model.load_state_dict(state)
print("✅ Weights loaded")

# ---- 1 epoch ----
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
model.train()
for batch in loader:
    eeg = batch['eeg'].to(device)
    optimizer.zero_grad()
    loss, pred, mask = model(eeg, mask_ratio=0.75)
    loss.backward()
    optimizer.step()
    print(f"  batch loss: {loss.item():.6f}")

# ---- Save ----
torch.save({
    'model': model.state_dict(),
    'epoch': ckpt.get('epoch', 0) + 1,
}, '/home/arnav-a5000/MTP-EEG/trial_mae_SEED4/results_31ch_kfold_fixed/resumed_model.pth')
print("✅ Saved resumed_model.pth")