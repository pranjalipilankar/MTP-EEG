import torch
from pathlib import Path

ckpt_path = Path("/home/arnav-a5000/MTP-EEG/New_LocalizeMI/original.pt") # change this
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

cfg = ckpt.get("config", {})
mae_ckpt = str(cfg.get("mae_checkpoint", ""))
print("checkpoint:", ckpt_path)
print("epoch:", ckpt.get("epoch"))
print("mae_fold:", cfg.get("mae_fold"))
print("mae_checkpoint:", mae_ckpt)
print("train_subjects:", cfg.get("train_subjects"))
print("val_subjects:", cfg.get("val_subjects"))

if "prc1" in mae_ckpt.lower():
    print("\nLikely source: PREPROCESSED training (X_prc1).")
elif "results_128ch_kfold" in mae_ckpt and "prc1" not in mae_ckpt.lower():
    print("\nLikely source: RAW-epochs kfold training.")
else:
    print("\nCould not confidently infer raw vs preprocessed from mae_checkpoint path.")
