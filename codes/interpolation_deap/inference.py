import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import RBTransformer  # adjust import if needed

################################################################################
# BDE Feature Extraction
################################################################################
def compute_bde_features(eeg_segment):
    """
    Convert raw EEG segment into BDE features.
    eeg_segment: np.array of shape (num_electrodes, time_points)
    returns: np.array of shape (num_electrodes, 4) -> bde_dim=4
    """
    mean_feat = np.mean(eeg_segment, axis=1, keepdims=True)
    std_feat = np.std(eeg_segment, axis=1, keepdims=True)
    max_feat = np.max(eeg_segment, axis=1, keepdims=True)
    min_feat = np.min(eeg_segment, axis=1, keepdims=True)

    bde = np.concatenate([mean_feat, std_feat, max_feat, min_feat], axis=1)
    return bde.astype(np.float32)

################################################################################
# Load Test EEG from .npy files
################################################################################
X_test = np.load("data/test/X_test.npy")  # shape: (num_samples, num_electrodes, time_points)
y_test = np.load("data/test/y_test.npy")  # shape: (num_samples,)

print(f"Loaded test EEG: {X_test.shape}, Labels: {y_test.shape}")

################################################################################
# Compute BDE Features
################################################################################
X_test_bde = np.array([compute_bde_features(seg) for seg in X_test])
print(f"BDE features shape: {X_test_bde.shape}")  # should be (num_samples, num_electrodes, 4)

test_dataset = TensorDataset(torch.tensor(X_test_bde))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

################################################################################
# Load RBTransformer
################################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ELECTRODES = X_test.shape[1]
BDE_DIM = X_test_bde.shape[2]
NUM_CLASSES = 2  # adjust if multi-class

# Load pretrained model config (adjust values if needed)
model = RBTransformer(
    num_electrodes=NUM_ELECTRODES,
    bde_dim=BDE_DIM,
    embed_dim=128,
    depth=4,
    heads=6,
    head_dim=32,
    mlp_hidden_dim=128,
    dropout=0.1,
    num_classes=NUM_CLASSES,
).to(DEVICE)

# Load pretrained weights
weight_path = "deap-binary-dominance-Kfold-2/model.safetensors"  # adjust path
from safetensors.torch import load_file
state_dict = load_file(weight_path, device=str(DEVICE))
model.load_state_dict(state_dict)
print("Loaded pretrained RBTransformer weights.")

################################################################################
# Inference
################################################################################
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(DEVICE)
        logits = model(x)  # shape: (B, num_classes)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

all_preds = np.array(all_preds)
print(f"Predictions shape: {all_preds.shape}")

# Optionally compare with ground truth
if y_test is not None:
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
