#!/usr/bin/env python3
"""
Train MAE on Localize-MI (128-channel downsampled) with subject-based K-fold CV.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_localizemi_128ch import Config_MAE_LocalizeMI_128ch
from dataset_localizemi import LocalizeMIPretrainDataset, localizemi_transform
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler


class LocalizeMI128ChDataset:
    """Wrapper to downsample 256->128 channels on the fly."""

    def __init__(self, full_dataset, downsample_factor=2, apply_transform=False, transform=None):
        self.full_dataset = full_dataset
        self.downsample_factor = downsample_factor
        self.apply_transform = apply_transform
        self.transform = transform

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, idx):
        data = self.full_dataset[idx]

        if isinstance(data, dict):
            eeg_full = data["eeg"]
        else:
            eeg_full = data

        if self.apply_transform and self.transform is not None:
            eeg_full_np = eeg_full.cpu().numpy() if torch.is_tensor(eeg_full) else eeg_full
            eeg_full_np = self.transform(eeg_full_np)
            eeg_full = torch.from_numpy(eeg_full_np).float()

        eeg_128 = eeg_full[:: self.downsample_factor, :]
        return {"eeg": eeg_128}


def create_subject_kfold_splits(metadata, n_folds=5, seed=42):
    """Create subject-wise folds so train/val subjects never overlap."""
    subject_ids = np.array([item["subject"] for item in metadata])
    unique_subjects = np.array(sorted(np.unique(subject_ids)))

    if n_folds > len(unique_subjects):
        raise ValueError(
            f"n_folds={n_folds} is larger than number of subjects={len(unique_subjects)}"
        )

    rng = np.random.default_rng(seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    val_subject_folds = np.array_split(shuffled_subjects, n_folds)

    splits = []
    all_indices = np.arange(len(metadata))

    for fold_idx, val_subjects in enumerate(val_subject_folds):
        train_subjects = np.array([s for s in shuffled_subjects if s not in set(val_subjects)])

        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)

        train_indices = all_indices[train_mask]
        val_indices = all_indices[val_mask]

        splits.append(
            {
                "fold": fold_idx + 1,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
            }
        )

    return splits


def validate_epoch(model, dataloader, device, config):
    """Validation with masked-patch reconstruction correlation."""
    model.eval()

    total_loss = 0.0
    all_correlations = []

    with torch.no_grad():
        for batch in dataloader:
            samples = batch["eeg"].to(device, non_blocking=True)
            img_features = None
            valid_idx = None

            loss, pred, mask = model(
                samples,
                img_features,
                valid_idx=valid_idx,
                mask_ratio=config.mask_ratio,
            )

            target = model.patchify(samples)

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                masked_patches = mask_np[i] == 1
                if masked_patches.sum() == 0:
                    continue

                pred_masked = pred_np[i][masked_patches].flatten()
                target_masked = target_np[i][masked_patches].flatten()

                if pred_masked.size > 1:
                    pred_mean = pred_masked.mean()
                    target_mean = target_masked.mean()
                    numerator = ((pred_masked - pred_mean) * (target_masked - target_mean)).sum()
                    denominator = np.sqrt(
                        ((pred_masked - pred_mean) ** 2).sum()
                        * ((target_masked - target_mean) ** 2).sum()
                    )
                    if denominator > 1e-8:
                        all_correlations.append(numerator / denominator)

            total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    avg_corr = float(np.mean(all_correlations)) if all_correlations else 0.0

    model.train()
    return avg_loss, avg_corr


def build_model(config, device):
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=8,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        norm_layer=torch.nn.LayerNorm,
        mask_ratio=config.mask_ratio,
    ).to(device)
    return model


def train_fold(fold_info, base_dataset, args, config, device, output_dir):
    fold_num = fold_info["fold"]

    print("=" * 80)
    print(f"Fold {fold_num}/{args.n_folds}")
    print("=" * 80)
    print(f"Train subjects: {sorted(fold_info['train_subjects'].tolist())}")
    print(f"Val subjects:   {sorted(fold_info['val_subjects'].tolist())}")

    train_base = Subset(base_dataset, fold_info["train_indices"])
    val_base = Subset(base_dataset, fold_info["val_indices"])

    train_dataset = LocalizeMI128ChDataset(
        train_base,
        downsample_factor=config.downsample_factor,
        apply_transform=True,
        transform=localizemi_transform,
    )
    val_dataset = LocalizeMI128ChDataset(
        val_base,
        downsample_factor=config.downsample_factor,
        apply_transform=False,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    model = build_model(config, device)
    model_without_ddp = model

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        [{"params": [p for _, p in model_without_ddp.named_parameters() if p.requires_grad], "lr": config.lr}],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_scaler = NativeScaler()

    fold_dir = output_dir / f"fold_{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_val_cor = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(config.num_epoch):
        train_loss, train_cor = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            config=config,
            model_without_ddp=model_without_ddp,
        )

        val_loss, val_cor = validate_epoch(model, val_loader, device, config)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{config.num_epoch} | "
            f"Train Loss {train_loss:.4f} | Train Cor {train_cor:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Cor {val_cor:.4f} | LR {current_lr:.2e}"
        )

        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "train_cor": float(train_cor),
                "val_loss": float(val_loss),
                "val_cor": float(val_cor),
                "lr": float(current_lr),
            }
        )

        if val_cor > best_val_cor:
            best_val_cor = float(val_cor)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch + 1)

            checkpoint = {
                "fold": int(fold_num),
                "epoch": int(epoch + 1),
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_scaler": loss_scaler.state_dict(),
                "val_loss": float(val_loss),
                "val_cor": float(val_cor),
                "train_subjects": fold_info["train_subjects"].tolist(),
                "val_subjects": fold_info["val_subjects"].tolist(),
                "config": config.__dict__,
                "args": vars(args),
            }
            torch.save(checkpoint, fold_dir / "best_model.pth")
            print(f"Saved best checkpoint for fold {fold_num}")

    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "fold": int(fold_num),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_cor": float(best_val_cor),
        "train_subjects": fold_info["train_subjects"].tolist(),
        "val_subjects": fold_info["val_subjects"].tolist(),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train MAE on Localize-MI 128-channel data with K-fold CV")
    project_root = Path(__file__).resolve().parents[1]
    raw_default = project_root / "DATA" / "Localize-MI" / "derivatives" / "epochs"
    prc1_default = project_root / "DATA" / "Localize-MI" / "derivatives" / "epochs_prc1"

    parser.add_argument(
        "--data-mode",
        default="prc1",
        choices=["raw", "prc1"],
        help="Data source mode: raw Localize-MI epochs or PrC-1 preprocessed epochs",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to Localize-MI data directory (raw epochs or PrC-1 outputs)",
    )
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs per fold")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--n-folds", default=5, type=int, help="Number of subject-wise folds")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num-workers", default=4, type=int, help="Dataloader workers")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, uses config output path with _kfold suffix.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config_MAE_LocalizeMI_128ch()
    config.data_path = str(prc1_default if args.data_mode == "prc1" else raw_default)
    if args.data_path is not None:
        config.data_path = args.data_path
    config.num_epoch = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    if args.output_dir is not None:
        config.output_path = args.output_dir
    else:
        base_output = str(config.output_path).rstrip("/")
        mode_tag = "prc1" if args.data_mode == "prc1" else "raw"
        config.output_path = f"{base_output}_kfold_{mode_tag}"

    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Localize-MI 128-Channel MAE K-Fold Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data mode: {args.data_mode}")
    print(f"Data path: {config.data_path}")
    print(f"Folds: {args.n_folds}")
    print(f"Epochs/fold: {config.num_epoch}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    base_dataset = LocalizeMIPretrainDataset(
        data_path=config.data_path,
        time_len=config.time_len,
        transform=None,
        orig_fs=8000,
        target_fs=config.sampling_rate,
    )

    # Persist sample-level metadata for downstream subject-aware processing.
    sample_metadata = []
    for i, item in enumerate(base_dataset.metadata):
        sample_metadata.append(
            {
                "sample_index": int(i),
                "subject": item.get("subject", ""),
                "run": item.get("run", ""),
                "epoch_idx": int(item.get("epoch_idx", i)),
                "source_file": item.get("source_file", ""),
                "source_epoch_idx": int(item.get("source_epoch_idx", item.get("epoch_idx", i))),
            }
        )
    with open(output_dir / "sample_metadata.json", "w") as f:
        json.dump(sample_metadata, f, indent=2)

    splits = create_subject_kfold_splits(base_dataset.metadata, n_folds=args.n_folds, seed=args.seed)

    serializable_splits = [
        {
            "fold": s["fold"],
            "train_subjects": s["train_subjects"].tolist(),
            "val_subjects": s["val_subjects"].tolist(),
            "train_samples": int(len(s["train_indices"])),
            "val_samples": int(len(s["val_indices"])),
        }
        for s in splits
    ]
    with open(output_dir / "fold_splits.json", "w") as f:
        json.dump(serializable_splits, f, indent=2)

    np.savez(
        output_dir / "fold_indices.npz",
        **{f"fold_{s['fold']}_train_indices": s["train_indices"] for s in splits},
        **{f"fold_{s['fold']}_val_indices": s["val_indices"] for s in splits},
    )

    fold_results = []
    for fold_info in splits:
        result = train_fold(
            fold_info=fold_info,
            base_dataset=base_dataset,
            args=args,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        fold_results.append(result)

    avg_val_loss = float(np.mean([r["best_val_loss"] for r in fold_results]))
    std_val_loss = float(np.std([r["best_val_loss"] for r in fold_results]))
    avg_val_cor = float(np.mean([r["best_val_cor"] for r in fold_results]))
    std_val_cor = float(np.std([r["best_val_cor"] for r in fold_results]))

    print("\n" + "=" * 80)
    print("K-FOLD SUMMARY (LOCALIZE-MI 128CH)")
    print("=" * 80)
    for r in fold_results:
        print(
            f"Fold {r['fold']}: Best Val Loss = {r['best_val_loss']:.6f}, "
            f"Best Val Cor = {r['best_val_cor']:.4f} (Epoch {r['best_epoch']})"
        )
    print(
        f"\nAverage: Val Loss = {avg_val_loss:.6f} +- {std_val_loss:.6f}, "
        f"Val Cor = {avg_val_cor:.4f} +- {std_val_cor:.4f}"
    )

    summary = {
        "n_folds": int(args.n_folds),
        "average_val_loss": avg_val_loss,
        "std_val_loss": std_val_loss,
        "average_val_cor": avg_val_cor,
        "std_val_cor": std_val_cor,
        "fold_results": fold_results,
        "args": vars(args),
    }
    with open(output_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
