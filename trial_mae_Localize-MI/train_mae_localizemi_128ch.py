import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import copy

from config_localizemi_128ch import Config_MAE_LocalizeMI_128ch
from dataset_localizemi import LocalizeMIPretrainDataset, split_dataset, localizemi_transform
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler
from utils import save_model

class TrainingMonitor:
    """Monitor training progress and implement early stopping"""
    
    def __init__(self, config, patience=30):
        self.config = config
        self.patience = patience
        self.best_cor = 0.0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        
        # Tracking
        self.history = {
            'epoch': [],
            'loss': [],
            'cor': [],
            'lr': [],
            'val_loss': [],
            'val_cor': []
        }
        
        # Create output directory
        os.makedirs(config.output_path, exist_ok=True)
        self.log_file = os.path.join(config.output_path, 'training_log.txt')
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.datetime.now()}\n")
            f.write(f"Configuration:\n")
            f.write(f"  Dataset: Localize-MI (128 channels, 8000Hz - downsampled from 256)\n")
            f.write(f"  Downsampling: Every 2nd channel via EGI montage\n")
            f.write(f"  Time length: {config.time_len} samples (~{config.time_len/config.sampling_rate*1000:.1f}ms, 350ms per STAD paper)\n")
            f.write(f"  LR: {config.lr}, Min LR: {config.min_lr}\n")
            f.write(f"  Batch size: {config.batch_size}\n")
            f.write(f"  Mask ratio: {config.mask_ratio}\n")
            f.write(f"  Epochs: {config.num_epoch}\n")
            f.write("="*60 + "\n\n")
    
    def update(self, epoch, loss, cor, lr, val_loss=None, val_cor=None):
        """Update metrics and check for improvement"""
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['cor'].append(cor)
        self.history['lr'].append(lr)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_cor is not None:
            self.history['val_cor'].append(val_cor)
        
        # Check for improvement (use validation metrics if available)
        improved = False
        check_cor = val_cor if val_cor is not None else cor
        check_loss = val_loss if val_loss is not None else loss
        
        if check_cor > self.best_cor:
            self.best_cor = check_cor
            self.best_loss = check_loss
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1
        
        # Log progress
        msg = f"Epoch {epoch:3d} | Loss: {loss:.4f} | Cor: {cor:.4f}"
        if val_loss is not None and val_cor is not None:
            msg += f" | Val Loss: {val_loss:.4f} | Val Cor: {val_cor:.4f}"
        msg += f" | LR: {lr:.2e}"
        if improved:
            msg += " ✓ BEST"
        
        print(msg)
        
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
        
        return improved
    
    def should_stop(self, epoch):
        """Check if training should stop"""
        # Don't stop during warmup
        if epoch < self.config.warmup_epochs:
            return False
        
        # Check patience
        if self.counter >= self.patience:
            msg = f"\n⚠️  Early stopping triggered at epoch {epoch}"
            msg += f"\n   Best correlation: {self.best_cor:.4f} at epoch {self.best_epoch}"
            print(msg)
            with open(self.log_file, 'a') as f:
                f.write(msg + "\n")
            return True
        
        return False
    
    def plot_training_curves(self):
        """Generate training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curve
        axes[0, 0].plot(self.history['epoch'], self.history['loss'], 'b-', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch, color='r', linestyle='--', 
                           label=f'Best (epoch {self.best_epoch})')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Correlation curve
        axes[0, 1].plot(self.history['epoch'], self.history['cor'], 'g-', linewidth=2, label='Train')
        if len(self.history['val_cor']) > 0:
            val_epochs = [self.history['epoch'][i] for i in range(len(self.history['epoch'])) if i < len(self.history['val_cor'])]
            axes[0, 1].plot(val_epochs, self.history['val_cor'], 'b-', linewidth=2, label='Val')
        axes[0, 1].axvline(x=self.best_epoch, color='r', linestyle='--')
        axes[0, 1].axhline(y=0.45, color='orange', linestyle=':', 
                           label='Threshold (0.45)')
        axes[0, 1].axhline(y=0.60, color='purple', linestyle=':', 
                           label='Good (0.60)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].set_title('Reconstruction Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[1, 0].plot(self.history['epoch'], self.history['lr'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss vs Correlation
        sc = axes[1, 1].scatter(self.history['loss'], self.history['cor'], 
                          c=self.history['epoch'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title('Loss vs Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=axes[1, 1])
        cbar.set_label('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_path, 'training_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_summary(self):
        """Save final training summary"""
        summary = f"\n{'='*60}\n"
        summary += "Training Summary - Localize-MI 128-Channel MAE\n"
        summary += f"{'='*60}\n"
        summary += f"Best Epoch: {self.best_epoch}\n"
        summary += f"Best Loss: {self.best_loss:.4f}\n"
        summary += f"Best Correlation: {self.best_cor:.4f}\n"
        summary += f"\nStatus for STAD Super-Resolution (128→256 channels):\n"
        
        if self.best_cor > 0.60:
            summary += "  ✓ EXCELLENT - Ready for high-quality super-resolution\n"
        elif self.best_cor > 0.45:
            summary += "  ✓ USABLE - Can proceed with STAD super-resolution\n"
        else:
            summary += "  ✗ INSUFFICIENT - Retrain with different hyperparameters\n"
        
        summary += f"{'='*60}\n"
        
        print(summary)
        with open(self.log_file, 'a') as f:
            f.write(summary)


class LocalizeMI128ChDataset:
    """Wrapper to downsample 256→128 channels on-the-fly during loading"""
    
    def __init__(self, full_dataset, downsample_factor=2, apply_transform=False, transform=None):
        self.full_dataset = full_dataset
        self.downsample_factor = downsample_factor
        self.apply_transform = apply_transform
        self.transform = transform
    
    def __len__(self):
        return len(self.full_dataset)
    
    def __getitem__(self, idx):
        data = self.full_dataset[idx]
        
        # Extract EEG array from dict
        if isinstance(data, dict):
            eeg_full = data['eeg']
        else:
            eeg_full = data
        
        # Apply transform if enabled (for training augmentation)
        if self.apply_transform and self.transform is not None:
            eeg_full_np = eeg_full.cpu().numpy() if torch.is_tensor(eeg_full) else eeg_full
            eeg_full_np = self.transform(eeg_full_np)
            eeg_full = torch.from_numpy(eeg_full_np).float()
        
        # Downsample channels: select every 2nd channel (256 → 128)
        eeg_128 = eeg_full[::self.downsample_factor, :]
        
        # Return as dict to match trainer.py expectations
        return {'eeg': eeg_128}


def validate_epoch(model, dataloader, device, config):
    """Run validation"""
    model.eval()
    total_loss = 0.0
    total_cor = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            samples = batch['eeg'].to(device, non_blocking=True)
            img_features = None
            valid_idx = None
            
            loss, pred, _ = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)
            
            # Calculate correlation
            pred_flat = pred.detach().cpu().flatten()
            target_flat = samples.detach().cpu().flatten()
            cor = np.corrcoef(pred_flat, target_flat)[0, 1]
            
            total_loss += loss.item()
            total_cor += cor if not np.isnan(cor) else 0.0
            num_batches += 1
    
    model.train()
    return total_loss / num_batches, total_cor / num_batches


def train_with_monitoring(model, train_loader, val_loader, optimizer, loss_scaler, config, model_without_ddp):
    """Main training loop with monitoring"""
    
    monitor = TrainingMonitor(config, patience=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(config.num_epoch):
        # Train one epoch
        current_loss, cor = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            config=config,
            model_without_ddp=model_without_ddp
        )
        
        # Get current metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        # Run validation every 10 epochs
        val_loss = None
        val_cor = None
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_loss, val_cor = validate_epoch(model, val_loader, device, config)
        
        # Update monitor
        improved = monitor.update(epoch, current_loss, cor, current_lr, val_loss, val_cor)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_scaler': loss_scaler.state_dict(),
            'correlation': monitor.best_cor,
            'loss': monitor.best_loss,
            'config': config.__dict__
        }
        
        # Save best checkpoint
        if improved:
            torch.save(checkpoint, os.path.join(config.output_path, 'best_checkpoint.pth'))
            print(f"  ✓ Saved best checkpoint")
        
        # Early stopping
        #if monitor.should_stop(epoch):
        #    break
    
    # Final visualization and summary
    monitor.plot_training_curves()
    monitor.save_summary()
    
    return monitor


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MAE on Localize-MI 128-channel data')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_default = os.path.join(project_root, 'DATA', 'Localize-MI', 'derivatives', 'epochs')
    prc1_default = os.path.join(project_root, 'DATA', 'Localize-MI', 'derivatives', 'epochs_prc1')

    parser.add_argument('--data-mode', default='raw', choices=['raw', 'prc1'],
                        help='Data source mode: raw Localize-MI epochs or PrC-1 preprocessed outputs')
    parser.add_argument('--data-path', default=None,
                        help='Path to Localize-MI dataset')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory. If omitted, uses mode-specific default.')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============================================
    # Configuration
    # ============================================
    config = Config_MAE_LocalizeMI_128ch()
    config.data_path = prc1_default if args.data_mode == 'prc1' else raw_default
    if args.data_path is not None:
        config.data_path = args.data_path
    config.num_epoch = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    if args.output_dir is not None:
        config.output_path = args.output_dir
    else:
        mode_tag = 'prc1' if args.data_mode == 'prc1' else 'raw'
        config.output_path = os.path.join(config.root_path, f'trial_mae_Localize-MI/results_128ch_{mode_tag}')

    # Ensure output directory exists before any artifact write (e.g., split indices).
    os.makedirs(config.output_path, exist_ok=True)
    
    print("="*60)
    print("Localize-MI 128-Channel MAE Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data mode: {args.data_mode}")
    print(f"Dataset: {config.data_path}")
    print(f"Channels: {config.original_channels} → {config.num_channels} (via EGI montage)")
    print(f"Epoch length: {config.time_len} samples ({config.time_len/config.sampling_rate*1000:.1f}ms @ {config.sampling_rate}Hz)")
    print(f"Batch size: {config.batch_size}")
    print(f"Output: {config.output_path}")
    print("="*60 + "\n")
    
    # ============================================
    # Dataset
    # ============================================
    print("Loading dataset...")
    
    # Load base dataset WITHOUT transform (will apply separately to train/val)
    base_dataset = LocalizeMIPretrainDataset(
        data_path=config.data_path,
        time_len=config.time_len,
        transform=None,  # No transform in base dataset
        orig_fs=8000,  # Original Localize-MI sampling rate
        target_fs=config.sampling_rate  # Target sampling rate from config
    )
    print(f"Loaded {len(base_dataset)} epochs from {config.data_path}")
    
    # RANDOM split (not sequential!) to avoid subject bias
    np.random.seed(42)
    total_size = len(base_dataset)
    indices = np.random.permutation(total_size)
    
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Save split indices for downstream processing (e.g., STAD)
    split_file = os.path.join(config.output_path, 'dataset_split_indices.npz')
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    np.savez(split_file, 
             train_indices=train_indices, 
             val_indices=val_indices, 
             test_indices=test_indices,
             seed=42,
             total_size=total_size)
    print(f"Saved split indices to {split_file}")
    
    # Create subsets
    train_base = Subset(base_dataset, train_indices)
    val_base = Subset(base_dataset, val_indices)
    
    # Wrap: train WITH augmentation, val WITHOUT
    train_dataset = LocalizeMI128ChDataset(
        train_base, 
        downsample_factor=config.downsample_factor,
        apply_transform=True,  # Enable augmentation for training
        transform=localizemi_transform
    )
    val_dataset = LocalizeMI128ChDataset(
        val_base, 
        downsample_factor=config.downsample_factor,
        apply_transform=False,  # No augmentation for validation
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Train: {len(train_dataset)} epochs (with augmentation)")
    print(f"Val:   {len(val_dataset)} epochs (no augmentation)")
    print(f"Test:  {len(test_indices)} epochs")
    print(f"Split: Random (not sequential) to avoid subject bias\n")
    
    # ============================================
    # Model
    # ============================================
    print("Creating model...")
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,  # 128 channels
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=8,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        norm_layer=torch.nn.LayerNorm,
        mask_ratio=config.mask_ratio
    ).to(device)
    
    model_without_ddp = model
    print(f"Model created with {config.num_channels} input channels")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # ============================================
    # Optimizer and Loss Scaler
    # ============================================
    param_groups = [
        {'params': [p for n, p in model_without_ddp.named_parameters() 
                   if p.requires_grad], 'lr': config.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    loss_scaler = NativeScaler()
    
    # ============================================
    # Training
    # ============================================
    print("Starting training...\n")
    monitor = train_with_monitoring(model, train_loader, val_loader, optimizer, loss_scaler, config, model_without_ddp)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best checkpoint saved to: {config.output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
