import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import copy

from config_seed import Config_MAE_SEED
from dataset_seed import SEEDPretrainDataset, seed_transform
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
            'lr': []
        }
        
        # Create output directory
        os.makedirs(config.output_path, exist_ok=True)
        self.log_file = os.path.join(config.output_path, 'training_log.txt')
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.datetime.now()}\n")
            f.write(f"Configuration:\n")
            f.write(f"  Dataset: SEED (62 channels, 200Hz)\n")
            f.write(f"  Segment length: {config.segment_length} samples\n")
            f.write(f"  LR: {config.lr}, Min LR: {config.min_lr}\n")
            f.write(f"  Batch size: {config.batch_size}\n")
            f.write(f"  Mask ratio: {config.mask_ratio}\n")
            f.write(f"  Epochs: {config.num_epoch}\n")
            f.write("="*60 + "\n\n")
    
    def update(self, epoch, loss, cor, lr):
        """Update metrics and check for improvement"""
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['cor'].append(cor)
        self.history['lr'].append(lr)
        
        # Check for improvement
        improved = False
        if cor > self.best_cor:
            self.best_cor = cor
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1
        
        # Log progress
        msg = f"Epoch {epoch:3d} | Loss: {loss:.4f} | Cor: {cor:.4f} | LR: {lr:.2e}"
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
        axes[0, 1].plot(self.history['epoch'], self.history['cor'], 'g-', linewidth=2)
        axes[0, 1].axvline(x=self.best_epoch, color='r', linestyle='--')
        axes[0, 1].axhline(y=0.45, color='orange', linestyle=':', 
                           label='STAD threshold (0.45)')
        axes[0, 1].axhline(y=0.60, color='purple', linestyle=':', 
                           label='Good threshold (0.60)')
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
        summary += "Training Summary - SEED Dataset\n"
        summary += f"{'='*60}\n"
        summary += f"Best Epoch: {self.best_epoch}\n"
        summary += f"Best Loss: {self.best_loss:.4f}\n"
        summary += f"Best Correlation: {self.best_cor:.4f}\n"
        summary += f"\nStatus for Super-Resolution:\n"
        
        if self.best_cor > 0.60:
            summary += "  ✓ EXCELLENT - Ready for high-quality super-resolution\n"
        elif self.best_cor > 0.45:
            summary += "  ✓ USABLE - Can proceed with super-resolution\n"
        else:
            summary += "  ✗ INSUFFICIENT - Retrain with different hyperparameters\n"
        
        summary += f"{'='*60}\n"
        
        print(summary)
        with open(self.log_file, 'a') as f:
            f.write(summary)


def train_with_monitoring(model, dataloader, optimizer, loss_scaler, config, model_without_ddp):
    """Main training loop with monitoring"""
    
    monitor = TrainingMonitor(config, patience=30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(config.num_epoch):
        # Train one epoch
        current_loss, cor = train_one_epoch(
            model=model,
            data_loader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            config=config,
            model_without_ddp=model_without_ddp
        )
        
        # Get current metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update monitor
        improved = monitor.update(epoch, current_loss, cor, current_lr)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_scaler': loss_scaler.state_dict(),
            'correlation': cor,
            'loss': current_loss,
            'config': config
        }
        
        # Save checkpoint if improved
        if improved:
            torch.save(checkpoint, 
                      os.path.join(config.output_path, 'best_checkpoint.pth'))
        
        # Regular checkpoint every 20 epochs
        if epoch % 20 == 0 or epoch == config.num_epoch - 1:
            torch.save(checkpoint,
                      os.path.join(config.output_path, f'checkpoint_epoch_{epoch}.pth'))
        
        # Check early stopping
        if monitor.should_stop(epoch):
            break
        
        # Check LR sanity
        if current_lr < 1e-7 and epoch > config.warmup_epochs:
            print(f"\n⚠️  Learning rate collapsed to {current_lr:.2e}, stopping")
            break
    
    # Finalize
    monitor.plot_training_curves()
    monitor.save_summary()
    
    return monitor.best_cor, monitor.best_epoch


def main():    
    # Initialize
    config = Config_MAE_SEED()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print("="*60)
    print("SEED MAE Pretraining")
    print("="*60)
    print(f"Output: {config.output_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Channels: {config.num_channels}, Sampling rate: {config.sampling_rate}Hz")
    print(f"Segment length: {config.segment_length} samples ({config.segment_length/config.sampling_rate:.1f}s)")
    print("="*60 + "\n")
    
    # Dataset
    dataset_train = SEEDPretrainDataset(
        data_path=config.data_path,
        split='train',
        num_channels=config.num_channels,
        segment_length=config.segment_length,
        segment_overlap=config.segment_overlap,
        transform=seed_transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset: {len(dataset_train)} segments")
    print(f"Batches per epoch: {len(dataloader)}\n")
    
    # Model - use segment_length instead of time_len
    model = MAEforEEG(
        time_len=config.segment_length,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model_without_ddp = model
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    
    loss_scaler = NativeScaler()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M\n")
    
    # Train with monitoring
    best_cor, best_epoch = train_with_monitoring(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        config=config,
        model_without_ddp=model_without_ddp
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Best correlation: {best_cor:.4f} at epoch {best_epoch}")
    print(f"  Results saved to: {config.output_path}")

if __name__ == '__main__':
    main()
