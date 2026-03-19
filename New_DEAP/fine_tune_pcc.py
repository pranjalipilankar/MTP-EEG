#!/usr/bin/env python3
"""
Fine-tune STAD Decoder with PCC Loss (NaN-SAFE VERSION)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_stad_complete import STAD, STADDataset, get_channel_positions, get_beta_schedule, get_diffusion_params

class PCCLoss(nn.Module):
    """
    Stable Pearson Correlation Coefficient Loss
    """
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, T) predicted signals
            target: (B, C, T) ground truth signals
        Returns:
            loss: 1 - mean PCC (lower is better)
        """
        # Always work in float32 for stability
        pred = pred.float()
        target = target.float()
        
        # Check for NaN/Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("WARNING: NaN/Inf in pred")
            return torch.tensor(1.0, device=pred.device)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("WARNING: NaN/Inf in target")
            return torch.tensor(1.0, device=pred.device)
        
        # Normalize
        pred_mean = pred.mean(dim=2, keepdim=True)
        target_mean = target.mean(dim=2, keepdim=True)
        
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        
        # Compute correlation
        numerator = (pred_centered * target_centered).sum(dim=2)
        
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=2) + self.eps)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=2) + self.eps)
        denominator = pred_std * target_std + self.eps
        
        pcc = torch.clamp(numerator / denominator, -1.0, 1.0)
        
        if self.reduction == 'mean':
            return 1.0 - pcc.mean()
        elif self.reduction == 'none':
            return 1.0 - pcc
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class CombinedLoss(nn.Module):
    """
    Stable Combined loss
    """
    def __init__(self, mse_weight=1.0, pcc_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pcc = PCCLoss()
        self.mse_weight = mse_weight
        self.pcc_weight = pcc_weight
    
    def forward(self, pred, target):
        # Convert to float32 for stability
        pred = pred.float()
        target = target.float()
        
        # Clamp pred to reasonable range
        pred = torch.clamp(pred, -10.0, 10.0)
        
        loss_mse = self.mse(pred, target)
        loss_pcc = self.pcc(pred, target)
        
        total_loss = (self.mse_weight * loss_mse + 
                      self.pcc_weight * loss_pcc)
        
        return total_loss, {
            'mse': loss_mse.item(),
            'pcc': loss_pcc.item(),
            'total': total_loss.item()
        }

def reconstruct_for_training(model, x_lr, diff_params, device, steps=10):
    """
    Fast reconstruction for training (fewer steps)
    """
    with torch.no_grad():
        B = x_lr.shape[0]
        lr_pos = get_channel_positions(16, device, B)
        
        # Start from noise
        zt = torch.randn(B, 100, 256, device=device)
        
        timesteps = torch.linspace(999, 0, steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            cond_tokens, cond_pooled = model.stc(x_lr, lr_pos, t_batch)
            pred_noise = model.mtd(zt, t_batch, cond_tokens, cond_pooled)
            
            alpha_t = diff_params['sqrt_alphas_cumprod'][t] ** 2
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = diff_params['sqrt_alphas_cumprod'][t_prev] ** 2
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            pred_x0 = (zt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            if i < len(timesteps) - 1:
                zt = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                zt = pred_x0
        
        return zt

def finetune_decoder_with_pcc(
    checkpoint_path='best_stad_DIMENSION_FIXED2.pt',
    dataset_path='/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
    num_epochs=50,
    batch_size=16,
    lr=5e-5,  # Lower learning rate for stability
    output_path='best_stad_pcc_finetuned.pt'
):
    """
    Fine-tune only the MAE decoder with PCC loss
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Fine-tuning STAD Decoder with PCC Loss (NaN-Safe)")
    print(f"   Device: {device}")
    
    # Load datasets
    train_dataset = STADDataset(dataset_path, 'train', window_size=400)
    val_dataset = STADDataset(dataset_path, 'val', window_size=400)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load model
    print("📥 Loading pretrained STAD model...")
    model = STAD(
        lr_channels=16,
        hr_channels=32,
        seq_len=400,
        latent_dim=256,
        n_harmonics=8
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"✅ Loaded from epoch {checkpoint['epoch']}")
    
    # Freeze everything except MAE decoder
    print("\n🔒 Freezing components...")
    trainable_names = []
    
    for name, param in model.named_parameters():
        if 'decoder' in name.lower() or 'decoder_pred' in name:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
    
    print(f"\n✅ Trainable parameters: {len(trainable_names)}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Diffusion params
    betas = get_beta_schedule(1000).to(device)
    diff_params = get_diffusion_params(betas)
    
    # Loss and optimizer (NO spectral loss - was causing NaN)
    criterion = CombinedLoss(mse_weight=1.0, pcc_weight=1.0)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
        eps=1e-8  # Increase epsilon for stability
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)
    
    best_pcc_loss = float('inf')
    best_actual_pcc = 0.0
    
    print(f"\n🚀 Starting fine-tuning for {num_epochs} epochs...\n")
    print(f"   Learning rate: {lr}")
    print(f"   Batch size: {batch_size}")
    print(f"   Loss weights: MSE=1.0, PCC=1.0\n")
    
    for epoch in range(num_epochs):
        # ==================== TRAINING ====================
        model.train()
        
        # Set frozen modules to eval
        model.stc.eval()
        model.mtd.eval()
        for name, module in model.mae.named_modules():
            if 'decoder' not in name.lower():
                module.eval()
        
        train_losses = {'mse': 0, 'pcc': 0, 'total': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x_lr, y_hr) in enumerate(pbar):
            x_lr, y_hr = x_lr.to(device), y_hr.to(device)
            B = x_lr.size(0)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Get latent (no gradient)
            latent = reconstruct_for_training(model, x_lr, diff_params, device, steps=10)
            
            # Decode (with gradient, in float32 for stability)
            cls_token = model.mae.cls_token.expand(B, -1, -1)
            latent_with_cls = torch.cat([cls_token, latent], dim=1)
            pred_patches = model.mae.decode_full(latent_with_cls)
            pred_eeg = model.mae.unpatchify(pred_patches)
            
            # Convert to float32
            pred_eeg = pred_eeg.float()
            y_hr = y_hr.float()
            
            # Compute loss
            loss, loss_dict = criterion(pred_eeg, y_hr)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping (aggressive)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if not np.isnan(v) and not np.isinf(v):
                    train_losses[k] += v
            
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'pcc': loss_dict['pcc'],
                'mse': loss_dict['mse']
            })
        
        # Average training losses
        for k in train_losses:
            train_losses[k] /= len(train_loader)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_losses = {'mse': 0, 'pcc': 0, 'total': 0}
        
        with torch.no_grad():
            for x_lr, y_hr in val_loader:
                x_lr, y_hr = x_lr.to(device), y_hr.to(device)
                B = x_lr.size(0)
                
                latent = reconstruct_for_training(model, x_lr, diff_params, device, steps=20)
                
                cls_token = model.mae.cls_token.expand(B, -1, -1)
                latent_with_cls = torch.cat([cls_token, latent], dim=1)
                pred_patches = model.mae.decode_full(latent_with_cls)
                pred_eeg = model.mae.unpatchify(pred_patches)
                
                loss, loss_dict = criterion(pred_eeg, y_hr)
                
                for k, v in loss_dict.items():
                    if not np.isnan(v) and not np.isinf(v):
                        val_losses[k] += v
        
        # Average validation losses
        for k in val_losses:
            val_losses[k] /= len(val_loader)
        
        scheduler.step()
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        print(f"TRAIN - Total: {train_losses['total']:.6f} | PCC: {train_losses['pcc']:.6f} | MSE: {train_losses['mse']:.6f}")
        print(f"VAL   - Total: {val_losses['total']:.6f} | PCC: {val_losses['pcc']:.6f} | MSE: {val_losses['mse']:.6f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        actual_pcc = 1.0 - val_losses['pcc']
        print(f"Actual PCC: {actual_pcc:.4f}")
        print(f"{'='*80}\n")
        
        # Save best model
        if val_losses['pcc'] < best_pcc_loss and not np.isnan(val_losses['pcc']):
            best_pcc_loss = val_losses['pcc']
            best_actual_pcc = actual_pcc
            
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'actual_pcc': actual_pcc,
                'diff_params': diff_params
            }, output_path)
            
            print(f"✅ SAVED BEST MODEL (PCC={actual_pcc:.4f})\n")
    
    print(f"\n🎉 Fine-tuning complete!")
    print(f"Best validation PCC: {best_actual_pcc:.4f}")
    print(f"Model saved to: {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()
    
    finetune_decoder_with_pcc(
        checkpoint_path='best_stad_DIMENSION_FIXED2.pt',
        dataset_path='/home/ab_students/EEG-MTP/codes/DEAP_split_dataset.npz',
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path='best_stad_pcc_finetuned.pt'
    )