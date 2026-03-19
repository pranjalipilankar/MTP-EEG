
#!/usr/bin/env python3
"""
Quick test to validate MAE model forward pass and correlation computation
without full training loop
"""

import torch
import numpy as np
from scipy.stats import pearsonr

from config_seed4 import Config_MAE_SEED4
from mae_for_eeg import MAEforEEG
from dataset_seed4_kfold import create_kfold_dataloaders

def test_model_forward():
    """Test model forward pass with dummy data"""
    print("="*80)
    print("Testing MAE Model Forward Pass")
    print("="*80)
    
    # Initialize config
    config = Config_MAE_SEED4()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    print("\n1. Building MAE model...")
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model built with {n_params:,} parameters")
    
    # Test with dummy data
    print("\n2. Testing forward pass with dummy data...")
    batch_size = 4
    dummy_eeg = torch.randn(batch_size, config.num_channels, config.time_len).to(device)
    print(f"   Input shape: {dummy_eeg.shape}")
    
    model.eval()
    with torch.no_grad():
        try:
            loss, pred, mask = model(dummy_eeg, mask_ratio=config.mask_ratio)
            print(f"   ✓ Forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Pred shape: {pred.shape}")
            print(f"   Mask shape: {mask.shape if mask is not None else 'None'}")
            
            # Check if pred shape matches input
            if pred.shape == dummy_eeg.shape:
                print(f"   ✓ Output shape matches input shape!")
            else:
                print(f"   ✗ ERROR: Shape mismatch!")
                print(f"     Expected: {dummy_eeg.shape}, Got: {pred.shape}")
                return False
                
        except Exception as e:
            print(f"   ✗ Forward pass FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test correlation computation
    print("\n3. Testing correlation computation...")
    pred_np = pred.cpu().numpy()
    target_np = dummy_eeg.cpu().numpy()
    
    # Flatten
    pred_flat = pred_np.reshape(-1)
    target_flat = target_np.reshape(-1)
    
    print(f"   Flattened shape: {pred_flat.shape}")
    print(f"   Pred stats: mean={pred_flat.mean():.4f}, std={pred_flat.std():.4f}")
    print(f"   Target stats: mean={target_flat.mean():.4f}, std={target_flat.std():.4f}")
    
    # Compute correlation
    if len(pred_flat) > 100:
        correlation, _ = pearsonr(pred_flat, target_flat)
        print(f"   ✓ Correlation computed: {correlation:.6f}")
    else:
        print(f"   ✗ Not enough data points for correlation")
        return False
    
    return True


def test_with_real_data():
    """Test with actual SEED-IV data"""
    print("\n" + "="*80)
    print("Testing with Real SEED-IV Data")
    print("="*80)
    
    config = Config_MAE_SEED4()
    config.n_folds = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load first fold
    print("\n1. Loading first fold of SEED-IV data...")
    try:
        train_loader, val_loader, test_loader, fold_info = create_kfold_dataloaders(
            config=config,
            fold_idx=0,
            num_workers=0,  # Single-threaded for testing
            pin_memory=False,
            verbose=True
        )
        print(f"   ✓ Data loaded successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"   ✗ Data loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create model
    print("\n2. Building model...")
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    ).to(device)
    print(f"   ✓ Model built")
    
    # Test one batch
    print("\n3. Testing one training batch...")
    model.eval()
    batch = next(iter(train_loader))
    eeg = batch['eeg'].to(device)
    
    print(f"   Batch shape: {eeg.shape}")
    print(f"   Batch stats: mean={eeg.mean():.4f}, std={eeg.std():.4f}")
    
    with torch.no_grad():
        try:
            loss, pred, mask = model(eeg, mask_ratio=config.mask_ratio)
            print(f"   ✓ Forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Pred shape: {pred.shape}")
            
            # Compute correlation
            pred_np = pred.cpu().numpy()
            target_np = eeg.cpu().numpy()
            
            pred_flat = pred_np.reshape(-1)
            target_flat = target_np.reshape(-1)
            
            # Sample if too large
            if len(pred_flat) > 100000:
                idx = np.random.choice(len(pred_flat), 100000, replace=False)
                pred_flat = pred_flat[idx]
                target_flat = target_flat[idx]
            
            correlation, _ = pearsonr(pred_flat, target_flat)
            print(f"   ✓ Correlation: {correlation:.6f}")
            
            # Per-sample correlation
            sample_cors = []
            for p, t in zip(pred_np[:10], target_np[:10]):
                p_f = p.reshape(-1)
                t_f = t.reshape(-1)
                if np.std(p_f) > 1e-6 and np.std(t_f) > 1e-6:
                    cor, _ = pearsonr(p_f, t_f)
                    sample_cors.append(cor)
            
            if sample_cors:
                print(f"   ✓ Per-sample correlation (first 10): {np.mean(sample_cors):.6f}")
            
        except Exception as e:
            print(f"   ✗ Test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test validation batch
    print("\n4. Testing one validation batch...")
    batch = next(iter(val_loader))
    eeg = batch['eeg'].to(device)
    
    with torch.no_grad():
        try:
            loss, pred, mask = model(eeg, mask_ratio=config.mask_ratio)
            print(f"   ✓ Validation forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            
            # Compute correlation
            pred_np = pred.cpu().numpy()
            target_np = eeg.cpu().numpy()
            
            pred_flat = pred_np.reshape(-1)
            target_flat = target_np.reshape(-1)
            
            correlation, _ = pearsonr(pred_flat[:100000], target_flat[:100000])
            print(f"   ✓ Correlation: {correlation:.6f}")
            
        except Exception as e:
            print(f"   ✗ Validation test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_unpatchify():
    """Specifically test unpatchify operation"""
    print("\n" + "="*80)
    print("Testing Unpatchify Operation")
    print("="*80)
    
    config = Config_MAE_SEED4()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MAEforEEG(
        time_len=config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=config.num_channels,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio
    ).to(device)
    
    # Test patchify -> unpatchify roundtrip
    print("\n1. Testing patchify -> unpatchify roundtrip...")
    x = torch.randn(2, config.num_channels, config.time_len).to(device)
    print(f"   Original shape: {x.shape}")
    
    patches = model.patchify(x)
    print(f"   Patched shape: {patches.shape}")
    
    reconstructed = model.unpatchify(patches)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    
    if reconstructed.shape == x.shape:
        print(f"   ✓ Shapes match!")
        
        # Check reconstruction error
        mse = ((x - reconstructed) ** 2).mean().item()
        print(f"   Reconstruction MSE: {mse:.10f}")
        
        if mse < 1e-6:
            print(f"   ✓ Perfect reconstruction!")
        else:
            print(f"   ⚠️  Reconstruction error detected")
            
        return True
    else:
        print(f"   ✗ Shape mismatch!")
        return False


if __name__ == '__main__':
    print("\n" + "🧪 QUICK MAE MODEL TESTS" + "\n")
    
    # Run tests
    test1 = test_model_forward()
    test2 = test_unpatchify()
    test3 = test_with_real_data()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Dummy data forward pass:  {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"  Unpatchify operation:     {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"  Real data forward pass:   {'✓ PASS' if test3 else '✗ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n✅ All tests passed! Model is ready for training.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before training.")
    
    print("="*80 + "\n")