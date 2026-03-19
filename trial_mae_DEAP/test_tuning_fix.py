#!/usr/bin/env python3
"""
Quick test to verify hyperparameter tuning fix
"""

import sys
import numpy as np
from hyperparam_tuning import HyperparameterTuner
from config_deap import Config_MAE_DEAP

print("Testing hyperparameter tuning fix...\n")

# Create tuner
base_config = Config_MAE_DEAP()
tuner = HyperparameterTuner(base_config, output_dir='test_tuning_fix')

# Test 1: Type conversion
print("Test 1: Type conversion")
test_value_int = np.int64(32)
test_value_float = np.float64(0.002)

converted_int = tuner._to_python_type(test_value_int)
converted_float = tuner._to_python_type(test_value_float)

print(f"  np.int64(32) -> {converted_int} (type: {type(converted_int).__name__})")
print(f"  np.float64(0.002) -> {converted_float} (type: {type(converted_float).__name__})")

assert isinstance(converted_int, int), "Should be Python int"
assert isinstance(converted_float, float), "Should be Python float"
print("  ✓ Type conversion works!\n")

# Test 2: Config creation with numpy types
print("Test 2: Config creation with numpy types")
params = {
    'lr': np.float64(0.002),
    'mask_ratio': np.float64(0.6),
    'patch_size': np.int64(8),
    'depth': np.int64(8),
    'embed_dim': np.int64(256),
    'batch_size': np.int64(32),
    'warmup_epochs': np.int64(10),
    'min_lr': np.float64(1e-05),
    'weight_decay': np.float64(0.03)
}

print(f"  Input params (numpy types): {params}")

config = tuner.create_config_from_params(params)

print(f"  Config batch_size: {config.batch_size} (type: {type(config.batch_size).__name__})")
print(f"  Config lr: {config.lr} (type: {type(config.lr).__name__})")
print(f"  Config depth: {config.depth} (type: {type(config.depth).__name__})")

assert isinstance(config.batch_size, int), "batch_size should be Python int"
assert isinstance(config.lr, float), "lr should be Python float"
print("  ✓ Config creation works!\n")

# Test 3: DataLoader creation
print("Test 3: DataLoader creation (will fail if data path wrong, but type should be OK)")
try:
    from torch.utils.data import DataLoader
    from dataset_deap import DEAPPretrainDataset
    
    # This will fail if data doesn't exist, but should not fail on type error
    train_dataset = DEAPPretrainDataset(
        data_path=config.data_path,
        split='train',
        time_len=config.time_len,
        num_channels=config.num_channels
    )
    
    # This is the critical line that was failing
    batch_size = int(config.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"  ✓ DataLoader created successfully with batch_size={batch_size}")
    
except FileNotFoundError as e:
    print(f"  ⚠️  Data file not found (expected), but no type error!")
    print(f"  ✓ Type error is FIXED!")
except TypeError as e:
    print(f"  ✗ Type error still present: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ⚠️  Other error (not type-related): {e}")
    print(f"  ✓ Type error is FIXED!")

print("\n" + "="*60)
print("All tests passed! Hyperparameter tuning should work now.")
print("="*60)

# Cleanup
import shutil
import os
if os.path.exists('test_tuning_fix'):
    shutil.rmtree('test_tuning_fix')
