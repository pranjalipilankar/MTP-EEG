import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import itertools
from datetime import datetime
import pandas as pd

from config_deap import Config_MAE_DEAP
from dataset_deap import DEAPPretrainDataset, deap_transform
from mae_for_eeg import MAEforEEG
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    Add weight decay to optimizer, but skip certain parameters
    Typically skip biases and normalization layer parameters
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip biases, layer norms, and anything in skip_list
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class HyperparameterTuner:
    """Systematic hyperparameter tuning for MAE on DEAP"""
    
    def __init__(self, base_config, output_dir='tuning_results'):
        self.base_config = base_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = []
        self.results_file = os.path.join(output_dir, 'tuning_results.csv')
        self.log_file = os.path.join(output_dir, 'tuning_log.txt')
        
        # Initialize log
        with open(self.log_file, 'w') as f:
            f.write(f"Hyperparameter Tuning Started: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
    
    def define_search_space(self, strategy='focused'):
        """
        Define hyperparameter search space
        
        Args:
            strategy: 'quick', 'focused', or 'exhaustive'
        """
        if strategy == 'quick':
            # Fast exploration of critical parameters
            search_space = {
                'lr': [1e-3, 2e-3, 3e-3],
                'mask_ratio': [0.4, 0.5, 0.6],
                'patch_size': [16, 32],
                'depth': [8, 12],
                'embed_dim': [256, 512],
                'batch_size': [32, 64],
                'warmup_epochs': [10],
                'min_lr': [1e-5],
                'weight_decay': [0.01, 0.03]
            }
        
        elif strategy == 'focused':
            # Focused on likely optimal ranges based on EEG characteristics
            search_space = {
                'lr': [1e-3, 1.5e-3, 2e-3, 3e-3],
                'mask_ratio': [0.4, 0.5, 0.6, 0.7],
                'patch_size': [8, 16, 32],
                'depth': [8, 10, 12],
                'embed_dim': [256, 384, 512],
                'batch_size': [32, 64],
                'warmup_epochs': [5, 10, 15],
                'min_lr': [1e-6, 1e-5],
                'weight_decay': [0.01, 0.02, 0.03]
            }
        
        else:  # exhaustive
            search_space = {
                'lr': [3e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4],
                'mask_ratio': [0.3, 0.4, 0.5, 0.6, 0.7, 0.75],
                'patch_size': [8, 16, 24, 32],
                'depth': [6, 8, 12, 16, 20],
                'embed_dim': [256, 384, 512, 768, 1024],
                'batch_size': [16, 32, 64],
                'warmup_epochs': [5, 10, 15, 20],
                'min_lr': [0, 1e-6, 1e-5],
                'weight_decay': [0.001, 0.01, 0.03, 0.05, 0.1]
            }
        
        return search_space
    
    def _to_python_type(self, value):
        """Convert numpy types to Python native types"""
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        else:
            return value
    
    def create_config_from_params(self, params):
        """Create a config object with specified parameters"""
        config = Config_MAE_DEAP()
        
        # Update with search parameters (ensure Python native types)
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, self._to_python_type(value))
        
        # Ensure decoder is proportional to encoder
        if 'embed_dim' in params:
            config.decoder_embed_dim = int(params['embed_dim']) // 2
        
        if 'depth' in params:
            config.decoder_depth = max(4, int(params['depth']) // 3)
        
        # Adjust num_heads based on embed_dim (must be divisible)
        if 'embed_dim' in params:
            embed_dim = int(params['embed_dim'])
            # Find closest divisor for num_heads
            if embed_dim >= 512:
                config.num_heads = embed_dim // 64
            elif embed_dim >= 256:
                config.num_heads = embed_dim // 64 if embed_dim % 64 == 0 else embed_dim // 32
            else:
                config.num_heads = 4
            
            config.decoder_num_heads = max(2, config.decoder_embed_dim // 64)
        
        return config
    
    def quick_train_eval(self, config, trial_epochs=30, device='cuda'):
        """
        Quick training evaluation for hyperparameter search
        
        Returns:
            dict with final loss, correlation, and training stability metrics
        """
        try:
            # Setup
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            
            # Data
            train_dataset = DEAPPretrainDataset(
                data_path=config.data_path,
                split='train',
                time_len=config.time_len,
                num_channels=config.num_channels,
                transform=lambda x: deap_transform(x, sparse_rate=config.sparse_rate)
            )
            
            # Ensure batch_size is Python int
            batch_size = int(config.batch_size)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            # Model
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
                mlp_ratio=config.mlp_ratio,
                focus_range=config.focus_range,
                focus_rate=config.focus_rate,
                img_recon_weight=config.img_recon_weight,
                use_nature_img_loss=config.use_nature_img_loss
            ).to(device)
            
            # Optimizer
            param_groups = add_weight_decay(model, config.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
            loss_scaler = NativeScaler()
            
            # Track metrics
            losses = []
            correlations = []
            
            # Quick training
            for epoch in range(trial_epochs):
                loss, cor = train_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    loss_scaler=loss_scaler,
                    config=config,
                    model_without_ddp=model
                )
                
                losses.append(loss)
                correlations.append(cor)
                
                # Early exit if clearly failing
                if epoch > 10 and cor < 0.01:
                    print(f"  Early exit: correlation too low ({cor:.4f})")
                    break
                
                if epoch > 10 and np.isnan(loss):
                    print(f"  Early exit: NaN loss")
                    break
            
            # Compute metrics
            final_loss = losses[-1] if losses else float('inf')
            final_cor = correlations[-1] if correlations else 0.0
            max_cor = max(correlations) if correlations else 0.0
            
            # Stability metrics
            loss_std = np.std(losses[-10:]) if len(losses) >= 10 else float('inf')
            cor_trend = correlations[-1] - correlations[min(5, len(correlations)//2)] if len(correlations) > 5 else 0
            
            # Clean up
            del model, optimizer, loss_scaler, train_loader, train_dataset
            torch.cuda.empty_cache()
            
            return {
                'final_loss': final_loss,
                'final_cor': final_cor,
                'max_cor': max_cor,
                'loss_std': loss_std,
                'cor_trend': cor_trend,
                'epochs_run': len(losses),
                'success': not np.isnan(final_loss) and final_cor > 0.01
            }
            
        except Exception as e:
            print(f"  Training failed with error: {e}")
            return {
                'final_loss': float('inf'),
                'final_cor': 0.0,
                'max_cor': 0.0,
                'loss_std': float('inf'),
                'cor_trend': 0.0,
                'epochs_run': 0,
                'success': False,
                'error': str(e)
            }
    
    def random_search(self, search_space, n_trials=20, trial_epochs=30):
        """Random search over hyperparameter space"""
        print(f"\n{'='*80}")
        print(f"Starting Random Search: {n_trials} trials, {trial_epochs} epochs each")
        print(f"{'='*80}\n")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for trial in range(n_trials):
            # Sample random configuration (convert numpy types to Python native)
            params = {key: self._to_python_type(np.random.choice(values))
                     for key, values in search_space.items()}
            
            print(f"\nTrial {trial+1}/{n_trials}")
            print(f"Parameters: {params}")
            
            # Create config and evaluate
            config = self.create_config_from_params(params)
            results = self.quick_train_eval(config, trial_epochs=trial_epochs, device=device)
            
            # Store results
            result_dict = {**params, **results, 'trial': trial}
            self.results.append(result_dict)
            
            # Log
            print(f"Results: Loss={results['final_loss']:.4f}, "
                  f"Cor={results['final_cor']:.4f}, "
                  f"Success={results['success']}")
            
            self._save_results()
            self._log_trial(trial, params, results)
        
        return self.get_best_configs()
    
    def grid_search(self, search_space, max_combinations=50, trial_epochs=30):
        """Grid search (limited to max_combinations)"""
        print(f"\n{'='*80}")
        print(f"Starting Grid Search (max {max_combinations} combinations)")
        print(f"{'='*80}\n")
        
        # Generate all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        all_combinations = list(itertools.product(*values))
        
        # Limit combinations
        if len(all_combinations) > max_combinations:
            print(f"Total combinations: {len(all_combinations)}, sampling {max_combinations}")
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for idx, combo in enumerate(combinations):
            # Convert to Python native types
            params = {key: self._to_python_type(value) 
                     for key, value in zip(keys, combo)}
            
            print(f"\nCombination {idx+1}/{len(combinations)}")
            print(f"Parameters: {params}")
            
            # Create config and evaluate
            config = self.create_config_from_params(params)
            results = self.quick_train_eval(config, trial_epochs=trial_epochs, device=device)
            
            # Store results
            result_dict = {**params, **results, 'trial': idx}
            self.results.append(result_dict)
            
            # Log
            print(f"Results: Loss={results['final_loss']:.4f}, "
                  f"Cor={results['final_cor']:.4f}, "
                  f"Success={results['success']}")
            
            self._save_results()
            self._log_trial(idx, params, results)
        
        return self.get_best_configs()
    
    def _save_results(self):
        """Save results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
    
    def _log_trial(self, trial, params, results):
        """Log trial to file"""
        with open(self.log_file, 'a') as f:
            f.write(f"\nTrial {trial}\n")
            f.write(f"Parameters: {params}\n")
            f.write(f"Results: {results}\n")
            f.write("-"*80 + "\n")
    
    def get_best_configs(self, top_k=5, metric='final_cor'):
        """Get top-k best configurations"""
        if not self.results:
            return []
        
        df = pd.DataFrame(self.results)
        
        # Filter successful runs
        df_success = df[df['success'] == True]
        
        if len(df_success) == 0:
            print("\n⚠️ No successful trials found!")
            return []
        
        # Sort by metric
        df_sorted = df_success.sort_values(by=metric, ascending=False)
        top_configs = df_sorted.head(top_k)
        
        print(f"\n{'='*80}")
        print(f"Top {top_k} Configurations (by {metric}):")
        print(f"{'='*80}\n")
        
        for idx, row in top_configs.iterrows():
            print(f"\nRank {top_configs.index.get_loc(idx) + 1}:")
            print(f"  Final Correlation: {row['final_cor']:.4f}")
            print(f"  Final Loss: {row['final_loss']:.4f}")
            print(f"  Max Correlation: {row['max_cor']:.4f}")
            print(f"  Parameters:")
            for key in ['lr', 'mask_ratio', 'patch_size', 'depth', 'embed_dim', 
                       'batch_size', 'warmup_epochs', 'weight_decay']:
                if key in row:
                    print(f"    {key}: {row[key]}")
        
        return top_configs
    
    def save_best_config(self, rank=1):
        """Save the best configuration to a Python file"""
        top_configs = self.get_best_configs(top_k=rank)
        
        if len(top_configs) == 0:
            print("No configurations to save")
            return
        
        best = top_configs.iloc[rank-1]
        
        config_content = f"""# Auto-generated optimal configuration
# Generated on: {datetime.now()}
# Correlation: {best['final_cor']:.4f}
# Loss: {best['final_loss']:.4f}

import os
from config_deap import Config_MAE_DEAP

class Config_MAE_DEAP_Tuned(Config_MAE_DEAP):
    def __init__(self):
        super().__init__()
        
        # Optimized hyperparameters
        self.lr = {best['lr']}
        self.mask_ratio = {best['mask_ratio']}
        self.patch_size = {int(best['patch_size'])}
        self.depth = {int(best['depth'])}
        self.embed_dim = {int(best['embed_dim'])}
        self.batch_size = {int(best['batch_size'])}
        self.warmup_epochs = {int(best['warmup_epochs'])}
        self.weight_decay = {best['weight_decay']}
        self.min_lr = {best['min_lr']}
        
        # Auto-adjusted parameters
        self.decoder_embed_dim = {int(best['embed_dim'])} // 2
        self.decoder_depth = max(4, {int(best['depth'])} // 3)
        self.num_heads = {int(best['embed_dim'])} // 64
        self.decoder_num_heads = self.decoder_embed_dim // 64
"""
        
        output_file = os.path.join(self.output_dir, 'config_deap_optimized.py')
        with open(output_file, 'w') as f:
            f.write(config_content)
        
        print(f"\n✓ Best configuration saved to: {output_file}")
        return output_file


def main():
    """Main tuning pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser('Hyperparameter Tuning for MAE-DEAP')
    parser.add_argument('--strategy', type=str, default='focused', 
                       choices=['quick', 'focused', 'exhaustive'],
                       help='Search strategy')
    parser.add_argument('--method', type=str, default='random',
                       choices=['random', 'grid'],
                       help='Search method')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of trials for random search')
    parser.add_argument('--trial_epochs', type=int, default=30,
                       help='Epochs per trial')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    base_config = Config_MAE_DEAP()
    tuner = HyperparameterTuner(base_config, output_dir=args.output_dir)
    
    # Define search space
    search_space = tuner.define_search_space(strategy=args.strategy)
    
    print(f"\nSearch Space:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    
    # Run search
    if args.method == 'random':
        best_configs = tuner.random_search(
            search_space, 
            n_trials=args.n_trials,
            trial_epochs=args.trial_epochs
        )
    else:
        best_configs = tuner.grid_search(
            search_space,
            max_combinations=args.n_trials,
            trial_epochs=args.trial_epochs
        )
    
    # Save best configuration
    if len(best_configs) > 0:
        tuner.save_best_config(rank=1)
    
    print(f"\n✓ Tuning complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
