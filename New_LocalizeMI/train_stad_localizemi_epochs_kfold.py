#!/usr/bin/env python3
"""
STAD training entrypoint for Localize-MI raw epochs + MAE results_128ch_kfold.

Uses:
- Dataset: /home/arnav-a5000/MTP-EEG/DATA/Localize-MI/derivatives/epochs
- MAE k-fold dir: /home/arnav-a5000/MTP-EEG/trial_mae_Localize-MI/results_128ch_kfold
"""

import argparse
from train_stad_localizemi import train_stad_localizemi


def main():
    parser = argparse.ArgumentParser(
        description='Train STAD on Localize-MI raw epochs using MAE results_128ch_kfold'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/arnav-a5000/MTP-EEG/DATA/Localize-MI/derivatives/epochs',
        help='Path to raw epochs folder (sub-*/eeg/*_epochs.npy)'
    )
    parser.add_argument(
        '--mae_results_dir',
        type=str,
        default='/home/arnav-a5000/MTP-EEG/trial_mae_Localize-MI/results_128ch_kfold',
        help='Path to MAE k-fold directory containing fold_splits.json and fold_*/best_model.pth'
    )
    parser.add_argument('--mae_fold', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_workers', type=int, default=0)
    parser.add_argument('--val_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--diff_weight', type=float, default=1.0)
    parser.add_argument('--sr_l1_weight', type=float, default=0.5)
    args = parser.parse_args()

    loss_weights = {
        'diff': args.diff_weight,
        'sr_l1': args.sr_l1_weight,
    }

    train_stad_localizemi(
        data_path=args.data_path,
        mae_results_dir=args.mae_results_dir,
        mae_fold=args.mae_fold,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        preprocessed=False,
        loss_weights=loss_weights,
    )


if __name__ == '__main__':
    main()
