#!/usr/bin/env python3
"""Prepare SEED-IV test dataset from processed windows with subject/session/trial metadata.

This script mirrors the processed-data layout used by training, but exports only the
held-out test split so evaluation can run without leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm


def get_seed4_channel_indices(target_channels):
    """Fixed SEED-IV channel subsets used during STAD training."""
    if target_channels == 62:
        return np.arange(62, dtype=int)
    if target_channels == 31:
        return np.array([
            0, 2, 4, 5, 7, 9, 11, 13,
            15, 17, 19, 21, 23, 25, 27, 29,
            31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 58, 60,
        ], dtype=int)
    if target_channels == 16:
        return np.array([
            0, 2, 5, 7, 9, 13, 17, 21,
            23, 27, 31, 35, 39, 45, 53, 60,
        ], dtype=int)
    return np.linspace(0, 61, target_channels, dtype=int)


def create_subject_split(n_folds=5, test_fold=0):
    """Reproduce the subject-level fold split used for STAD testing."""
    all_subjects = [str(i) for i in range(1, 16)]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2024)

    splits = list(kf.split(all_subjects))
    train_val_idx, test_idx = splits[test_fold]

    val_size = len(train_val_idx) // 5
    val_idx = train_val_idx[:val_size]
    train_idx = train_val_idx[val_size:]

    return {
        'train': [all_subjects[i] for i in train_idx],
        'val': [all_subjects[i] for i in val_idx],
        'test': [all_subjects[i] for i in test_idx],
    }


def load_processed_windows(data_root, subjects):
    """Load processed windows for the requested subjects and preserve metadata."""
    data_root = Path(data_root)

    lr_indices = get_seed4_channel_indices(16)
    hr_indices = get_seed4_channel_indices(31)

    all_sr = []
    all_labels = []
    all_subject_ids = []
    all_session_ids = []
    all_trial_ids = []
    all_trial_labels = []
    all_source_files = []
    all_norm_stats = []
    all_prc1_meta = []

    for session in ['1', '2', '3']:
        session_path = data_root / session
        if not session_path.exists():
            print(f"Warning: session {session} not found at {session_path}")
            continue

        subject_folders = sorted([folder for folder in session_path.iterdir() if folder.is_dir()])
        for folder in tqdm(subject_folders, desc=f"Session {session}"):
            subject_id = folder.name.split('_')[0]
            if subject_id not in subjects:
                continue

            x_file = folder / 'X_prc1.npy'
            label_file = folder / 'labels.npy'
            trial_labels_file = folder / 'trial_labels.json'
            meta_file = folder / 'prc1_meta.json'
            stats_file = folder / 'X_prc1_norm_stats.npy'

            if not x_file.exists() or not label_file.exists():
                continue

            x_data = np.load(x_file).astype(np.float32)
            labels = np.load(label_file).astype(np.int64)

            if trial_labels_file.exists():
                with open(trial_labels_file, 'r', encoding='utf-8') as f:
                    trial_meta = json.load(f)
                windows_per_trial = trial_meta.get('windows_per_trial', [])
                trial_label_list = trial_meta.get('trial_labels', [])
            else:
                windows_per_trial = []
                trial_label_list = []

            if windows_per_trial and sum(int(v) for v in windows_per_trial) == len(x_data):
                trial_ids = []
                trial_labels = []
                for trial_idx, n_windows in enumerate(windows_per_trial):
                    trial_ids.extend([trial_idx] * int(n_windows))
                    if trial_idx < len(trial_label_list):
                        trial_labels.extend([int(trial_label_list[trial_idx])] * int(n_windows))
                    else:
                        trial_labels.extend([int(labels[0]) if len(labels) else 0] * int(n_windows))
                trial_ids = np.asarray(trial_ids, dtype=np.int64)
                trial_labels = np.asarray(trial_labels, dtype=np.int64)
            else:
                # Fallback: stable per-window ids when explicit trial windows are unavailable.
                trial_ids = np.arange(len(x_data), dtype=np.int64)
                trial_labels = np.asarray(labels, dtype=np.int64)

            all_sr.append(x_data)
            all_labels.append(labels)
            all_subject_ids.extend([str(subject_id)] * len(x_data))
            all_session_ids.extend([str(session)] * len(x_data))
            all_trial_ids.extend(trial_ids.tolist())
            all_trial_labels.extend(trial_labels.tolist())
            all_source_files.extend([folder.name] * len(x_data))

            if stats_file.exists():
                stats = np.load(stats_file)
                if len(stats) == len(x_data):
                    all_norm_stats.append(stats)

            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    all_prc1_meta.append(json.load(f))

    if not all_sr:
        raise ValueError(f"No processed data found for subjects {subjects} in {data_root}")

    sr = np.concatenate(all_sr, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    subject_ids = np.asarray(all_subject_ids, dtype=str)
    session_ids = np.asarray(all_session_ids, dtype=str)
    trial_ids = np.asarray(all_trial_ids, dtype=np.int64)
    trial_labels = np.asarray(all_trial_labels, dtype=np.int64)
    source_files = np.asarray(all_source_files, dtype=str)

    lr = sr[:, lr_indices, :].astype(np.float32)
    hr = sr[:, hr_indices, :].astype(np.float32)
    sr = sr.astype(np.float32)

    if all_norm_stats and sum(len(x) for x in all_norm_stats) == len(sr):
        norm_stats = np.concatenate(all_norm_stats, axis=0).astype(np.float32)
    else:
        norm_stats = None

    prc1_meta = all_prc1_meta[0] if all_prc1_meta else None

    return {
        'LR': lr,
        'HR': hr,
        'SR': sr,
        'labels': labels,
        'subject_ids': subject_ids,
        'session_ids': session_ids,
        'trial_ids': trial_ids,
        'trial_labels': trial_labels,
        'source_files': source_files,
        'norm_stats': norm_stats,
        'prc1_meta': prc1_meta,
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare SEED-IV test dataset from processed windows')
    parser.add_argument(
        '--data_root', '--raw_data_root',
        type=str,
        default='/DATA/EEG-MTP/seed4/eeg_processed_data',
        help='Processed SEED-IV root containing session/subject folders',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/DATA/EEG-MTP/seed4/test_data.npz',
        help='Output test NPZ path',
    )
    parser.add_argument(
        '--test_fold',
        type=int,
        default=0,
        help='Fold index used to choose held-out test subjects',
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of subject folds',
    )
    args = parser.parse_args()

    # Common user typo: eeg_preprocessed_data -> eeg_processed_data
    data_root = Path(args.data_root)
    if not data_root.exists() and 'eeg_preprocessed_data' in str(data_root):
        fallback = Path(str(data_root).replace('eeg_preprocessed_data', 'eeg_processed_data'))
        if fallback.exists():
            print(f"Info: {data_root} not found, using {fallback}")
            data_root = fallback

    # If user passed the parent seed4 path, append the expected processed folder.
    if data_root.is_dir() and not (data_root / '1').exists() and (data_root / 'eeg_processed_data').exists():
        fallback = data_root / 'eeg_processed_data'
        print(f"Info: using processed-data subfolder {fallback}")
        data_root = fallback

    splits = create_subject_split(n_folds=args.n_folds, test_fold=args.test_fold)
    test_subjects = splits['test']

    print('Preparing SEED-IV processed test dataset')
    print(f'Using test subjects: {test_subjects}')
    payload = load_processed_windows(data_root, test_subjects)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        LR=payload['LR'],
        HR=payload['HR'],
        SR=payload['SR'],
        labels=payload['labels'],
        subject_ids=payload['subject_ids'],
        session_ids=payload['session_ids'],
        trial_ids=payload['trial_ids'],
        trial_labels=payload['trial_labels'],
        source_files=payload['source_files'],
        test_subjects=np.asarray(test_subjects, dtype=str),
        test_fold=np.asarray([args.test_fold], dtype=np.int64),
    )

    loaded = np.load(output_path, allow_pickle=True)
    print('\nSaved NPZ keys:', loaded.files)
    print('LR:', loaded['LR'].shape)
    print('HR:', loaded['HR'].shape)
    print('SR:', loaded['SR'].shape)
    print('subject_ids:', loaded['subject_ids'].shape)
    print('session_ids:', loaded['session_ids'].shape)
    print('trial_ids:', loaded['trial_ids'].shape)
    print('test_subjects:', loaded['test_subjects'])


if __name__ == '__main__':
    main()