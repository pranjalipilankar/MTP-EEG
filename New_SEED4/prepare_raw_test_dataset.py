#!/usr/bin/env python3
"""Prepare SEED-IV raw test data with session_ids and trial_ids for leakage-safe evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
from tqdm import tqdm


SESSION_LABELS = {
    '1': [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    '2': [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    '3': [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def load_and_process_raw_data(data_root):
    """Load SEED-IV raw .mat files and emit per-window subject/session/trial metadata."""
    data_root = Path(data_root)

    all_data = []
    all_labels = []
    all_subject_ids = []
    all_session_ids = []
    all_trial_ids = []
    all_trial_labels = []
    all_source_files = []

    for session in ['1', '2', '3']:
        session_path = data_root / session
        if not session_path.exists():
            print(f"Warning: session {session} not found at {session_path}")
            continue

        mat_files = sorted(session_path.glob('*.mat'))
        if not mat_files:
            print(f"Warning: no .mat files found in {session_path}")
            continue

        for mat_file in tqdm(mat_files, desc=f"Session {session}"):
            subject_id = mat_file.stem.split('_')[0]

            try:
                mat = loadmat(mat_file)
                eeg_keys = sorted([key for key in mat.keys() if not key.startswith('__')])
                trial_labels = SESSION_LABELS.get(session, [0] * 24)

                for trial_idx, eeg_key in enumerate(eeg_keys):
                    eeg_data = mat[eeg_key]

                    if eeg_data.shape[0] != 62:
                        eeg_data = eeg_data.T

                    eeg_downsampled = decimate(eeg_data, q=4, axis=1)

                    n_samples = eeg_downsampled.shape[1]
                    window_size = 1000
                    n_windows = n_samples // window_size

                    for window_idx in range(n_windows):
                        start_idx = window_idx * window_size
                        end_idx = start_idx + window_size
                        window = eeg_downsampled[:, start_idx:end_idx]

                        all_data.append(window)
                        all_labels.append(trial_labels[trial_idx] if trial_idx < len(trial_labels) else 0)
                        all_subject_ids.append(str(subject_id))
                        all_session_ids.append(str(session))
                        all_trial_ids.append(int(trial_idx))
                        all_trial_labels.append(int(trial_labels[trial_idx] if trial_idx < len(trial_labels) else 0))
                        all_source_files.append(mat_file.name)

                print(f"Loaded {mat_file.name}: {len(eeg_keys)} trials")

            except Exception as exc:
                print(f"Warning: failed to load {mat_file.name}: {exc}")

    if not all_data:
        raise ValueError(f"No raw EEG windows found in {data_root}")

    all_data = np.stack(all_data, axis=0).astype(np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int64)
    all_subject_ids = np.asarray(all_subject_ids, dtype=str)
    all_session_ids = np.asarray(all_session_ids, dtype=str)
    all_trial_ids = np.asarray(all_trial_ids, dtype=np.int64)
    all_trial_labels = np.asarray(all_trial_labels, dtype=np.int64)
    all_source_files = np.asarray(all_source_files, dtype=str)

    print(f"\nTotal raw windows: {all_data.shape}")
    print(f"Subjects: {len(np.unique(all_subject_ids))}")
    print(f"Sessions: {sorted(np.unique(all_session_ids).tolist())}")
    print(f"Trials: {len(np.unique(all_trial_ids))} unique trial ids per session-agnostic window stream")

    return {
        'SR': all_data,
        'labels': all_labels,
        'subject_ids': all_subject_ids,
        'session_ids': all_session_ids,
        'trial_ids': all_trial_ids,
        'trial_labels': all_trial_labels,
        'source_files': all_source_files,
    }


def create_multiresolution(sr_data):
    """Create LR/HR views from the 62-channel SR windows."""
    _, channels, _ = sr_data.shape

    sr = sr_data.astype(np.float32)
    hr_indices = np.linspace(0, channels - 1, 31, dtype=int)
    lr_indices = np.linspace(0, channels - 1, 16, dtype=int)

    hr = sr_data[:, hr_indices, :].astype(np.float32)
    lr = sr_data[:, lr_indices, :].astype(np.float32)

    print("\nCreated multi-resolution views:")
    print(f"  LR: {lr.shape}")
    print(f"  HR: {hr.shape}")
    print(f"  SR: {sr.shape}")

    return lr, hr, sr


def main():
    parser = argparse.ArgumentParser(description='Prepare raw SEED-IV test dataset with session_ids and trial_ids')
    parser.add_argument(
        '--raw_data_root',
        type=str,
        default='/DATA/EEG-MTP/seed4/eeg_raw_data',
        help='Path to SEED-IV raw data root containing session folders 1/2/3',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/DATA/EEG-MTP/seed4/raw_test_data.npz',
        help='Output .npz path',
    )
    args = parser.parse_args()

    print('Preparing SEED-IV raw test dataset')
    print(f'Raw root: {args.raw_data_root}')
    print(f'Output:   {args.output_path}')

    payload = load_and_process_raw_data(args.raw_data_root)
    lr, hr, sr = create_multiresolution(payload['SR'])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        LR=lr,
        HR=hr,
        SR=sr,
        labels=payload['labels'],
        subject_ids=payload['subject_ids'],
        session_ids=payload['session_ids'],
        trial_ids=payload['trial_ids'],
        trial_labels=payload['trial_labels'],
        source_files=payload['source_files'],
    )

    loaded = np.load(output_path, allow_pickle=True)
    print('\nSaved NPZ keys:', loaded.files)
    print('LR:', loaded['LR'].shape)
    print('HR:', loaded['HR'].shape)
    print('SR:', loaded['SR'].shape)
    print('subject_ids:', loaded['subject_ids'].shape)
    print('session_ids:', loaded['session_ids'].shape)
    print('trial_ids:', loaded['trial_ids'].shape)
    print('Done.')


if __name__ == '__main__':
    main()