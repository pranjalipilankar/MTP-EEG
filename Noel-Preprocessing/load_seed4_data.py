    """
Helper script to load processed SEED-IV data with labels

Usage example:
    from load_seed4_data import load_subject_session, load_all_data
    
    # Load single subject/session
    X, labels, meta = load_subject_session(session='1', subject='1_20160518')
    
    # Load all data
    all_data = load_all_data()
"""

import os
import json
import numpy as np
from pathlib import Path

BASE_DIR = "/home/ab_students/EEG-MTP/DATA/seed4/eeg_processed_data"

EMOTION_LABELS = {
    0: 'neutral',
    1: 'sad',
    2: 'fear',
    3: 'happy'
}


def load_subject_session(session, subject, base_dir=BASE_DIR, load_reversed=False):
    """
    Load preprocessed EEG data for a specific subject and session.
    
    Parameters:
    -----------
    session : str
        Session number ('1', '2', or '3')
    subject : str
        Subject identifier (e.g., '1_20160518')
    base_dir : str
        Base directory for processed data
    load_reversed : bool
        If True, also load the reversed (reconstructed) data
        
    Returns:
    --------
    X : ndarray
        Preprocessed EEG windows, shape (n_windows, n_channels, window_samples)
    labels : ndarray
        Emotion labels for each window, shape (n_windows,)
    meta : dict
        Metadata including sampling rate, window size, etc.
    reversed_X : ndarray (optional)
        Reversed data if load_reversed=True
    """
    data_path = Path(base_dir) / session / subject
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    # Load preprocessed data
    X = np.load(data_path / "X_prc1.npy")
    labels = np.load(data_path / "labels.npy")
    
    # Load metadata
    with open(data_path / "prc1_meta.json", 'r') as f:
        meta = json.load(f)
    
    # Load normalization stats if needed
    meta['norm_stats_path'] = str(data_path / "X_prc1_norm_stats.npy")
    
    print(f"Loaded: Session {session}, Subject {subject}")
    print(f"  Shape: {X.shape}")
    print(f"  Sampling rate: {meta['target_fs']} Hz")
    print(f"  Window size: {meta['window_sec']} seconds ({meta['window_samples']} samples)")
    print(f"  Label distribution:")
    for label_val, count in meta['label_distribution'].items():
        emotion = EMOTION_LABELS[int(label_val)]
        print(f"    {emotion}: {count} windows ({count/X.shape[0]*100:.1f}%)")
    
    if load_reversed:
        reversed_X = np.load(data_path / "X_prc1_reversed.npy")
        return X, labels, meta, reversed_X
    
    return X, labels, meta


def load_all_data(base_dir=BASE_DIR, load_reversed=False):
    """
    Load all processed SEED-IV data.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for processed data
    load_reversed : bool
        If True, also load reversed data
        
    Returns:
    --------
    all_data : dict
        Dictionary with structure: {session: {subject: {'X': X, 'labels': labels, 'meta': meta}}}
    """
    base_path = Path(base_dir)
    all_data = {}
    
    # Iterate through sessions
    for session_dir in sorted(base_path.iterdir()):
        if not session_dir.is_dir():
            continue
            
        session = session_dir.name
        all_data[session] = {}
        
        # Iterate through subjects
        for subject_dir in sorted(session_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
                
            subject = subject_dir.name
            
            try:
                if load_reversed:
                    X, labels, meta, reversed_X = load_subject_session(
                        session, subject, base_dir, load_reversed=True
                    )
                    all_data[session][subject] = {
                        'X': X,
                        'labels': labels,
                        'meta': meta,
                        'reversed_X': reversed_X
                    }
                else:
                    X, labels, meta = load_subject_session(session, subject, base_dir)
                    all_data[session][subject] = {
                        'X': X,
                        'labels': labels,
                        'meta': meta
                    }
            except Exception as e:
                print(f"Error loading {session}/{subject}: {e}")
                continue
    
    return all_data


def get_data_by_emotion(session, subject, emotion, base_dir=BASE_DIR):
    """
    Load data filtered by a specific emotion.
    
    Parameters:
    -----------
    session : str
        Session number ('1', '2', or '3')
    subject : str
        Subject identifier
    emotion : str or int
        Emotion name ('neutral', 'sad', 'fear', 'happy') or label (0, 1, 2, 3)
    base_dir : str
        Base directory for processed data
        
    Returns:
    --------
    X_filtered : ndarray
        Windows with the specified emotion label
    """
    X, labels, meta = load_subject_session(session, subject, base_dir)
    
    # Convert emotion name to label if needed
    if isinstance(emotion, str):
        emotion_to_label = {v: k for k, v in EMOTION_LABELS.items()}
        emotion_label = emotion_to_label[emotion.lower()]
    else:
        emotion_label = emotion
    
    # Filter by emotion
    mask = labels == emotion_label
    X_filtered = X[mask]
    
    print(f"Filtered {X_filtered.shape[0]} windows for emotion '{EMOTION_LABELS[emotion_label]}'")
    
    return X_filtered


def create_dataset_splits(all_data, train_sessions=['1', '2'], test_session='3'):
    """
    Create train/test splits by session (leave-one-session-out).
    
    Parameters:
    -----------
    all_data : dict
        Output from load_all_data()
    train_sessions : list
        List of session IDs for training
    test_session : str
        Session ID for testing
        
    Returns:
    --------
    X_train, y_train, X_test, y_test : ndarrays
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for session, subjects in all_data.items():
        for subject, data in subjects.items():
            if session in train_sessions:
                X_train_list.append(data['X'])
                y_train_list.append(data['labels'])
            elif session == test_session:
                X_test_list.append(data['X'])
                y_test_list.append(data['labels'])
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    print(f"Train: {X_train.shape[0]} windows from sessions {train_sessions}")
    print(f"Test: {X_test.shape[0]} windows from session {test_session}")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("SEED-IV Data Loading Examples")
    print("="*70)
    
    # Example 1: Load single subject/session
    print("\n[Example 1] Load single subject/session:")
    try:
        X, labels, meta = load_subject_session(session='1', subject='1_20160518')
        print(f"Successfully loaded data with shape: {X.shape}")
    except FileNotFoundError:
        print("Data not yet processed. Run FinalPrC-1.py first.")
    
    # Example 2: Load data for specific emotion
    print("\n[Example 2] Load data for specific emotion:")
    try:
        X_happy = get_data_by_emotion(session='1', subject='1_20160518', emotion='happy')
        print(f"Happy emotion windows: {X_happy.shape}")
    except FileNotFoundError:
        print("Data not yet processed. Run FinalPrC-1.py first.")
    
    # Example 3: Load all data
    print("\n[Example 3] Load all data:")
    try:
        all_data = load_all_data()
        total_subjects = sum(len(subjects) for subjects in all_data.values())
        print(f"Loaded {len(all_data)} sessions with {total_subjects} total subject recordings")
    except Exception as e:
        print(f"Could not load all data: {e}")
    
    print("\n" + "="*70)
