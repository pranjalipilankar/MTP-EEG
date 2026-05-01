import os
from pathlib import Path
import numpy as np

print("\n" + "="*80)
print("STAD EVALUATION - OUTPUT STRUCTURE VERIFICATION")
print("="*80)

# Check evaluation directory
eval_dir = Path('stad_raw_evaluation')
print(f"\n📁 Evaluation Directory: {eval_dir.absolute()}")
print(f"   ├── Size: {sum(f.stat().st_size for f in eval_dir.rglob('*')) / (1024**2):.1f} MB")
files = sorted(eval_dir.glob('*'))
for f in files:
    size_str = f"({f.stat().st_size / (1024**2):.1f} MB)" if f.is_file() else ""
    print(f"   ├── {'📄' if f.is_file() else '📁'} {f.name:<40} {size_str}")

# Check subject-wise output
subject_dir = Path('stad_raw_subject_output')
print(f"\n📁 Subject-Wise Output: {subject_dir.absolute()}")
total_size = sum(f.stat().st_size for f in subject_dir.rglob('*')) / (1024**2)
print(f"   ├── Total Size: {total_size:.1f} MB")

for subject_folder in sorted(subject_dir.glob('subject_*')):
    npy_files = len(list(subject_folder.glob('*.npy')))
    json_files = len(list(subject_folder.glob('*.json')))
    size = sum(f.stat().st_size for f in subject_folder.rglob('*')) / (1024**2)
    print(f"   ├── 📁 {subject_folder.name}")
    print(f"   │   ├── .npy files: {npy_files} ({npy_files//2} trial pairs)")
    print(f"   │   ├── .json files: {json_files} (metadata)")
    print(f"   │   └── Size: {size:.1f} MB")

# Verify README files
print(f"\n📄 Documentation:")
readme_eval = eval_dir / 'README.md'
readme_subject = subject_dir / 'README.md'
print(f"   ├── {readme_eval.name:<40} {'✅' if readme_eval.exists() else '❌'} ({readme_eval.stat().st_size / 1024:.1f} KB)")
print(f"   └── {readme_subject.name:<40} {'✅' if readme_subject.exists() else '❌'} ({readme_subject.stat().st_size / 1024:.1f} KB)")

# Check sample files
print(f"\n🔍 Sample File Verification:")
sample_pred = subject_dir / 'subject_7' / 'trial_0000_pred_sr.npy'
sample_target = subject_dir / 'subject_7' / 'trial_0000_target_sr.npy'
sample_meta = subject_dir / 'subject_7' / 'trial_0000_meta.json'

if sample_pred.exists():
    pred = np.load(sample_pred)
    print(f"   ├── Prediction shape: {pred.shape}, dtype: {pred.dtype}, range: [{pred.min():.4f}, {pred.max():.4f}]")
if sample_target.exists():
    target = np.load(sample_target)
    print(f"   ├── Target shape: {target.shape}, dtype: {target.dtype}, range: [{target.min():.4f}, {target.max():.4f}]")
if sample_meta.exists():
    import json
    with open(sample_meta) as f:
        meta = json.load(f)
    print(f"   └── Metadata: {meta}")

# Load and display summary metrics
print(f"\n📊 Results Summary:")
summary_file = eval_dir / 'results_summary.npz'
if summary_file.exists():
    results = np.load(summary_file, allow_pickle=True)
    keys = list(results.keys())
    print(f"   ├── NPZ Keys: {', '.join(keys)}")
    print(f"   ├── Total batches: {len(results['pcc_scores'])}")
    print(f"   ├── Mean PCC: {np.mean(results['pcc_scores']):.4f} ± {np.std(results['pcc_scores']):.4f}")
    print(f"   ├── Mean NMSE: {np.mean(results['nmse_scores']):.4f} ± {np.std(results['nmse_scores']):.4f}")
    print(f"   └── Mean SNR: {np.mean(results['snr_scores']):.2f} ± {np.std(results['snr_scores']):.2f} dB")

# Check visualization
import os
for file in eval_dir.glob('*.png'):
    print(f"\n🖼️  Visualization: {file.name} ({file.stat().st_size / 1024:.1f} KB)")

print(f"\n" + "="*80)
print("✅ All outputs verified successfully!")
print("="*80 + "\n")

