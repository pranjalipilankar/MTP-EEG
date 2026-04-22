#!/bin/bash
echo "=========================================="
echo "MFE Loss Setup Verification"
echo "=========================================="
echo ""

# Check Python imports
echo "1. Checking Python imports..."
python3 -c "from mfe_profile_loss import MFEProfileLoss; print('   ✅ MFE loss imports OK')" 2>/dev/null || echo "   ❌ MFE import failed"
python3 -c "from stad_model_CORRECT import STADModel; print('   ✅ STAD model imports OK')" 2>/dev/null || echo "   ❌ STAD import failed"
python3 -c "from mae_for_eeg import MAEforEEG; print('   ✅ MAE model imports OK')" 2>/dev/null || echo "   ❌ MAE import failed"

# Check file sizes
echo ""
echo "2. Verifying file sizes..."
echo "   MFE loss: $(du -h mfe_profile_loss.py | cut -f1)"
echo "   Training script: $(du -h seed_stad_train_mfe.py | cut -f1)"
echo "   STAD model: $(du -h stad_model_CORRECT.py | cut -f1)"

# Check documentation
echo ""
echo "3. Documentation files:"
ls -1 *.md | sed 's/^/   ✅ /'

# Check data path
echo ""
echo "4. Data availability:"
if [ -d "/DATA/EEG-MTP/seed4/eeg_processed_data" ]; then
    echo "   ✅ SEED-IV data found"
else
    echo "   ⚠️  SEED-IV data not found at /DATA/EEG-MTP/seed4/eeg_processed_data"
fi

# Check MAE checkpoint location
echo ""
echo "5. MAE checkpoint:"
MAE_DIR="/home/ab_students/EEG-MTP/trial_mae_SEED4/results_31ch_kfold_fixed"
if [ -d "$MAE_DIR" ]; then
    CHECKPOINT_COUNT=$(find "$MAE_DIR" -name "best_model.pth" 2>/dev/null | wc -l)
    echo "   ✅ MAE checkpoint directory found ($CHECKPOINT_COUNT checkpoints)"
else
    echo "   ⚠️  MAE checkpoint directory not found"
fi

# Check GPU
echo ""
echo "6. GPU status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "   ✅ $line"
    done
else
    echo "   ⚠️  NVIDIA GPU not detected"
fi

echo ""
echo "=========================================="
echo "Setup verification complete!"
echo "=========================================="
