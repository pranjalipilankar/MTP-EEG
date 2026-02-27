#!/bin/bash
# Clean Python cache files
find /home/ab_students/EEG-MTP/New_LocalizeMI -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /home/ab_students/EEG-MTP/New_LocalizeMI -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Cleaned Python cache files"
