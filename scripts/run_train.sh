#!/bin/bash
# Training script for PROTAC Flow Matching model
# Usage: bash scripts/run_train.sh

# Activate conda environment
source activate linker  # or: conda activate linker

# Set working directory
cd "$(dirname "$0")/.."

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export KMP_DUPLICATE_LIB_OK=TRUE

echo "============================================================"
echo "PROTAC Flow Matching - Training"
echo "============================================================"
echo "Config: configs/training/fm_protac.yml"
echo "Log directory: ./logs"
echo "============================================================"

# Create logs directory if not exists
mkdir -p ./logs

# Run training
python scripts/train_fm_protac.py configs/training/fm_protac.yml --device cuda --logdir ./logs

echo "Training completed!"
