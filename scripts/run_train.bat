@echo off
REM Training script for PROTAC Flow Matching model
REM Based on LinkerNet training parameters

call mamba activate linker

cd /d %~dp0\..

REM Set environment variables
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=%cd%

echo ============================================================
echo PROTAC Flow Matching - Training
echo ============================================================
echo Config: configs/protac.yml
echo Log directory: ./logs
echo ============================================================

python scripts/train_protac.py configs/protac.yml --device cuda --logdir ./logs

pause
