@echo off
REM Sampling script for PROTAC Flow Matching model
REM Based on LinkerNet sampling approach

call mamba activate linker

cd /d %~dp0\..

REM Set environment variables
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=%cd%

echo ============================================================
echo PROTAC Flow Matching - Sampling
echo ============================================================
echo Please specify the checkpoint path when running this script
echo Example: run_sample.bat path/to/checkpoint.pt
echo ============================================================

REM Check if checkpoint path is provided
if "%~1"=="" (
    echo Error: Please provide checkpoint path as argument
    echo Usage: run_sample.bat path/to/checkpoint.pt
    pause
    exit /b 1
)

set CKPT_PATH=%~1

echo Checkpoint: %CKPT_PATH%
echo Output directory: ./outputs
echo ============================================================

python scripts/sample_protac.py configs/protac.yml --ckpt %CKPT_PATH% --device cuda --output_dir ./outputs --num_samples 100

pause
