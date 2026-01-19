@echo off
REM Evaluation script for PROTAC Flow Matching model
REM Based on LinkerNet evaluation metrics

call mamba activate linker

cd /d %~dp0\..

REM Set environment variables
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=%cd%

echo ============================================================
echo PROTAC Flow Matching - Evaluation
echo ============================================================

REM Check if results path is provided
if "%~1"=="" (
    set RESULTS_PATH=./outputs/sampling_results.pt
) else (
    set RESULTS_PATH=%~1
)

echo Results: %RESULTS_PATH%
echo Config: configs/protac.yml
echo Output: ./outputs/eval_results.pt
echo ============================================================

python scripts/eval_protac.py --results %RESULTS_PATH% --config configs/protac.yml --output ./outputs/eval_results.pt

pause
