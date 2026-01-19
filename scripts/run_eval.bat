@echo off
REM Evaluation script for PROTAC Flow Matching model

call mamba activate linker

cd /d %~dp0\..

python scripts/eval_protac.py --results ./outputs/sampling_results.pt --config configs/protac.yml --output ./outputs/eval_results.pt

pause
