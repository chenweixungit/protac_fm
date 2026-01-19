@echo off
REM Training script for PROTAC Flow Matching model

call mamba activate linker

cd /d %~dp0\..

python scripts/train_protac.py configs/protac.yml --device cuda --logdir ./logs --train_report_iter 200 --sampling_iter 10000

pause
