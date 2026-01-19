@echo off
REM Sampling script for PROTAC Flow Matching model

call mamba activate linker

cd /d %~dp0\..

REM Change the checkpoint path as needed
python scripts/sample_protac.py configs/protac.yml --ckpt ./logs/protac_*/checkpoints/best.pt --device cuda --output_dir ./outputs --num_samples 100

pause
