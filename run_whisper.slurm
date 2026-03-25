#!/bin/bash
#SBATCH --job-name=whisperx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s|v100"
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=whisper_%j.out
#SBATCH --error=whisper_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisperx

cd /scratch1/shihe/github/EE519_Project

echo "Start: $(date)"
echo "Host: $(hostname)"
which python
python -V
nvidia-smi

python -u translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset crossfade_switchlingua_seame

echo "Done: $(date)"