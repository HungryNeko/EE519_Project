#!/bin/bash
#SBATCH --job-name=whisper_cf_hi_talcs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s|v100"
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=whisper_cf_hi_talcs_%j.out
#SBATCH --error=whisper_cf_hi_talcs_%j.err

set -euo pipefail

CONDA_SH=~/miniforge3/etc/profile.d/conda.sh
PROJECT_DIR=/scratch1/xwei6575/github/EE519_Project

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$CONDA_SH"
conda activate whisperx

cd "$PROJECT_DIR"

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
which python
python -V
nvidia-smi

python -u translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset crossfade_hindi_english_talcs

echo "Done: $(date)"

# 如果作业中断但 jsonl 还在，需要手动合并时再跑这条：
# python -u translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset crossfade_hindi_english_talcs --finalize-only
