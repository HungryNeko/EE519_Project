#!/bin/bash
#SBATCH --job-name=whisperx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=30:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=whisper_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ee519mlp

cd /scratch1/shihe/github/EE519_Project

python -u dl_model/build_feature_json.py