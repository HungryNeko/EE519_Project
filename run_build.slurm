#!/bin/bash
#SBATCH --job-name=whisperx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=whisper_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate whisperx

cd /scratch1/xwei6575/github/EE519_Project

python -u dl_model/build_feature_json.py