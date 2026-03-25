#!/bin/bash
#SBATCH --job-name=whisperx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s|v100"
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=whisper_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisperx

cd /scratch1/shihe/github/EE519_Project

python -u translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset seame_dev_sge_cn_en_2000
python -u translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset switchlingua_audio_2000
