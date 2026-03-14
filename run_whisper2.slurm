#!/bin/bash
#SBATCH --job-name=whisperx_hinglish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s|v100"
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=whisper_hinglish_%j.out

# 避免 OpenBLAS / tqdm 线程问题
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 正确的 conda 路径（你机器上是 miniforge3）
source ~/miniforge3/etc/profile.d/conda.sh
conda activate whisperx

cd /scratch1/xwei6575/github/EE519_Project

python -u translate/whisperX/whisper_wordlevel_json_corss_new.py --dataset hinglish
