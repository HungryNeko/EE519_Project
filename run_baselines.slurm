#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/baseline_%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ee519mlp

cd /scratch1/shihe/github/EE519_Project

mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${SCRATCH:-$PWD}/.cache/huggingface"
export TORCH_HOME="${SCRATCH:-$PWD}/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# If this environment does not already have the baseline packages,
# run the next line once on the server before submitting the job.
# pip install -r baseline/requirements.txt

python -u -m baseline.run_baselines \
  --device cpu \
  --csv dl_model/test_baseline_segments.csv \
  --output-dir "baseline/results/job_${SLURM_JOB_ID}"
