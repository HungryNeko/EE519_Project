#!/bin/bash
#SBATCH --job-name=prepare_hindi_en
# 如果你的集群必须指定 partition，在下一行取消注释并修改
##SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=prepare_hindi_en_%j.out
#SBATCH --error=prepare_hindi_en_%j.err

set -euo pipefail

# 只需要按你的 server 环境改这三行
CONDA_SH=~/miniforge3/etc/profile.d/conda.sh
PROJECT_DIR=/scratch1/xwei6575/github/EE519_Project
ARCHIVE_PATH=/scratch1/xwei6575/github/EE519_Project/datasets/Hindi-English_train.tar.gz

MIN_SEC=4
WORKERS=${SLURM_CPUS_PER_TASK:-8}

source "$CONDA_SH"
conda activate whisperx

cd "$PROJECT_DIR"

DATASET_DIR="$PROJECT_DIR/datasets/Hindi-English_train"
STAGING_DIR="$PROJECT_DIR/datasets/.hindi_english_extract_tmp"

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Archive: $ARCHIVE_PATH"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "Archive not found: $ARCHIVE_PATH" >&2
  exit 1
fi

mkdir -p "$DATASET_DIR"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

case "$ARCHIVE_PATH" in
  *.zip)
    unzip -q "$ARCHIVE_PATH" -d "$STAGING_DIR"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE_PATH" -C "$STAGING_DIR"
    ;;
  *.tar)
    tar -xf "$ARCHIVE_PATH" -C "$STAGING_DIR"
    ;;
  *)
    echo "Unsupported archive format: $ARCHIVE_PATH" >&2
    exit 1
    ;;
esac

EXTRACTED_TRAIN=""
if [[ -d "$STAGING_DIR/train" ]]; then
  EXTRACTED_TRAIN="$STAGING_DIR/train"
else
  while IFS= read -r candidate; do
    if [[ -d "$candidate/train" ]]; then
      EXTRACTED_TRAIN="$candidate/train"
      break
    fi
  done < <(find "$STAGING_DIR" -mindepth 1 -maxdepth 2 -type d | sort)
fi

if [[ -z "$EXTRACTED_TRAIN" ]]; then
  echo "Cannot locate extracted train directory under $STAGING_DIR" >&2
  exit 1
fi

rm -rf "$DATASET_DIR/train"
mv "$EXTRACTED_TRAIN" "$DATASET_DIR/train"

rm -rf "$STAGING_DIR"

python datasets/Hindi-English_train/filter_min_duration_flat.py \
  --audio-dir datasets/Hindi-English_train/train \
  --min-sec "$MIN_SEC" \
  --workers "$WORKERS"

echo "Remaining counts after filtering:"
find datasets/Hindi-English_train/train -type f -name '*.wav' | wc -l

echo "Done: $(date)"
