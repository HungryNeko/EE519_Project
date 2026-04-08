#!/bin/bash
#SBATCH --job-name=prepare_talcs
# 如果你的集群必须指定 partition，在下一行取消注释并修改
##SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=prepare_talcs_%j.out
#SBATCH --error=prepare_talcs_%j.err

set -euo pipefail

# 只需要按你的 server 环境改这三行
CONDA_SH=~/miniforge3/etc/profile.d/conda.sh
PROJECT_DIR=/scratch1/xwei6575/github/EE519_Project
ARCHIVE_PATH=/scratch1/xwei6575/github/EE519_Project/datasets/TALCS_corpus.tar

MIN_SEC=4
WORKERS=${SLURM_CPUS_PER_TASK:-8}

source "$CONDA_SH"
conda activate whisperx

cd "$PROJECT_DIR"

DATASET_DIR="$PROJECT_DIR/datasets/TALCS_corpus"
STAGING_DIR="$PROJECT_DIR/datasets/.talcs_extract_tmp"

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

EXTRACTED_ROOT=""
if [[ -d "$STAGING_DIR/TALCS_corpus" ]]; then
  EXTRACTED_ROOT="$STAGING_DIR/TALCS_corpus"
elif [[ -d "$STAGING_DIR/train_set" && -d "$STAGING_DIR/dev_set" && -d "$STAGING_DIR/test_set" ]]; then
  EXTRACTED_ROOT="$STAGING_DIR"
else
  while IFS= read -r candidate; do
    if [[ -d "$candidate/train_set" && -d "$candidate/dev_set" && -d "$candidate/test_set" ]]; then
      EXTRACTED_ROOT="$candidate"
      break
    fi
  done < <(find "$STAGING_DIR" -mindepth 1 -maxdepth 2 -type d | sort)
fi

if [[ -z "$EXTRACTED_ROOT" ]]; then
  echo "Cannot locate extracted TALCS_corpus root under $STAGING_DIR" >&2
  exit 1
fi

for split in train_set dev_set test_set; do
  if [[ ! -d "$EXTRACTED_ROOT/$split" ]]; then
    echo "Missing extracted split: $EXTRACTED_ROOT/$split" >&2
    exit 1
  fi
  rm -rf "$DATASET_DIR/$split"
  mv "$EXTRACTED_ROOT/$split" "$DATASET_DIR/"
done

rm -rf "$STAGING_DIR"

python datasets/TALCS_corpus/filter_min_duration.py --min-sec "$MIN_SEC" --workers "$WORKERS"

echo "Remaining counts after filtering:"
python - <<'PY'
from pathlib import Path
for split in ['train_set', 'dev_set', 'test_set']:
    wav_dir = Path('datasets/TALCS_corpus') / split / 'wav'
    label = Path('datasets/TALCS_corpus') / split / 'label.txt'
    wav_count = sum(1 for _ in wav_dir.glob('*.wav'))
    label_count = sum(1 for _ in label.open('r', encoding='utf-8'))
    print(split, 'wav_files=', wav_count, 'label_lines=', label_count)
PY

echo "Done: $(date)"
