import argparse
import csv
import os
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Assuming librosa is available, else use scipy
try:
    import librosa
except ImportError:
    librosa = None

def load_audio(path: Path, sr=16000):
    """Load audio and resample to 16kHz mono."""
    wav, src_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if src_sr != sr:
        if librosa:
            wav = librosa.resample(wav, orig_sr=src_sr, target_sr=sr)
        else:
            from scipy import signal
            gcd = np.gcd(src_sr, sr)
            up = sr // gcd
            down = src_sr // gcd
            wav = signal.resample_poly(wav, up=up, down=down)
    return wav, sr

def extract_window(wav: np.ndarray, sr: int, start_time: float, end_time: float) -> np.ndarray:
    """Extract window from start_time to end_time, pad with zeros if necessary."""
    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    length = max(1, end_i - start_i)
    out = np.zeros(length, dtype=np.float32)

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)
    if src_end > src_start:
        dst_start = src_start - start_i
        dst_end = dst_start + (src_end - src_start)
        out[dst_start:dst_end] = wav[src_start:src_end]
    return out

def resolve_audio_path(audio_rel_path: str, base_dir: Path):
    """Resolve audio path with case-insensitive matching."""
    parts = audio_rel_path.split('/')
    current = base_dir
    for part in parts:
        if not current.exists():
            return None
        candidates = list(current.iterdir())
        match = None
        for cand in candidates:
            if cand.name.lower() == part.lower():
                match = cand
                break
        if match is None:
            return None
        current = match
    return current

def main():
    parser = argparse.ArgumentParser(description="Extract 4s audio segments and package into tar.")
    parser.add_argument("--csv-dir", default="dl_model/csv2", help="Directory containing CSV files")
    parser.add_argument("--output-dir", default="datasets/train_test2", help="Output directory for train/test")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    base_dir = Path.cwd()  # ee519_project
    csv_dir = base_dir / args.csv_dir
    output_base = base_dir / args.output_dir
    train_dir = output_base / "train"
    test_dir = output_base / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        ("baseline_train_test_segments.csv", "train"),
        ("baseline_train_test_segments_switchlingua_seame.csv", "test"),
    ]

    total_processed = 0

    for csv_name, default_split in csv_files:
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for idx, row in tqdm(enumerate(rows, 1), desc=f"Processing {csv_name}", total=len(rows)):
            audio_rel_path = row['audio_path'].strip()
            split = row.get('split', default_split).lower()
            left_start = float(row['left_start'])
            right_end = float(row['right_end'])

            # Determine output dir
            out_dir = train_dir if split == 'train' else test_dir

            # File index: use test_row_index if available, else row index
            file_index = row.get('test_row_index', idx)

            out_path = out_dir / f"{file_index}.wav"

            # Resolve audio path
            audio_path = resolve_audio_path(audio_rel_path, base_dir)
            if audio_path is None or not audio_path.exists():
                print(f"Audio not found: {audio_rel_path}")
                continue

            try:
                wav, sr = load_audio(audio_path, args.sr)
            except Exception as e:
                print(f"Failed to load {audio_rel_path}: {e}")
                continue

            # Extract 4s segment
            segment = extract_window(wav, sr, left_start, right_end)

            # Save
            sf.write(str(out_path), segment, sr)
            total_processed += 1

    print(f"Processed {total_processed} segments.")

    # Create tar archive (no compression)
    tar_path = base_dir / "train_test2.tar"
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(str(output_base), arcname='train_test2')

    print(f"Tar archive created: {tar_path}")

if __name__ == "__main__":
    main()