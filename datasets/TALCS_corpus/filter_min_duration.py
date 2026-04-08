import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf


SPLITS = ("train_set", "dev_set", "test_set")


def get_duration(audio_path: Path) -> tuple[str, float]:
    info = sf.info(str(audio_path))
    return audio_path.stem, float(info.duration)


def process_split(base_dir: Path, split: str, min_sec: float, workers: int) -> dict:
    split_dir = base_dir / split
    wav_dir = split_dir / "wav"
    label_path = split_dir / "label.txt"

    wav_paths = sorted(wav_dir.glob("*.wav"))
    durations = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for stem, duration in executor.map(get_duration, wav_paths, chunksize=128):
            durations[stem] = duration

    keep_stems = {stem for stem, duration in durations.items() if duration >= min_sec}
    remove_paths = [path for path in wav_paths if path.stem not in keep_stems]

    kept_lines = []
    removed_label_lines = 0
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            stem = line.split(maxsplit=1)[0]
            if stem in keep_stems:
                kept_lines.append(raw_line)
            else:
                removed_label_lines += 1

    for path in remove_paths:
        path.unlink()

    with label_path.open("w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    return {
        "split": split,
        "min_sec": min_sec,
        "original_wav_count": len(wav_paths),
        "kept_wav_count": len(keep_stems),
        "removed_wav_count": len(remove_paths),
        "original_label_count": len(kept_lines) + removed_label_lines,
        "kept_label_count": len(kept_lines),
        "removed_label_count": removed_label_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-sec", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    summary = []
    for split in SPLITS:
        summary.append(process_split(base_dir, split, args.min_sec, args.workers))

    summary_path = base_dir / "filter_min_duration_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
