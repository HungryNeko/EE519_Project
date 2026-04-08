import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf


def get_duration(audio_path: Path) -> tuple[str, float]:
    info = sf.info(str(audio_path))
    return audio_path.name, float(info.duration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="datasets/Hindi-English_train/train")
    parser.add_argument("--min-sec", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    wav_paths = sorted(audio_dir.glob("*.wav"))

    durations = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for name, duration in executor.map(get_duration, wav_paths, chunksize=128):
            durations[name] = duration

    keep_names = {name for name, duration in durations.items() if duration >= args.min_sec}
    remove_paths = [path for path in wav_paths if path.name not in keep_names]

    for path in remove_paths:
        path.unlink()

    summary = {
        "audio_dir": audio_dir.as_posix(),
        "min_sec": args.min_sec,
        "original_wav_count": len(wav_paths),
        "kept_wav_count": len(keep_names),
        "removed_wav_count": len(remove_paths),
    }

    summary_path = audio_dir.parent / "filter_min_duration_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
