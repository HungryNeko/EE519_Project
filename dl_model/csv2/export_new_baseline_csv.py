import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parent


def resolve_case_insensitive(path: Path):
    if path.exists():
        return path

    parts = path.parts
    current = Path(parts[0]) if path.is_absolute() else Path()

    for part in parts:
        if not current.exists():
            return None

        try:
            candidates = list(current.iterdir())
        except Exception:
            return None

        match = None
        for candidate in candidates:
            if candidate.name.lower() == part.lower():
                match = candidate
                break

        if match is None:
            return None

        current = match

    return current


def iter_switch_samples(item):
    segments = item.get("segments", [])

    for segment in segments:
        spans = segment.get("language_spans", [])
        if len(spans) < 2:
            continue

        for i in range(len(spans) - 1):
            left = spans[i]
            right = spans[i + 1]
            if left.get("language") == right.get("language"):
                continue

            yield {
                "segment_id": segment.get("segment_id"),
                "switch_index": i,
                "switch_time": float(left.get("end", 0.0)),
                "gap_start": float(left.get("end", 0.0)),
                "gap_end": float(right.get("start", left.get("end", 0.0))),
            }

    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]

        spans1 = seg1.get("language_spans", [])
        spans2 = seg2.get("language_spans", [])
        if (not spans1) or (not spans2):
            continue

        if spans1[-1].get("language") == spans2[0].get("language"):
            continue

        yield {
            "segment_id": f"{seg1.get('segment_id')}_{seg2.get('segment_id')}",
            "switch_index": i,
            "switch_time": float(seg1.get("end", 0.0)),
            "gap_start": float(seg1.get("end", 0.0)),
            "gap_end": float(seg2.get("start", seg1.get("end", 0.0))),
        }


def load_cache_records(cache_path: Path):
    cache_records = []
    json_order = []
    seen_json = set()
    audio_counter = Counter()

    with open(cache_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except Exception:
                continue

            audio_path = str(record["audio_path"]).replace("\\", "/")
            json_path = str(record["json_path"]).replace("\\", "/")

            if json_path not in seen_json:
                seen_json.add(json_path)
                json_order.append(json_path)

            cache_records.append(
                {
                    "cache_line_number": line_no,
                    "json_path": json_path,
                    "audio_path": audio_path,
                    "is_switch": record.get("is_switch", False),
                    "test_row_index": record.get("test_row_index"),
                    "csv_index": record.get("csv_index"),
                    "id": record.get("id"),
                    "index": record.get("index"),
                }
            )
            audio_counter[audio_path] += 1

    return cache_records, json_order, audio_counter


def build_source_samples_by_json(root: Path, json_order, window_sec: float):
    samples_by_json = {}

    for json_rel in json_order:
        json_path = resolve_case_insensitive(root / json_rel)
        if json_path is None:
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        audio_counter = Counter()

        for item in data:
            audio_path = str(item.get("path", "")).replace("\\", "/")
            audio_counter[audio_path] += 1

            for sample in iter_switch_samples(item):
                switch_time = sample["switch_time"]
                gap_end = sample["gap_end"]

                samples.append(
                    {
                        "json_path": json_rel,
                        "audio_path": audio_path,
                        "segment_id": sample["segment_id"],
                        "switch_index": sample["switch_index"],
                        "switch_time": switch_time,
                        "window_sec": float(window_sec),
                        "gap_start": float(sample["gap_start"]),
                        "gap_end": gap_end,
                        "left_start": switch_time - 2.0,  # 前一个语言结束前2s
                        "left_end": switch_time,
                        "right_start": switch_time,
                        "right_end": gap_end + 2.0,  # 后一个语言开始后2s
                        "audio_occurrence_in_source_json": audio_counter[audio_path],
                    }
                )

        samples_by_json[json_rel] = samples

    return samples_by_json


def align_cache_with_source(cache_records, samples_by_json):
    pointers = defaultdict(int)
    aligned = []

    for record in cache_records:
        json_path = record["json_path"]
        wanted_audio = record["audio_path"]
        samples = samples_by_json[json_path]

        idx = pointers[json_path]
        while idx < len(samples):
            if samples[idx]["audio_path"] == wanted_audio:
                break
            idx += 1

        if idx >= len(samples):
            raise RuntimeError(
                "Failed to align cache record with source sample: "
                f"cache_line={record['cache_line_number']} "
                f"json_path={json_path} audio_path={wanted_audio}"
            )

        source_sample = samples[idx]
        pointers[json_path] = idx + 1
        aligned.append({**record, **source_sample})

    return aligned


def split_train_test(records, seed=42):
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    return records[: int(n * 0.8)], records[int(n * 0.8) :]


def balance(records):
    same = [r for r in records if not r["is_switch"]]
    diff = [r for r in records if r["is_switch"]]

    min_count = min(len(same), len(diff))
    same = same[:min_count]
    diff = diff[:min_count]

    balanced = same + diff
    random.Random(42).shuffle(balanced)
    return balanced


def write_baseline_csv(all_records, output_path: Path, root: Path, window_sec: float):
    fieldnames = [
        "audio_path",
        "is_switch",
        "split",
        "left_start",
        "switch_time",
        "right_end",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in all_records:
            writer.writerow(
                {
                    "audio_path": record["audio_path"],
                    "is_switch": record["is_switch"],
                    "split": record["split"],
                    "left_start": f"{record['left_start']:.6f}",
                    "switch_time": f"{record['switch_time']:.6f}",
                    "right_end": f"{record['right_end']:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Export train and test segments used by train_net.py for baseline models."
    )
    parser.add_argument(
        "--output",
        default="dl_model/csv2/baseline_train_test_segments.csv",
        help="Output CSV path (contains both train and test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window size on each side of switch_time (default: 1.0, total 2.0s)",
    )
    args = parser.parse_args()

    root = project_root()
    output_path = root / args.output
    print(f"Root: {root}")
    print(f"Output path: {output_path}")

    # Load JSON files directly
    json_files = [
        "datasets/hinglish/whisper_segment_hinglish.json",
        "datasets/crossfade_insertions/whisper_segment_crossfade_insertions.json",
        "datasets/corpus/whisper_segment_corpus.json",
        "datasets/ascend/whisper_segment_ascend.json",
    ]

    all_samples = []
    for json_rel in json_files:
        json_path = root / json_rel
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            audio_path = str(item.get("path", "")).replace("\\", "/")

            for sample in iter_switch_samples(item):
                switch_time = sample["switch_time"]
                gap_end = sample["gap_end"]

                all_samples.append(
                    {
                        "audio_path": audio_path,
                        "is_switch": True,  # Assume all are switches for now
                        "split": "train",
                        "left_start": switch_time - 2.0,
                        "switch_time": switch_time,
                        "right_end": gap_end + 2.0,
                    }
                )

    # For simplicity, take a subset
    random.Random(args.seed).shuffle(all_samples)
    all_records = all_samples[:1000]  # Limit for testing

    write_baseline_csv(all_records, output_path, root, args.window_sec)

    print(f"Output written to: {output_path}")
    print(f"Total records: {len(all_records)}")


if __name__ == "__main__":
    main()