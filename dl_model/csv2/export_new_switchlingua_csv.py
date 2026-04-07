import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_audio_path(raw_path: str) -> str:
    return str(raw_path).replace("\\", "/").strip()


def iter_switch_samples(item):
    audio_path = normalize_audio_path(item.get("path", ""))
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
                "audio_path": audio_path,
                "switch_time": float(left.get("end", 0.0)),
                "gap_start": float(left.get("end", 0.0)),
                "gap_end": float(right.get("start", left.get("end", 0.0))),
            }

    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        spans1 = seg1.get("language_spans", [])
        spans2 = seg2.get("language_spans", [])
        if not spans1 or not spans2:
            continue
        if spans1[-1].get("language") == spans2[0].get("language"):
            continue

        yield {
            "audio_path": audio_path,
            "switch_time": float(seg1.get("end", 0.0)),
            "gap_start": float(seg1.get("end", 0.0)),
            "gap_end": float(seg2.get("start", seg1.get("end", 0.0))),
        }


def iter_true_insert_switches(item):
    audio_path = normalize_audio_path(item.get("path", ""))
    for switch in item.get("true_insert_switches", []):
        expected_time = switch.get("expected_time")
        if not isinstance(expected_time, (int, float)):
            continue
        yield {
            "audio_path": audio_path,
            "switch_time": float(expected_time),
            "gap_start": float(expected_time),
            "gap_end": float(expected_time),
        }


def load_json_list(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_source_samples(root: Path, audio_paths, window_sec: float):
    samples_by_key = defaultdict(list)
    json_files = list((root / "datasets").rglob("*.json"))

    for json_path in json_files:
        try:
            data = load_json_list(json_path)
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        for item in data:
            audio_path = normalize_audio_path(item.get("path", ""))
            if audio_path not in audio_paths:
                continue

            for sample in iter_switch_samples(item):
                key = (sample["audio_path"].lower(), f"{sample['switch_time']:.6f}")
                samples_by_key[key].append(sample)

            for sample in iter_true_insert_switches(item):
                key = (sample["audio_path"].lower(), f"{sample['switch_time']:.6f}")
                samples_by_key[key].append(sample)

    return samples_by_key


def load_original_rows(input_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match_rows_to_samples(rows, samples_by_key):
    matched_rows = []
    missing = []

    for row in rows:
        audio_path = normalize_audio_path(row["audio_path"])
        switch_time = float(row["switch_time"])
        key = (audio_path.lower(), f"{switch_time:.6f}")
        if key not in samples_by_key or not samples_by_key[key]:
            missing.append((audio_path, switch_time))
            matched_rows.append({**row, "_sample": None})
            continue
        sample = samples_by_key[key].pop(0)
        matched_rows.append({**row, "_sample": sample})

    if missing:
        missing_sample = missing[0]
        raise RuntimeError(
            f"Missing JSON sample for audio_path={missing_sample[0]} switch_time={missing_sample[1]:.6f}"
        )

    return matched_rows


def write_new_switchlingua_csv(rows, output_path: Path, window_sec: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_row_index",
                "audio_path",
                "is_switch",
                "split",
                "left_start",
                "left_end",
                "switch_time",
                "right_start",
                "right_end",
            ],
        )
        writer.writeheader()
        for row in rows:
            sample = row["_sample"]
            writer.writerow(
                {
                    "test_row_index": row.get("test_row_index"),
                    "audio_path": normalize_audio_path(row["audio_path"]),
                    "is_switch": row.get("is_switch", "0"),
                    "split": row.get("split", "test"),
                    "left_start": f"{sample['gap_start'] - window_sec:.6f}",
                    "left_end": f"{sample['gap_start']:.6f}",
                    "switch_time": f"{sample['switch_time']:.6f}",
                    "right_start": f"{sample['gap_end']:.6f}",
                    "right_end": f"{sample['gap_end'] + window_sec:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a csv2 switchlingua test CSV from original switchlingua CSV and source JSONs."
    )
    parser.add_argument(
        "--input",
        default="dl_model/baseline_train_test_segments_switchlingua_seame.csv",
        help="Original switchlingua CSV path",
    )
    parser.add_argument(
        "--output",
        default="dl_model/csv2/baseline_train_test_segments_switchlingua_seame.csv",
        help="Output csv2 switchlingua CSV path",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Window length for each side in seconds",
    )
    args = parser.parse_args()

    root = project_root()
    input_path = root / args.input
    output_path = root / args.output

    rows = load_original_rows(input_path)
    audio_paths = {normalize_audio_path(row["audio_path"]) for row in rows}
    sample_index = collect_source_samples(root, audio_paths, args.window_sec)
    matched_rows = match_rows_to_samples(rows, sample_index)
    write_new_switchlingua_csv(matched_rows, output_path, args.window_sec)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()