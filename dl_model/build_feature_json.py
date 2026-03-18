import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from functions import SpeakerFeatureExtractor


FEATURE_NAMES = [
    "emb_cos",
    "emb_l2",
    "emb_ratio",
    "pitch_mean_diff",
    "pitch_std_diff",
    "voiced_diff",
    "duration_diff",
    "time_gap",
]


def project_root():
    return Path(__file__).resolve().parents[1]


def load_used_json_list(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_audio(path: Path, sr=16000):
    wav, src_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if src_sr != sr:
        wav = librosa.resample(wav, orig_sr=src_sr, target_sr=sr)
    return wav.astype(np.float32)


def extract_window(wav: np.ndarray, sr: int, start_time: float, end_time: float):
    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    length = max(1, end_i - start_i)
    out = np.zeros(length, dtype=np.float32)

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)
    if src_end <= src_start:
        return out

    dst_start = src_start - start_i
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = wav[src_start:src_end]
    return out


def iter_switch_samples(item):
    audio_rel = item.get("path")
    for segment in item.get("segments", []):
        spans = segment.get("language_spans", [])
        if len(spans) < 2:
            continue

        for i in range(len(spans) - 1):
            left = spans[i]
            right = spans[i + 1]
            if left.get("language") == right.get("language"):
                continue

            yield {
                "audio_rel_path": audio_rel,
                "segment_id": segment.get("segment_id"),
                "segment_start": segment.get("start"),
                "segment_end": segment.get("end"),
                "segment_text": segment.get("text"),
                "left_span": left,
                "right_span": right,
                "switch_time": float(left.get("end", 0.0)),
                "gap_start": float(left.get("end", 0.0)),
                "gap_end": float(right.get("start", left.get("end", 0.0))),
                "switch_index": i,
            }


def make_sample_key(source_json: Path, audio_path: Path, sample):
    switch_time = round(float(sample["switch_time"]), 6)
    segment_id = sample.get("segment_id")
    switch_index = sample.get("switch_index")
    return f"{source_json.as_posix()}|{audio_path.as_posix()}|{segment_id}|{switch_index}|{switch_time}"


def build_record(extractor, wav, audio_path: Path, source_json: Path, sample, window_sec=1.0):
    switch_time = sample["switch_time"]
    seg1 = extract_window(wav, extractor.sr, switch_time - window_sec, switch_time)
    seg2 = extract_window(wav, extractor.sr, switch_time, switch_time + window_sec)

    feature_vector = extractor.build_features(
        seg1,
        seg2,
        t1_end=sample["gap_start"],
        t2_start=sample["gap_end"],
    )
    feature_values = {name: float(value) for name, value in zip(FEATURE_NAMES, feature_vector.tolist())}

    root = project_root()
    source_json_str = source_json.relative_to(root).as_posix()
    audio_path_str = audio_path.relative_to(root).as_posix()
    is_switch = "crossfade_insertions" not in source_json_str.lower()

    return {
        "json_path": source_json_str,
        "json_name": source_json.name,
        "audio_path": audio_path_str,
        "audio_name": audio_path.name,
        "feature": feature_values,
        "is_switch": is_switch,
    }


def load_existing_state(output_path: Path, progress_path: Path):
    records = []
    completed_keys = set()

    if output_path.exists():
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except Exception:
            records = []

    if progress_path.exists():
        try:
            loaded = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_keys = set(loaded.get("completed_keys", []))
        except Exception:
            completed_keys = set()

    return records, completed_keys


def dedupe_records_by_json_name(records):
    deduped = {}
    for record in records:
        json_name = record.get("json_name")
        if json_name:
            deduped[json_name] = record
    return list(deduped.values())


def save_state(output_path: Path, progress_path: Path, records, completed_keys):
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    progress = {"completed_keys": sorted(completed_keys)}
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def process_json_file(extractor, json_path: Path, records, completed_keys,
                      output_path: Path, progress_path: Path,
                      test_mode=False, window_sec=1.0):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    root = project_root()
    new_count = 0

    for item in data:
        audio_rel = item.get("path")
        if not audio_rel:
            continue

        audio_path = root / audio_rel
        if not audio_path.exists():
            print(f"missing audio: {Path(audio_rel).as_posix()}")
            continue

        try:
            wav = load_audio(audio_path, sr=extractor.sr)
        except Exception:
            print(f"failed audio load: {Path(audio_rel).as_posix()}")
            continue

        for sample in iter_switch_samples(item):
            sample_key = make_sample_key(json_path, audio_path, sample)
            if sample_key in completed_keys:
                continue

            records.append(build_record(extractor, wav, audio_path, json_path, sample, window_sec=window_sec))
            completed_keys.add(sample_key)
            new_count += 1
            save_state(output_path, progress_path, records, completed_keys)

            if test_mode:
                return new_count

    return new_count


def main():
    parser = argparse.ArgumentParser(description="Build cached MLP training JSON from used_json.txt")
    parser.add_argument("--used-json", default="dl_model/used_json.txt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--test", action="store_true", help="Keep one sample from each source JSON file")
    parser.add_argument("--window-sec", type=float, default=1.0)
    args = parser.parse_args()

    root = project_root()
    used_json_path = root / args.used_json
    json_files = load_used_json_list(used_json_path)
    if args.output is None:
        output_name = "mlp_feature_cache_test.json" if args.test else "mlp_feature_cache.json"
        output_path = root / "dl_model" / output_name
    else:
        output_path = root / args.output
    progress_path = output_path.with_name(output_path.name + ".progress.json")

    extractor = SpeakerFeatureExtractor(sr=16000)
    all_records, completed_keys = load_existing_state(output_path, progress_path)
    if args.test:
        all_records = dedupe_records_by_json_name(all_records)
    total_new = 0

    for rel_path in json_files:
        json_path = root / rel_path
        if not json_path.exists():
            print(f"skip missing json: {json_path}")
            continue
        new_count = process_json_file(
            extractor=extractor,
            json_path=json_path,
            records=all_records,
            completed_keys=completed_keys,
            output_path=output_path,
            progress_path=progress_path,
            test_mode=args.test,
            window_sec=args.window_sec,
        )
        if new_count == 0:
            print(f"no valid sample found for {json_path}")
        total_new += new_count

    save_state(output_path, progress_path, all_records, completed_keys)
    print(f"saved {len(all_records)} total records to {output_path}")
    print(f"added {total_new} new records")
    print(f"progress file: {progress_path}")


if __name__ == "__main__":
    main()
