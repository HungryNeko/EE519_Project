import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

BASE_DIR = Path("datasets/crossfade_hindi_english_talcs")
INPUT_MIXED_JSON = BASE_DIR / "crossfade_hindi_english_talcs_mixed_language.json"
OUTPUT_MIXED_JSON = BASE_DIR / "crossfade_hindi_english_talcs_mixed_language_cleaned.json"
OUTPUT_STATS_JSON = BASE_DIR / "crossfade_hindi_english_talcs_mixed_language_cleaned_stats.json"

TIME_TOLERANCE_SEC = 0.5
ALLOWED_LANGS = {"en", "hi", "zh"}


def normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    path_lower = path.lower()
    base_dir = str(BASE_DIR).replace("\\", "/")
    base_dir_lower = base_dir.lower()
    if base_dir_lower in path_lower:
        start = path_lower.index(base_dir_lower)
        return path[start:]
    return path


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def find_matching_source_spans(item: dict, manifest_row: dict, tolerance_sec: float) -> list[dict]:
    insert_start = manifest_row["insert_at_sec"]
    insert_end = insert_start + manifest_row["insert_duration_sec"]
    allowed_start = insert_start - tolerance_sec
    allowed_end = insert_end + tolerance_sec
    source_lang = manifest_row["insert_source_whisper_language"]

    matches = []
    for seg in item.get("segments", []):
        for span in seg.get("language_spans", []):
            if span.get("language") != source_lang:
                continue

            overlap_sec = interval_overlap(
                span.get("start", 0.0),
                span.get("end", 0.0),
                allowed_start,
                allowed_end,
            )
            if overlap_sec <= 0:
                continue

            matches.append(
                {
                    "segment_id": seg.get("segment_id"),
                    "span_start": span.get("start"),
                    "span_end": span.get("end"),
                    "language": source_lang,
                    "overlap_sec": round(overlap_sec, 6),
                }
            )
    return matches


def trim_item_to_insert_window(item: dict, manifest_row: dict, tolerance_sec: float) -> tuple[dict, int, int, int]:
    insert_start = manifest_row["insert_at_sec"]
    insert_end = insert_start + manifest_row["insert_duration_sec"]
    allowed_start = insert_start - tolerance_sec
    allowed_end = insert_end + tolerance_sec

    trimmed_item = deepcopy(item)
    kept_segments = []
    removed_spans = 0
    removed_disallowed_spans = 0
    kept_spans = 0

    for seg in trimmed_item.get("segments", []):
        original_spans = seg.get("language_spans", [])
        trimmed_spans = []
        for span in original_spans:
            if span.get("language") not in ALLOWED_LANGS:
                removed_disallowed_spans += 1
                continue

            overlap_sec = interval_overlap(
                span.get("start", 0.0),
                span.get("end", 0.0),
                allowed_start,
                allowed_end,
            )
            if overlap_sec <= 0:
                removed_spans += 1
                continue

            trimmed_span = deepcopy(span)
            trimmed_span["start"] = max(span.get("start", 0.0), allowed_start)
            trimmed_span["end"] = min(span.get("end", 0.0), allowed_end)
            trimmed_spans.append(trimmed_span)
            kept_spans += 1

        if not trimmed_spans:
            continue

        seg["language_spans"] = trimmed_spans
        seg["start"] = min(span["start"] for span in trimmed_spans)
        seg["end"] = max(span["end"] for span in trimmed_spans)
        kept_segments.append(seg)

    trimmed_item["segments"] = kept_segments
    return trimmed_item, kept_spans, removed_spans, removed_disallowed_spans


def load_manifest_map() -> dict:
    manifest_map = {}
    for manifest_path in BASE_DIR.rglob("mixed_manifest.json"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            norm_path = normalize_path(row["path"]).lower()
            row["path"] = normalize_path(row["path"])
            row["manifest_path"] = str(manifest_path).replace("\\", "/")
            manifest_map[norm_path] = row
    return manifest_map


with open(INPUT_MIXED_JSON, "r", encoding="utf-8") as f:
    mixed_data = json.load(f)

manifest_map = load_manifest_map()

cleaned_mixed = []
stats = Counter()
folder_stats = defaultdict(Counter)
drop_examples = []

for item in mixed_data:
    path = normalize_path(item.get("path", ""))
    if path:
        item["path"] = path

    subgroup = Path(path).parts[2] if len(Path(path).parts) > 2 else "unknown"
    folder_stats[subgroup]["total_mixed"] += 1
    stats["total_mixed"] += 1

    manifest_row = manifest_map.get(path.lower())
    if not manifest_row:
        stats["missing_manifest"] += 1
        folder_stats[subgroup]["missing_manifest"] += 1
        if len(drop_examples) < 20:
            drop_examples.append({"path": path, "reason": "missing_manifest"})
        continue

    matches = find_matching_source_spans(item, manifest_row, TIME_TOLERANCE_SEC)
    if matches:
        trimmed_item, kept_spans, removed_spans, removed_disallowed_spans = trim_item_to_insert_window(
            item, manifest_row, TIME_TOLERANCE_SEC
        )
        if not trimmed_item.get("segments"):
            stats["dropped_empty_after_trim"] += 1
            folder_stats[subgroup]["dropped_empty_after_trim"] += 1
            continue

        trimmed_item["path"] = path
        trimmed_item["audio_name"] = Path(path).name
        cleaned_mixed.append(trimmed_item)
        stats["kept"] += 1
        stats["kept_spans_near_insert"] += kept_spans
        stats["removed_spans_outside_insert_window"] += removed_spans
        stats["removed_spans_disallowed_language"] += removed_disallowed_spans
        folder_stats[subgroup]["kept"] += 1
        folder_stats[subgroup]["kept_spans_near_insert"] += kept_spans
        folder_stats[subgroup]["removed_spans_outside_insert_window"] += removed_spans
        folder_stats[subgroup]["removed_spans_disallowed_language"] += removed_disallowed_spans
    else:
        stats["dropped_time_mismatch"] += 1
        folder_stats[subgroup]["dropped_time_mismatch"] += 1
        if len(drop_examples) < 20:
            drop_examples.append(
                {
                    "path": path,
                    "reason": "time_mismatch",
                    "insert_at_sec": manifest_row["insert_at_sec"],
                    "insert_duration_sec": manifest_row["insert_duration_sec"],
                    "insert_source_whisper_language": manifest_row["insert_source_whisper_language"],
                }
            )

stats["kept_ratio_percent"] = round(
    100.0 * stats["kept"] / stats["total_mixed"], 2
) if stats["total_mixed"] else 0.0

folder_summary = {}
for subgroup, counter in sorted(folder_stats.items()):
    total = counter["total_mixed"]
    kept = counter["kept"]
    folder_summary[subgroup] = {
        "total_mixed": total,
        "kept": kept,
        "dropped_time_mismatch": counter["dropped_time_mismatch"],
        "dropped_empty_after_trim": counter["dropped_empty_after_trim"],
        "missing_manifest": counter["missing_manifest"],
        "kept_spans_near_insert": counter["kept_spans_near_insert"],
        "removed_spans_outside_insert_window": counter["removed_spans_outside_insert_window"],
        "removed_spans_disallowed_language": counter["removed_spans_disallowed_language"],
        "kept_ratio_percent": round(100.0 * kept / total, 2) if total else 0.0,
    }

stats_payload = {
    "input_json": str(INPUT_MIXED_JSON).replace("\\", "/"),
    "output_json": str(OUTPUT_MIXED_JSON).replace("\\", "/"),
    "time_tolerance_sec": TIME_TOLERANCE_SEC,
    "allowed_languages": sorted(ALLOWED_LANGS),
    "overall": dict(stats),
    "by_subfolder": folder_summary,
    "drop_examples": drop_examples,
}

with open(OUTPUT_MIXED_JSON, "w", encoding="utf-8") as f:
    json.dump(cleaned_mixed, f, ensure_ascii=False, indent=2)

with open(OUTPUT_STATS_JSON, "w", encoding="utf-8") as f:
    json.dump(stats_payload, f, ensure_ascii=False, indent=2)

print(f"Input mixed     : {stats['total_mixed']}")
print(f"Kept            : {stats['kept']}")
print(f"Dropped mismatch: {stats['dropped_time_mismatch']}")
print(f"Dropped empty   : {stats['dropped_empty_after_trim']}")
print(f"Missing manifest: {stats['missing_manifest']}")
print(f"Kept spans      : {stats['kept_spans_near_insert']}")
print(f"Removed spans   : {stats['removed_spans_outside_insert_window']}")
print(f"Removed other   : {stats['removed_spans_disallowed_language']}")
print(f"Kept ratio      : {stats['kept_ratio_percent']}%")
print(f"Output mixed    : {OUTPUT_MIXED_JSON}")
print(f"Output stats    : {OUTPUT_STATS_JSON}")

for subgroup, summary in folder_summary.items():
    print(
        f"{subgroup}: total={summary['total_mixed']}, "
        f"kept={summary['kept']}, "
        f"dropped={summary['dropped_time_mismatch']}, "
        f"dropped_empty={summary['dropped_empty_after_trim']}, "
        f"missing_manifest={summary['missing_manifest']}, "
        f"kept_spans={summary['kept_spans_near_insert']}, "
        f"removed_spans={summary['removed_spans_outside_insert_window']}, "
        f"removed_other={summary['removed_spans_disallowed_language']}, "
        f"kept_ratio={summary['kept_ratio_percent']}%"
    )
