from pathlib import Path
import json
from datetime import datetime, timezone
from collections import deque
from typing import Any, Dict, List, Set
import argparse

import whisper
import whisperx
from tqdm import tqdm

# ======================
# Configuration
# ======================
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
MODEL_NAME = "large-v3"
DEVICE = "cuda"
USE_WHISPERX_ALIGN = False

# ======================
# Helpers
# ======================
def collect_all_audio_files(root: Path) -> List[Path]:
    dir_queue = deque([root])
    files_by_dir: List[List[Path]] = []

    while dir_queue:
        current_dir = dir_queue.popleft()
        try:
            children = sorted(current_dir.iterdir(), key=lambda p: p.name.lower())
        except Exception:
            continue

        dir_files = [
            p for p in children
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
        if dir_files:
            files_by_dir.append(dir_files)

        for child in children:
            if child.is_dir():
                dir_queue.append(child)

    ordered_files: List[Path] = []
    indexes = [0] * len(files_by_dir)
    made_progress = True
    while made_progress:
        made_progress = False
        for i, dir_files in enumerate(files_by_dir):
            if indexes[i] < len(dir_files):
                ordered_files.append(dir_files[indexes[i]])
                indexes[i] += 1
                made_progress = True

    return ordered_files

def to_jsonable(x: Any):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x

def canonical_path_for_storage(p: Path) -> str:
    return p.resolve(strict=False).as_posix().replace("\\", "/").lower()

def corpus_relative_identity(p: Path, dataset: str) -> str:
    p_norm = p.resolve(strict=False).as_posix().replace("\\", "/").lower()
    anchor = f"/datasets/{dataset.lower()}/"
    idx = p_norm.find(anchor)
    if idx == -1:
        raise ValueError(f"path not under datasets/{dataset}: {p}")
    return p_norm[idx + 1:]

# ======================
# JSON IO
# ======================
def load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []

def save_records(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def iter_jsonl_records(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row

def append_jsonl_record(path: Path, record: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")

def read_paths_from_records(path: Path, dataset: str) -> Set[str]:
    rows = load_existing_records(path)
    out = set()
    for r in rows:
        raw = r.get("path")
        if not raw:
            continue
        try:
            out.add(corpus_relative_identity(Path(raw), dataset))
        except Exception:
            continue
    return out

def read_paths_from_jsonl(path: Path, dataset: str) -> Set[str]:
    out = set()
    for r in iter_jsonl_records(path):
        raw = r.get("path")
        if not raw:
            continue
        try:
            out.add(corpus_relative_identity(Path(raw), dataset))
        except Exception:
            continue
    return out

def finalize_json_output(json_path: Path, jsonl_path: Path):
    if not jsonl_path.exists():
        return

    if json_path.exists():
        merged = load_existing_records(json_path)
        merged.extend(iter_jsonl_records(jsonl_path))
        save_records(json_path, merged)
    else:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            f.write("[\n")
            first = True
            for row in iter_jsonl_records(jsonl_path):
                if not first:
                    f.write(",\n")
                json.dump(row, f, ensure_ascii=False, indent=2)
                first = False
            f.write("\n]\n")

    jsonl_path.unlink()

# ======================
# Language detection
# ======================
def detect_lang_by_char(ch: str) -> str:
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return "zh"
    if 0x0900 <= o <= 0x097F:
        return "hi"
    if "a" <= ch.lower() <= "z":
        return "en"
    return "other"

# ======================
# Core processing
# ======================
def process_single_audio(whisper_model, audio_path: Path):
    storage_path = canonical_path_for_storage(audio_path)
    full_audio = whisper.load_audio(str(audio_path))
    sr = whisper.audio.SAMPLE_RATE

    base_result = whisper_model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,
        verbose=False,
        fp16=False,
        word_timestamps=False,
    )

    segments_out = []

    for i, seg in enumerate(base_result.get("segments", [])):
        seg_start = seg.get("start")
        seg_end = seg.get("end")

        seg_audio = full_audio[int(seg_start * sr): int(seg_end * sr)]

        seg_result = whisper_model.transcribe(
            seg_audio,
            task="transcribe",
            language=None,
            verbose=False,
            fp16=False,
            temperature=0.2,
            beam_size=1,
            word_timestamps=True,
        )

        seg_language = seg_result.get("language", "unknown")

        words = []
        for wseg in seg_result.get("segments", []):
            for w in wseg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": seg_start + w["start"],
                    "end": seg_start + w["end"],
                    "score": w.get("probability", 0.0),
                })

        language_spans = []
        cur = None

        def flush():
            nonlocal cur
            if cur:
                cur["text"] = cur["text"].strip()
                cur["score"] = cur["score_sum"] / max(cur["count"], 1)
                cur.pop("score_sum")
                cur.pop("count")
                language_spans.append(cur)
                cur = None

        for w in words:
            if not w["word"]:
                continue
            lang = detect_lang_by_char(w["word"][0])
            if lang == "other":
                flush()
                continue
            if cur is None or cur["language"] != lang:
                flush()
                cur = {
                    "language": lang,
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["word"],
                    "score_sum": w["score"] or 0.0,
                    "count": 1,
                }
            else:
                cur["end"] = w["end"]
                cur["text"] += " " + w["word"]
                cur["score_sum"] += w["score"] or 0.0
                cur["count"] += 1

        flush()

        if not language_spans:
            language_spans = [{
                "language": seg_language,
                "start": seg_start,
                "end": seg_end,
                "text": seg_result.get("text", "").strip(),
                "score": seg.get("avg_logprob"),
            }]

        segments_out.append({
            "segment_id": i,
            "start": seg_start,
            "end": seg_end,
            "text": seg_result.get("text", "").strip(),
            "scores": {
                "avg_logprob": seg.get("avg_logprob"),
                "compression_ratio": seg.get("compression_ratio"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "temperature": seg.get("temperature"),
            },
            "language_spans": language_spans,
        })

    return to_jsonable({
        "path": storage_path,
        "whisper_language": "segment_based",
        "segments": segments_out,
    })

def build_language_switch_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    spans = []
    for seg in record.get("segments", []):
        seg_id = seg.get("segment_id")
        for span in seg.get("language_spans", []):
            spans.append({
                "segment_id": seg_id,
                "language": span.get("language", "unknown"),
                "start": span.get("start"),
                "end": span.get("end"),
            })

    spans = [
        s for s in spans
        if s["start"] is not None and s["end"] is not None and s["language"] is not None
    ]
    spans.sort(key=lambda s: (s["start"], s["end"]))

    switches = []
    prev = None
    for s in spans:
        if prev is None:
            prev = s
            continue
        if s["language"] != prev["language"]:
            switches.append({
                "from_language": prev["language"],
                "to_language": s["language"],
                "switch_time": s["start"],
                "from_segment_id": prev["segment_id"],
                "to_segment_id": s["segment_id"],
            })
        prev = s

    return {
        "path": record.get("path"),
        "switch_count": len(switches),
        "switch_positions": switches,
    }

# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset

    CORPUS_ROOT = Path(f"./datasets/{dataset}")
    OUTPUT_JSON = CORPUS_ROOT / f"whisper_segment_{dataset}.json"
    FAILED_JSON = CORPUS_ROOT / f"whisper_failed_{dataset}.json"
    SWITCH_JSON = CORPUS_ROOT / f"whisper_language_switch_{dataset}.json"
    OUTPUT_JSONL = CORPUS_ROOT / f"whisper_segment_{dataset}.jsonl"
    FAILED_JSONL = CORPUS_ROOT / f"whisper_failed_{dataset}.jsonl"
    SWITCH_JSONL = CORPUS_ROOT / f"whisper_language_switch_{dataset}.jsonl"

    if args.finalize_only:
        finalize_json_output(OUTPUT_JSON, OUTPUT_JSONL)
        finalize_json_output(FAILED_JSON, FAILED_JSONL)
        finalize_json_output(SWITCH_JSON, SWITCH_JSONL)
        return

    processed_set = (
        read_paths_from_records(OUTPUT_JSON, dataset)
        | read_paths_from_jsonl(OUTPUT_JSONL, dataset)
    )
    failed_set = (
        read_paths_from_records(FAILED_JSON, dataset)
        | read_paths_from_jsonl(FAILED_JSONL, dataset)
    )
    switch_set = (
        read_paths_from_records(SWITCH_JSON, dataset)
        | read_paths_from_jsonl(SWITCH_JSONL, dataset)
    )

    audio_files = collect_all_audio_files(CORPUS_ROOT)
    files_and_keys = [(p, corpus_relative_identity(p, dataset)) for p in audio_files]

    skip_set = processed_set | failed_set
    to_process = [p for p, key in files_and_keys if key not in skip_set]

    whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)

    for audio_path in tqdm(to_process, desc="Processing", unit="file"):
        try:
            record = process_single_audio(whisper_model, audio_path)
            key = corpus_relative_identity(audio_path, dataset)
            append_jsonl_record(OUTPUT_JSONL, record)
            processed_set.add(key)

            if key not in switch_set:
                append_jsonl_record(SWITCH_JSONL, build_language_switch_summary(record))
                switch_set.add(key)

        except Exception as e:
            key = corpus_relative_identity(audio_path, dataset)
            if key not in failed_set:
                append_jsonl_record(FAILED_JSONL, {
                    "path": canonical_path_for_storage(audio_path),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                })
                failed_set.add(key)

    finalize_json_output(OUTPUT_JSON, OUTPUT_JSONL)
    finalize_json_output(FAILED_JSON, FAILED_JSONL)
    finalize_json_output(SWITCH_JSON, SWITCH_JSONL)

if __name__ == "__main__":
    main()
