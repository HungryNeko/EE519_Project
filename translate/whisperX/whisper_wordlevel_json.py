from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

import whisper
import whisperx
from tqdm import tqdm

# ======================
# Configuration
# ======================
dataset = "crossfade_insertions"
CORPUS_ROOT = Path(f"./datasets/{dataset}")
OUTPUT_JSON = CORPUS_ROOT / f"whisper_segment_{dataset}.json"
FAILED_JSON = CORPUS_ROOT / f"whisper_failed_{dataset}.json"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
MODEL_NAME = "large-v3"
DEVICE = "cpu"
USE_WHISPERX_ALIGN = False

# ======================
# Helpers
# ======================
def collect_all_audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )

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

# ======================
# 核心：真实位置身份（只认 datasets/corpus 之后）
# ======================
def corpus_relative_identity(p: Path) -> str:
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

def read_processed_paths(path: Path) -> Set[str]:
    rows = load_existing_records(path)
    out = set()
    for r in rows:
        raw = r.get("path")
        if not raw:
            continue
        try:
            out.add(corpus_relative_identity(Path(raw)))
        except Exception:
            continue
    return out

def read_failed_paths(path: Path) -> Set[str]:
    rows = load_existing_records(path)
    out = set()
    for r in rows:
        raw = r.get("path")
        if not raw:
            continue
        try:
            out.add(corpus_relative_identity(Path(raw)))
        except Exception:
            continue
    return out

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
# Alignment cache
# ======================
def get_align_resources(language_code: str, cache):
    if language_code in cache:
        return cache[language_code]
    try:
        cache[language_code] = whisperx.load_align_model(
            language_code=language_code,
            device=DEVICE,
        )
        return cache[language_code]
    except Exception:
        cache[language_code] = (None, None)
        return (None, None)

# ======================
# Core processing
# ======================
def process_single_audio(whisper_model, align_cache, audio_path: Path):
    storage_path = canonical_path_for_storage(audio_path)

    result = whisper_model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,
        verbose=False,
        fp16=False,
        word_timestamps=not USE_WHISPERX_ALIGN,
    )

    language = result.get("language", "unknown")
    all_words = []

    if USE_WHISPERX_ALIGN:
        align_model, metadata = get_align_resources(language, align_cache)
    else:
        align_model, metadata = (None, None)

    if align_model is not None:
        aligned = whisperx.align(
            result.get("segments", []),
            align_model,
            metadata,
            str(audio_path),
            DEVICE,
        )
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                if w.get("start") is not None and w.get("end") is not None:
                    all_words.append({
                        "word": w.get("word", "").strip(),
                        "start": w["start"],
                        "end": w["end"],
                        "score": w.get("score", 0.0),
                    })
    else:
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if w.get("start") is not None and w.get("end") is not None:
                    all_words.append({
                        "word": w.get("word", "").strip(),
                        "start": w["start"],
                        "end": w["end"],
                        "score": w.get("probability", 0.0),
                    })

    def build_language_spans(words, seg_start, seg_end):
        spans = []
        cur = None

        def flush():
            nonlocal cur
            if cur:
                cur["text"] = cur["text"].strip()
                cur["score"] = cur["score_sum"] / max(cur["count"], 1)
                cur.pop("score_sum")
                cur.pop("count")
                spans.append(cur)
                cur = None

        for w in words:
            if w["end"] < seg_start or w["start"] > seg_end:
                continue
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
        return spans

    segments_out = []
    for i, seg in enumerate(result.get("segments", [])):
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        seg_text = seg.get("text", "").strip()

        language_spans = build_language_spans(all_words, seg_start, seg_end) if all_words else []
        if not language_spans:
            language_spans = [{
                "language": language,
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "score": seg.get("avg_logprob"),
            }]

        segments_out.append({
            "segment_id": i,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "scores": {
                "avg_logprob": seg.get("avg_logprob"),
                "compression_ratio": seg.get("compression_ratio"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "temperature": seg.get("temperature", result.get("temperature")),
            },
            "language_spans": language_spans,
        })

    return to_jsonable({
        "path": storage_path,
        "whisper_language": language,
        "segments": segments_out,
    })

# ======================
# Main
# ======================
def main():
    existing = load_existing_records(OUTPUT_JSON)
    failed_records = load_existing_records(FAILED_JSON)

    processed_set = read_processed_paths(OUTPUT_JSON)
    failed_set = read_failed_paths(FAILED_JSON)

    audio_files = collect_all_audio_files(CORPUS_ROOT)

    files_and_keys = [
        (p, corpus_relative_identity(p))
        for p in audio_files
    ]

    skip_set = processed_set | failed_set
    to_process = [p for p, key in files_and_keys if key not in skip_set]

    whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)
    align_cache = {}

    for audio_path in tqdm(to_process, desc="Processing", unit="file"):
        try:
            record = process_single_audio(
                whisper_model,
                align_cache,
                audio_path,
            )
            existing.append(record)
            save_records(OUTPUT_JSON, existing)
        except Exception as e:
            key = corpus_relative_identity(audio_path)
            if key not in failed_set:
                failed_records.append({
                    "path": canonical_path_for_storage(audio_path),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                })
                failed_set.add(key)
                save_records(FAILED_JSON, failed_records)

if __name__ == "__main__":
    main()
