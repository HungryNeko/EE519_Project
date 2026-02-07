from pathlib import Path
import json
from typing import Any, Dict, List, Set, Tuple

import torch
import whisper
import whisperx
from tqdm import tqdm

# ======================
# Configuration
# ======================
dataset = "ascend"
CORPUS_ROOT = Path(f"./datasets/{dataset}")
OUTPUT_JSON = CORPUS_ROOT / f"whisper_segment_{dataset}.json"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
MODEL_NAME = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    # 统一为绝对 POSIX 小写字符串，保证 Windows/macOS/Linux 可比较
    try:
        return p.resolve().as_posix().lower()
    except Exception:
        # fallback: non-strict resolve
        return p.expanduser().resolve(strict=False).as_posix().lower()

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
    """
    从已有 JSON 读取已处理路径，并把它们 canonical 化为与 canonical_path_for_storage 一致的格式。
    使用 resolve(strict=False) 以避免历史记录中不存在的路径导致异常。
    """
    rows = load_existing_records(path)
    processed = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        raw = r.get("path")
        if not raw:
            continue
        try:
            p = Path(raw)
            canon = p.expanduser().resolve(strict=False).as_posix().lower()
        except Exception:
            canon = str(raw).replace("\\", "/").lower()
        processed.add(canon)
    return processed

# ======================
# Language detection by Unicode (简单)
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
# Alignment cache (safe)
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
    )

    language = result.get("language", "unknown")
    align_model, metadata = get_align_resources(language, align_cache)

    all_words = []
    if align_model is not None:
        try:
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
        except Exception:
            all_words = []

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

    # 返回时确保 path 字段使用 canonical 存储格式
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
    processed_set = read_processed_paths(OUTPUT_JSON)  # 读取并 canonical 化已保存的 path

    audio_files = collect_all_audio_files(CORPUS_ROOT)

    # 先生成每个文件对应的“存储路径”，再与 JSON 中比较
    files_and_storage = [(p, canonical_path_for_storage(p)) for p in audio_files]
    to_process = [p for p, storage in files_and_storage if storage not in processed_set]

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
        except Exception:
            # 单文件异常直接跳过
            continue

if __name__ == "__main__":
    main()
