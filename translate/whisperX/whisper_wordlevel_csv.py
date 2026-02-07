from pathlib import Path
import json
from typing import Any, Dict, List, Set, Tuple

import torch
import whisper
import whisperx
from tqdm import tqdm


# ======================
# Fixed configuration
# ======================
CORPUS_ROOT = Path(r".\datasets\Corpus")
OUTPUT_JSON = CORPUS_ROOT / "whisper_word_corpus.json"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
MODEL_NAME = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# Utils
# ======================
def collect_all_audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def to_jsonable(value: Any):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    # Handle numpy scalar types without importing numpy directly.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)

    return value


def load_existing_records(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        return []

    text = json_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {json_path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {json_path}.")

    records: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "path" in item:
            records.append(item)
    return records


def save_records(json_path: Path, records: List[Dict[str, Any]]):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def read_processed_files(json_path: Path) -> Set[str]:
    records = load_existing_records(json_path)
    if not records:
        return set()

    return {
        str(row["path"])
        for row in records
        if row.get("path")
    }


def get_align_resources(
    language_code: str,
    align_cache: Dict[str, Tuple[Any, Any]],
):
    if language_code not in align_cache:
        align_cache[language_code] = whisperx.load_align_model(
            language_code=language_code,
            device=DEVICE,
        )
    return align_cache[language_code]


# ======================
# Core processing
# ======================
def process_single_audio(
    whisper_model,
    align_cache: Dict[str, Tuple[Any, Any]],
    audio_path: Path,
) -> Dict[str, Any]:
    result = whisper_model.transcribe(
        str(audio_path),
        verbose=False,
    )
    language = result.get("language", "unknown")
    align_model, metadata = get_align_resources(language, align_cache)

    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        str(audio_path),
        DEVICE,
    )

    words: List[Dict[str, Any]] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            start = w.get("start")
            end = w.get("end")

            if not word or start is None or end is None:
                continue

            words.append({
                "word": word,
                "start": start,
                "end": end,
                "score": w.get("score"),
            })

    return to_jsonable({
        "path": str(audio_path),
        "whisper_language": language,
        "whisper_result": result,
        "whisperx_aligned": aligned,
        "words": words,
    })


# ======================
# Main
# ======================
def main():
    if not CORPUS_ROOT.exists():
        raise SystemExit(f"Corpus root does not exist: {CORPUS_ROOT}")

    existing_records = load_existing_records(OUTPUT_JSON)

    audio_files = collect_all_audio_files(CORPUS_ROOT)
    if not audio_files:
        raise SystemExit("No audio files found.")

    processed_files = read_processed_files(OUTPUT_JSON)
    to_process = [p for p in audio_files if str(p) not in processed_files]

    if not to_process:
        print("All audio files have already been processed.")
        return

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"New audio files: {len(to_process)}")

    # Load models ONCE
    print("Loading Whisper model...")
    whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)

    print("Preparing WhisperX alignment model cache...")
    align_cache: Dict[str, Tuple[Any, Any]] = {}

    for audio_path in tqdm(to_process, desc="Processing audio", unit="file"):
        record = process_single_audio(
            whisper_model,
            align_cache,
            audio_path,
        )
        existing_records.append(record)
        save_records(OUTPUT_JSON, existing_records)

    print(f"Done. Results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
