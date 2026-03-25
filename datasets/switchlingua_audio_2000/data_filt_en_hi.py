import json
from pathlib import Path

INPUT_JSON = "datasets/switchlingua_audio_2000/whisper_segment_switchlingua_audio_2000.json"
OUT_EN_HI = "datasets/switchlingua_audio_2000/whisper_segment_switchlingua_audio_2000_en_hi_only.json"

BASE_DIR = "datasets/switchlingua_audio_2000"
ALLOWED_LANGS = {"en", "hi"}


def normalize_path(path: str) -> str:
    """Convert absolute path to relative datasets/switchlingua_audio_2000/..."""
    path = path.replace("\\", "/")
    path_lower = path.lower()
    base_lower = BASE_DIR.lower()
    if base_lower in path_lower:
        start = path_lower.index(base_lower)
        return path[start:]
    return path


def get_span_languages(item: dict) -> set[str]:
    """Collect all languages from language_spans."""
    langs = set()
    for seg in item.get("segments", []):
        for span in seg.get("language_spans", []):
            lang = span.get("language")
            if lang:
                langs.add(lang)
    return langs


def classify_languages(span_langs: set[str]) -> str:
    has_non_en_hi = bool(span_langs - ALLOWED_LANGS)
    has_en = "en" in span_langs
    has_hi = "hi" in span_langs

    if has_non_en_hi:
        return "non_en_hi"
    if has_hi and not has_en:
        return "hi"
    if has_en and not has_hi:
        return "en"
    if has_en and has_hi:
        return "mixed"
    return "non_en_hi"


def main() -> None:
    input_path = Path(INPUT_JSON)
    output_path = Path(OUT_EN_HI)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    skipped = 0

    for item in data:
        normalized = dict(item)
        if "path" in normalized:
            normalized["path"] = normalize_path(normalized["path"])

        span_langs = get_span_languages(normalized)
        if not span_langs:
            skipped += 1
            continue

        whisper_language = classify_languages(span_langs)
        if whisper_language == "non_en_hi":
            skipped += 1
            continue

        normalized["audio_name"] = Path(normalized["path"]).name
        normalized["whisper_language"] = whisper_language
        cleaned.append(normalized)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"input_records: {len(data)}")
    print(f"output_records: {len(cleaned)}")
    print(f"skipped_records: {skipped}")
    print(f"output_path: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
