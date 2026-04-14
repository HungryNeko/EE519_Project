import json
from pathlib import Path

INPUT_JSON = "datasets/TALCS_corpus/whisper_segment_TALCS_corpus.json"

OUT_EN = "datasets/TALCS_corpus/TALCS_corpus_en_language.json"
OUT_ZH = "datasets/TALCS_corpus/TALCS_corpus_zh_language.json"
OUT_MIXED = "datasets/TALCS_corpus/TALCS_corpus_mixed_language.json"
OUT_NON_EN_ZH = "datasets/TALCS_corpus/TALCS_corpus_non_en_zh_language.json"

BASE_DIR = "datasets/TALCS_corpus"
ALLOWED_LANGS = {"zh", "en"}


def normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    path_lower = path.lower()
    base_dir_lower = BASE_DIR.lower()
    if base_dir_lower in path_lower:
        start = path_lower.index(base_dir_lower)
        suffix = path[start + len(BASE_DIR):]
        return f"{BASE_DIR}{suffix}"
    return path


def get_span_languages(item: dict) -> set:
    langs = set()
    for seg in item.get("segments", []):
        for span in seg.get("language_spans", []):
            lang = span.get("language")
            if lang:
                langs.add(lang)
    return langs


with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

zh_only = []
en_only = []
mixed_language = []
non_en_zh_language = []

for item in data:
    if "path" in item:
        item["path"] = normalize_path(item["path"])
        item["audio_name"] = Path(item["path"]).name

    span_langs = get_span_languages(item)
    if not span_langs:
        continue

    has_non_en_zh = bool(span_langs - ALLOWED_LANGS)
    has_en = "en" in span_langs
    has_zh = "zh" in span_langs

    if has_non_en_zh:
        item["whisper_language"] = "non_en_zh"
        non_en_zh_language.append(item)
    elif has_zh and not has_en:
        item["whisper_language"] = "zh"
        zh_only.append(item)
    elif has_en and not has_zh:
        item["whisper_language"] = "en"
        en_only.append(item)
    elif has_en and has_zh:
        item["whisper_language"] = "mixed"
        mixed_language.append(item)
    else:
        item["whisper_language"] = "non_en_zh"
        non_en_zh_language.append(item)


with open(OUT_ZH, "w", encoding="utf-8") as f:
    json.dump(zh_only, f, ensure_ascii=False, indent=2)

with open(OUT_EN, "w", encoding="utf-8") as f:
    json.dump(en_only, f, ensure_ascii=False, indent=2)

with open(OUT_MIXED, "w", encoding="utf-8") as f:
    json.dump(mixed_language, f, ensure_ascii=False, indent=2)

with open(OUT_NON_EN_ZH, "w", encoding="utf-8") as f:
    json.dump(non_en_zh_language, f, ensure_ascii=False, indent=2)


print(f"ZH only    : {len(zh_only)}")
print(f"EN only    : {len(en_only)}")
print(f"Mixed en+zh: {len(mixed_language)}")
print(f"Non en/zh  : {len(non_en_zh_language)}")
