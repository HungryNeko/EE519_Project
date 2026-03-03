import json

INPUT_JSON = "datasets/ascend/whisper_segment_ascend.json"

OUT_EN = "datasets/ascend/ascend_en_language.json"
OUT_ZH = "datasets/ascend/ascend_zh_language.json"
OUT_MIXED = "datasets/ascend/ascend_mixed_language.json"
OUT_NON_EN_ZH = "datasets/ascend/ascend_non_en_zh_language.json"

BASE_DIR = "datasets/ascend"
ALLOWED_LANGS = {"zh", "en"}


def normalize_path(path: str) -> str:
    """Convert absolute path to relative datasets/ascend/..."""
    path = path.replace("\\", "/")
    if BASE_DIR in path:
        return path[path.index(BASE_DIR):]
    return path


def get_span_languages(item: dict) -> set:
    """Collect all languages from language_spans."""
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
    # 1) Normalize path
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    # 2) Determine languages from all spans
    span_langs = get_span_languages(item)

    # Skip if no spans
    if not span_langs:
        continue

    has_non_en_zh = bool(span_langs - ALLOWED_LANGS)
    has_en = "en" in span_langs
    has_zh = "zh" in span_langs

    # 3) Split into 4 mutually exclusive classes
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
        # Defensive fallback for unexpected labels
        item["whisper_language"] = "non_en_zh"
        non_en_zh_language.append(item)


# Save outputs
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
