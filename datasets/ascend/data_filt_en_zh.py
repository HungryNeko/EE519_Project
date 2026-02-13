import json

INPUT_JSON = "datasets/ascend/whisper_segment_ascend.json"

OUT_EN = "datasets/ascend/ascend_en_language.json"
OUT_ZH = "datasets/ascend/ascend_zh_language.json"
OUT_MIXED = "datasets/ascend/ascend_mixed_language.json"

BASE_DIR = "datasets/ascend"
ALLOWED_LANGS = {"zh", "en"}


def normalize_path(path: str) -> str:
    """Convert absolute path to relative datasets/ascend/..."""
    path = path.replace("\\", "/")
    if BASE_DIR in path:
        return path[path.index(BASE_DIR):]
    return path


def get_span_languages(item: dict) -> set:
    """Collect zh/en languages from language_spans only"""
    langs = set()
    for seg in item.get("segments", []):
        for span in seg.get("language_spans", []):
            lang = span.get("language")
            if lang in ALLOWED_LANGS:
                langs.add(lang)
    return langs


with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

zh_only = []
en_only = []
mixed_language = []

for item in data:
    # 1️⃣ Normalize path
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    # 2️⃣ Determine language strictly from spans
    span_langs = get_span_languages(item)

    # Skip if no valid zh/en spans
    if not span_langs:
        continue

    # 3️⃣ Split
    if len(span_langs) == 1:
        lang = list(span_langs)[0]
        item["whisper_language"] = lang

        if lang == "zh":
            zh_only.append(item)
        elif lang == "en":
            en_only.append(item)

    else:
        item["whisper_language"] = "mixed"
        mixed_language.append(item)


# Save outputs
with open(OUT_ZH, "w", encoding="utf-8") as f:
    json.dump(zh_only, f, ensure_ascii=False, indent=2)

with open(OUT_EN, "w", encoding="utf-8") as f:
    json.dump(en_only, f, ensure_ascii=False, indent=2)

with open(OUT_MIXED, "w", encoding="utf-8") as f:
    json.dump(mixed_language, f, ensure_ascii=False, indent=2)


print(f"ZH only    : {len(zh_only)}")
print(f"EN only    : {len(en_only)}")
print(f"Mixed lang : {len(mixed_language)}")
