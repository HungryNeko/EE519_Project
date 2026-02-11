import json

INPUT_JSON = "datasets/ascend/whisper_segment_ascend.json"

OUT_SAME_LANG = "datasets/ascend/same_language.json"
OUT_MIXED_LANG = "datasets/ascend/mixed_language.json"

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

same_language = []
mixed_language = []

for item in data:
    # 1️⃣ Normalize path
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    # 2️⃣ Determine language strictly from spans
    span_langs = get_span_languages(item)

    # Skip if no valid zh/en spans found
    if not span_langs:
        continue

    # 3️⃣ Fix whisper_language
    if len(span_langs) == 1:
        item["whisper_language"] = list(span_langs)[0]
        same_language.append(item)
    else:
        item["whisper_language"] = "mixed"
        mixed_language.append(item)


# Save outputs
with open(OUT_SAME_LANG, "w", encoding="utf-8") as f:
    json.dump(same_language, f, ensure_ascii=False, indent=2)

with open(OUT_MIXED_LANG, "w", encoding="utf-8") as f:
    json.dump(mixed_language, f, ensure_ascii=False, indent=2)

print(f"Same-language items : {len(same_language)}")
print(f"Mixed-language items: {len(mixed_language)}")
