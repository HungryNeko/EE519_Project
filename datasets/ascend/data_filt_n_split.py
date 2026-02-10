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


def get_languages(item: dict) -> set:
    """Collect all languages appearing in segments + whisper_language"""
    langs = set()

    # top-level whisper language
    if "whisper_language" in item:
        langs.add(item["whisper_language"])

    # segment-level language spans
    for seg in item.get("segments", []):
        for span in seg.get("language_spans", []):
            lang = span.get("language")
            if lang:
                langs.add(lang)

    return langs


with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

same_language = []
mixed_language = []

for item in data:
    # 1️⃣ normalize path
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    # 2️⃣ collect languages
    langs = get_languages(item)

    # optional: restrict to zh/en only
    langs = {l for l in langs if l in ALLOWED_LANGS}

    # 3️⃣ split
    if len(langs) <= 1:
        same_language.append(item)
    else:
        mixed_language.append(item)

# write outputs
with open(OUT_SAME_LANG, "w", encoding="utf-8") as f:
    json.dump(same_language, f, ensure_ascii=False, indent=2)

with open(OUT_MIXED_LANG, "w", encoding="utf-8") as f:
    json.dump(mixed_language, f, ensure_ascii=False, indent=2)

print(f"Same-language items : {len(same_language)}")
print(f"Mixed-language items: {len(mixed_language)}")
