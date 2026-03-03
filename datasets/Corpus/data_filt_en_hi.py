import json

INPUT_JSON = "datasets/Corpus/whisper_segment_Corpus.json"

OUT_EN = "datasets/Corpus/corpus_en_language.json"
OUT_HI = "datasets/Corpus/corpus_hi_language.json"
OUT_MIXED = "datasets/Corpus/corpus_mixed_language.json"
OUT_NON_EN_HI = "datasets/Corpus/corpus_non_en_hi_language.json"

BASE_DIR = "datasets/Corpus"
ALLOWED_LANGS = {"hi", "en"}


def normalize_path(path: str) -> str:
    """Convert absolute path to relative datasets/Corpus/..."""
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

hi_only = []
en_only = []
mixed_language = []
non_en_hi_language = []

for item in data:
    # 1) Normalize path
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    # 2) Determine languages from all spans
    span_langs = get_span_languages(item)

    # Skip if no spans
    if not span_langs:
        continue

    has_non_en_hi = bool(span_langs - ALLOWED_LANGS)
    has_en = "en" in span_langs
    has_hi = "hi" in span_langs

    # 3) Split into 4 mutually exclusive classes
    if has_non_en_hi:
        item["whisper_language"] = "non_en_hi"
        non_en_hi_language.append(item)
    elif has_hi and not has_en:
        item["whisper_language"] = "hi"
        hi_only.append(item)
    elif has_en and not has_hi:
        item["whisper_language"] = "en"
        en_only.append(item)
    elif has_en and has_hi:
        item["whisper_language"] = "mixed"
        mixed_language.append(item)
    else:
        # Defensive fallback for unexpected labels
        item["whisper_language"] = "non_en_hi"
        non_en_hi_language.append(item)


# Save outputs
with open(OUT_HI, "w", encoding="utf-8") as f:
    json.dump(hi_only, f, ensure_ascii=False, indent=2)

with open(OUT_EN, "w", encoding="utf-8") as f:
    json.dump(en_only, f, ensure_ascii=False, indent=2)

with open(OUT_MIXED, "w", encoding="utf-8") as f:
    json.dump(mixed_language, f, ensure_ascii=False, indent=2)

with open(OUT_NON_EN_HI, "w", encoding="utf-8") as f:
    json.dump(non_en_hi_language, f, ensure_ascii=False, indent=2)


print(f"HI only    : {len(hi_only)}")
print(f"EN only    : {len(en_only)}")
print(f"Mixed en+hi: {len(mixed_language)}")
print(f"Non en/hi  : {len(non_en_hi_language)}")
