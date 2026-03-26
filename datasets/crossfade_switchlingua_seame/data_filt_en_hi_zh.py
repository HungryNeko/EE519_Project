import json

INPUT_JSON = "datasets/crossfade_switchlingua_seame/whisper_segment_crossfade_switchlingua_seame.json"

OUT_EN = "datasets/crossfade_switchlingua_seame/crossfade_switchlingua_seame_en_language.json"
OUT_HI = "datasets/crossfade_switchlingua_seame/crossfade_switchlingua_seame_hi_language.json"
OUT_ZH = "datasets/crossfade_switchlingua_seame/crossfade_switchlingua_seame_zh_language.json"
OUT_MIXED = "datasets/crossfade_switchlingua_seame/crossfade_switchlingua_seame_mixed_language.json"
OUT_NON_EN_HI_ZH = "datasets/crossfade_switchlingua_seame/crossfade_switchlingua_seame_non_en_hi_zh_language.json"

BASE_DIR = "datasets/crossfade_switchlingua_seame"
ALLOWED_LANGS = {"zh", "hi", "en"}


def normalize_path(path: str) -> str:
    """Convert absolute path to relative datasets/crossfade_switchlingua_seame/..."""
    path = path.replace("\\", "/")
    path_lower = path.lower()
    base_dir_lower = BASE_DIR.lower()
    if base_dir_lower in path_lower:
        start = path_lower.index(base_dir_lower)
        return path[start:]
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
hi_only = []
en_only = []
mixed_language = []
non_en_hi_zh_language = []

empty_span_items = 0

for item in data:
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    span_langs = get_span_languages(item)

    if not span_langs:
        empty_span_items += 1
        continue

    has_non_en_hi_zh = bool(span_langs - ALLOWED_LANGS)
    has_en = "en" in span_langs
    has_hi = "hi" in span_langs
    has_zh = "zh" in span_langs

    if has_non_en_hi_zh:
        item["whisper_language"] = "non_en_hi_zh"
        non_en_hi_zh_language.append(item)
    elif has_zh and not has_hi and not has_en:
        item["whisper_language"] = "zh"
        zh_only.append(item)
    elif has_hi and not has_zh and not has_en:
        item["whisper_language"] = "hi"
        hi_only.append(item)
    elif has_en and not has_zh and not has_hi:
        item["whisper_language"] = "en"
        en_only.append(item)
    elif len(span_langs) >= 2:
        item["whisper_language"] = "mixed"
        mixed_language.append(item)
    else:
        item["whisper_language"] = "non_en_hi_zh"
        non_en_hi_zh_language.append(item)


with open(OUT_ZH, "w", encoding="utf-8") as f:
    json.dump(zh_only, f, ensure_ascii=False, indent=2)

with open(OUT_HI, "w", encoding="utf-8") as f:
    json.dump(hi_only, f, ensure_ascii=False, indent=2)

with open(OUT_EN, "w", encoding="utf-8") as f:
    json.dump(en_only, f, ensure_ascii=False, indent=2)

with open(OUT_MIXED, "w", encoding="utf-8") as f:
    json.dump(mixed_language, f, ensure_ascii=False, indent=2)

with open(OUT_NON_EN_HI_ZH, "w", encoding="utf-8") as f:
    json.dump(non_en_hi_zh_language, f, ensure_ascii=False, indent=2)


print(f"ZH only       : {len(zh_only)}")
print(f"HI only       : {len(hi_only)}")
print(f"EN only       : {len(en_only)}")
print(f"Mixed allowed : {len(mixed_language)}")
print(f"Non en/hi/zh  : {len(non_en_hi_zh_language)}")
print(f"Empty spans   : {empty_span_items}")
