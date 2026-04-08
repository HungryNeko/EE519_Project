import json

INPUT_SEGMENT_JSON = "datasets/Hindi-English_train/whisper_segment_Hindi-English_train.json"
INPUT_SWITCH_JSON = "datasets/Hindi-English_train/whisper_language_switch_Hindi-English_train.json"

OUT_EN = "datasets/Hindi-English_train/Hindi-English_train_en_language.json"
OUT_HI = "datasets/Hindi-English_train/Hindi-English_train_hi_language.json"
OUT_MIXED = "datasets/Hindi-English_train/Hindi-English_train_mixed_language.json"
OUT_NON_EN_HI = "datasets/Hindi-English_train/Hindi-English_train_non_en_hi_language.json"

BASE_DIR = "datasets/Hindi-English_train"
ALLOWED_LANGS = {"hi", "en"}


def normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    path_lower = path.lower()
    base_dir_lower = BASE_DIR.lower()
    if base_dir_lower in path_lower:
        start = path_lower.index(base_dir_lower)
        return path[start:]
    return path


def get_span_languages(item: dict) -> set:
    langs = set()
    for seg in item.get("segments", []):
        for span in seg.get("language_spans", []):
            lang = span.get("language")
            if lang:
                langs.add(lang)
    return langs


with open(INPUT_SEGMENT_JSON, "r", encoding="utf-8") as f:
    segment_data = json.load(f)

with open(INPUT_SWITCH_JSON, "r", encoding="utf-8") as f:
    switch_data = json.load(f)

switch_map = {}
for item in switch_data:
    path = item.get("path")
    if not path:
        continue
    switch_map[normalize_path(path)] = {
        "switch_count": item.get("switch_count", 0),
        "switch_positions": item.get("switch_positions", []),
    }

hi_only = []
en_only = []
mixed_language = []
non_en_hi_language = []

missing_switch_info = 0
empty_span_items = 0

for item in segment_data:
    if "path" in item:
        item["path"] = normalize_path(item["path"])

    switch_info = switch_map.get(item.get("path"))
    if switch_info:
        item["switch_count"] = switch_info["switch_count"]
        item["switch_positions"] = switch_info["switch_positions"]
    else:
        item["switch_count"] = item.get("switch_count", 0)
        item["switch_positions"] = item.get("switch_positions", [])
        missing_switch_info += 1

    span_langs = get_span_languages(item)
    if not span_langs:
        empty_span_items += 1
        continue

    has_non_en_hi = bool(span_langs - ALLOWED_LANGS)
    has_en = "en" in span_langs
    has_hi = "hi" in span_langs

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
        item["whisper_language"] = "non_en_hi"
        non_en_hi_language.append(item)

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
print(f"Empty spans : {empty_span_items}")
print(f"Missing switch info: {missing_switch_info}")
