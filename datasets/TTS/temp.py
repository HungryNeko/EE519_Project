import json
from pathlib import Path

# ====== 配置 ======
INPUT_JSON = "datasets/TTS/whisper_segment_TTS.json"
OUTPUT_JSON = None  # None 表示覆盖原文件


def fix_path(p: str) -> str:
    p = p.replace("\\", "/")

    # 统一 datasets/tts → datasets/TTS
    if "datasets/tts/" in p.lower():
        idx = p.lower().index("datasets/tts/")
        suffix = p[idx + len("datasets/tts/"):]
        return "datasets/TTS/" + suffix

    return p


def main():
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSON) if OUTPUT_JSON else input_path

    data = json.loads(input_path.read_text(encoding="utf-8"))

    changed = 0

    for item in data:
        old_path = item.get("path", "")
        new_path = fix_path(old_path)

        if old_path != new_path:
            item["path"] = new_path
            changed += 1

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Done. Changed {changed} paths.")


if __name__ == "__main__":
    main()