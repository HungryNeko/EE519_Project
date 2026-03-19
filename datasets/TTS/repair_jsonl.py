import json
from pathlib import Path

INPUT = "dl_model/mlp_feature_cache.jsonl"
OUTPUT = None  # None = 覆盖原文件


def normalize_path(p: str) -> str:
    # 统一路径格式 + 小写
    return p.replace("\\", "/").strip().lower()


def main():
    input_path = Path(INPUT)
    output_path = Path(OUTPUT) if OUTPUT else input_path.with_suffix(".tmp")

    total = 0
    modified = 0
    bad = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            total += 1

            try:
                data = json.loads(line)
            except Exception:
                bad += 1
                continue

            audio_path = data.get("audio_path", "")
            norm_path = normalize_path(audio_path)

            # 🔴 核心修改逻辑
            if "datasets/tts/mix/" in norm_path:
                if data.get("is_switch") != False:
                    data["is_switch"] = False
                    modified += 1

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    # 覆盖原文件（安全替换）
    if OUTPUT is None:
        input_path.unlink()
        output_path.rename(input_path)

    print(f"done.")
    print(f"total   = {total}")
    print(f"modified= {modified}")
    print(f"bad     = {bad}")
    print(f"kept    = {total - bad}")


if __name__ == "__main__":
    main()