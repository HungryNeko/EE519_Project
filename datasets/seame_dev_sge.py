import json
import re
import shutil
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATASET_NAME = "AudioLLMs/seame_dev_sge"
SPLIT = "test"
TARGET_SAMPLES = 2000

OUT_DIR = Path(__file__).resolve().parent / "seame_dev_sge_cn_en_2000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {DATASET_NAME}")
print(f"Target samples: {TARGET_SAMPLES}")
print(f"Saving to: {OUT_DIR}")

# 中文 + 英文检测
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_RE = re.compile(r"[A-Za-z]")

def has_zh_en(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(CHINESE_RE.search(text)) and bool(ENGLISH_RE.search(text))

def parse_remote_path(path_str: str):
    """
    尝试从 HuggingFace 远程路径中解析 revision 和 repo 内文件名。
    """
    m = re.search(rf"{re.escape(DATASET_NAME)}@([^/]+)/(.+)$", path_str)
    if m:
        return m.group(1), m.group(2)

    m = re.search(r"@([^/]+)/(.+)$", path_str)
    if m:
        return m.group(1), m.group(2)

    return None, None

def save_audio_from_value(value, out_path: Path):
    """
    兼容几种常见音频存法：
    1) dict: {'array', 'sampling_rate'}
    2) dict: {'bytes'} / {'path'}
    3) str path
    4) bytes
    """
    if isinstance(value, dict):
        if "array" in value and value["array"] is not None:
            sf.write(str(out_path), value["array"], value["sampling_rate"])
            return

        if "bytes" in value and value["bytes"] is not None:
            out_path.write_bytes(value["bytes"])
            return

        if "path" in value and value["path"]:
            p = Path(str(value["path"]))
            if p.exists():
                shutil.copy2(p, out_path)
                return

            revision, filename = parse_remote_path(str(value["path"]))
            if filename is not None:
                downloaded = hf_hub_download(
                    repo_id=DATASET_NAME,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                )
                shutil.copy2(downloaded, out_path)
                return

    if isinstance(value, (bytes, bytearray)):
        out_path.write_bytes(value)
        return

    if isinstance(value, str) and value:
        p = Path(value)
        if p.exists():
            shutil.copy2(p, out_path)
            return

        revision, filename = parse_remote_path(value)
        if filename is not None:
            downloaded = hf_hub_download(
                repo_id=DATASET_NAME,
                repo_type="dataset",
                filename=filename,
                revision=revision,
            )
            shutil.copy2(downloaded, out_path)
            return

    raise KeyError(f"Cannot save audio from value type={type(value)}")

print("Loading dataset...")
ds = load_dataset(DATASET_NAME, split=SPLIT)

print(f"Loaded {len(ds)} total rows")
print("Sample keys:", list(ds[0].keys()))

metadata = []
kept = 0

for idx, sample in enumerate(tqdm(ds, desc="Filtering+Saving", ncols=80)):
    answer = sample.get("answer", "")
    if not has_zh_en(answer):
        continue

    audio_value = sample.get("context", None)
    if audio_value is None:
        continue

    wav_path = OUT_DIR / f"{kept:05d}.wav"

    try:
        save_audio_from_value(audio_value, wav_path)
    except Exception as e:
        # 音频保存失败就跳过，不中断
        print(f"\nSkip row {idx} because audio save failed: {e}")
        continue

    item = {}
    for k, v in sample.items():
        if k != "context":
            item[k] = v
    item["wav_path"] = str(wav_path)
    item["source_row_idx"] = idx
    metadata.append(item)

    kept += 1
    if kept % 50 == 0:
        print(f"Kept {kept}/{TARGET_SAMPLES}")

    if kept >= TARGET_SAMPLES:
        break

meta_path = OUT_DIR / "metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nDone.")
print(f"Saved {len(metadata)} samples")
print(f"Output dir: {OUT_DIR}")