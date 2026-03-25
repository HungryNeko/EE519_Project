import json
import re
import shutil
from itertools import islice
from pathlib import Path

from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATASET_NAME = "Shelton1013/SwitchLingua_audio"
TOTAL_SAMPLES = 2000

OUT_DIR = Path(__file__).resolve().parent / "switchlingua_audio_2000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Streaming {TOTAL_SAMPLES} samples from {DATASET_NAME}")
print(f"Saving to: {OUT_DIR}")

# 只取 streaming，不要经典 load_dataset 物化整个数据集
ds = load_dataset(DATASET_NAME, streaming=True)

# 这个数据集当前只有 train split
if isinstance(ds, dict) or hasattr(ds, "keys"):
    print("Available splits:", list(ds.keys()))
    stream = ds["train"]
else:
    stream = ds

# 关键：不要自动解码
stream = stream.cast_column("audio", Audio(decode=False))

def parse_remote_audio_path(path_str: str):
    """
    尝试从 HuggingFace 的远程路径里解析：
    1) revision
    2) repo 内相对文件名 filename
    """
    # 例如：datasets/Shelton1013/SwitchLingua_audio@987ee.../Hindi/0_0.wav
    m = re.search(rf"{re.escape(DATASET_NAME)}@([^/]+)/(.+)$", path_str)
    if m:
        return m.group(1), m.group(2)

    # 兜底：匹配任意 @revision/filename 结构
    m = re.search(r"@([^/]+)/(.+)$", path_str)
    if m:
        return m.group(1), m.group(2)

    return None, None

def materialize_audio(audio_dict, out_path: Path):
    """
    把 decode=False 返回的 audio 落盘为本地文件。
    优先 bytes，其次 path。
    """
    b = audio_dict.get("bytes", None)
    if b is not None:
        out_path.write_bytes(b)
        return

    p = audio_dict.get("path", None)
    if p is None:
        raise ValueError("audio dict has neither 'bytes' nor 'path'")

    # 本地路径，直接复制
    local_p = Path(p)
    if local_p.exists():
        shutil.copy2(local_p, out_path)
        return

    # 远程 HuggingFace 路径：尝试从 path 里解析 revision + filename
    revision, filename = parse_remote_audio_path(str(p))
    if filename is None:
        raise ValueError(f"Cannot parse remote audio path: {p}")

    kwargs = dict(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=filename,
    )
    if revision is not None:
        kwargs["revision"] = revision

    downloaded = hf_hub_download(**kwargs)
    shutil.copy2(downloaded, out_path)

metadata = []

for i, sample in enumerate(
    tqdm(islice(stream, TOTAL_SAMPLES), total=TOTAL_SAMPLES, desc="Saving", ncols=80)
):
    audio = sample["audio"]
    wav_path = OUT_DIR / f"{i}.wav"
    materialize_audio(audio, wav_path)

    item = {k: v for k, v in sample.items() if k != "audio"}
    item["wav_path"] = str(wav_path)
    metadata.append(item)

meta_path = OUT_DIR / "metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nDone. Saved {len(metadata)} samples.")
print(f"Output: {OUT_DIR}")