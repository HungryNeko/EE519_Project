

import os
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

OUT_ROOT = "datasets/ascend/audio"

ds = load_dataset(
    "CAiRE/ASCEND",
    cache_dir="datasets/.cache",
)

for split in ds:
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(len(ds[split])), desc=f"Export {split}"):
        audio = ds[split][i]["audio"]   # ← 这里这次一定不会炸
        sf.write(
            os.path.join(out_dir, f"{i:06d}.wav"),
            audio["array"],
            audio["sampling_rate"],
        )
