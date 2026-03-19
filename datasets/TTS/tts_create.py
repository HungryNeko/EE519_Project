import asyncio
import edge_tts
import torchaudio
import torch
import io
import os
import json
import random
import re
from tqdm import tqdm

# =====================
# 配置
# =====================

INPUT_JSON = r"datasets\TTS\tts_dataset_final.json"
BASE_OUT_DIR = os.path.join("datasets", "tts")

MAX_CONCURRENCY = 8
MAX_RETRIES = 2
TARGET_SAMPLE_RATE = 24000

manifest = []

# =====================
# 工具
# =====================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def sanitize_filename(name):
    name = os.path.basename(name.strip())
    name = re.sub(r"[^\w\-.]+", "_", name)
    if not name.endswith(".wav"):
        name += ".wav"
    return name

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================
# 音频处理（核心）
# =====================

def trim_silence(wav, threshold=0.01):
    energy = torch.abs(wav)
    idx = (energy > threshold).nonzero()

    if len(idx) == 0:
        return wav

    start = idx[0,1]
    end = idx[-1,1]
    return wav[:, start:end+1]

def crossfade(a, b, sr, overlap_sec=0.03):
    overlap = int(sr * overlap_sec)

    if a.shape[1] < overlap or b.shape[1] < overlap:
        return torch.cat([a, b], dim=1)

    a_end = a[:, -overlap:]
    b_start = b[:, :overlap]

    fade_out = torch.linspace(1, 0, overlap)
    fade_in = torch.linspace(0, 1, overlap)

    cross = a_end * fade_out + b_start * fade_in

    return torch.cat([
        a[:, :-overlap],
        cross,
        b[:, overlap:]
    ], dim=1)

def energy_jump(a, b):
    e1 = torch.mean(torch.abs(a[:, -1000:]))
    e2 = torch.mean(torch.abs(b[:, :1000]))
    return abs(e1 - e2)

# =====================
# TTS
# =====================

async def tts_to_tensor(text, voice):
    for attempt in range(MAX_RETRIES):
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=f"{random.randint(-10,10)}%",
                pitch=f"{random.randint(-3,3)}Hz",
            )

            audio_bytes = bytearray()
            async for msg in communicate.stream():
                if msg["type"] == "audio":
                    audio_bytes.extend(msg["data"])

            wav, sr = torchaudio.load(io.BytesIO(audio_bytes))

            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

            if sr != TARGET_SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(wav)
                sr = TARGET_SAMPLE_RATE

            return wav, sr

        except Exception:
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError("TTS failed")

# =====================
# 核心处理
# =====================

async def process_item(item, idx, sem, pbar):
    async with sem:
        try:
            item_type = item["type"]
            output = sanitize_filename(item["output"])
            segments = item["segments"]

            wavs = []
            sr = None

            for seg in segments:
                wav, sr = await tts_to_tensor(seg["text"], seg["voice"])
                wav = trim_silence(wav)

                # 能量检测（过滤坏语音）
                if torch.mean(torch.abs(wav)) < 1e-4:
                    raise RuntimeError("bad audio")

                wavs.append(wav)

            # =====================
            # 无缝拼接（核心）
            # =====================

            combined = wavs[0]
            switch_points = []

            current_time = wavs[0].shape[1] / sr

            for i in range(1, len(wavs)):
                if energy_jump(combined, wavs[i]) > 0.2:
                    raise RuntimeError("energy mismatch")

                combined = crossfade(combined, wavs[i], sr)

                switch_points.append(current_time)
                current_time += wavs[i].shape[1] / sr

            # =====================
            # 保存音频
            # =====================

            out_dir = os.path.join(BASE_OUT_DIR, item_type)
            ensure_dir(out_dir)

            out_path = os.path.join(out_dir, output)
            torchaudio.save(out_path, combined, sr)

            # =====================
            # 生成标注（只保留有切换的）
            # =====================

            if len(switch_points) == 0:
                pbar.update(1)
                return

            segments_out = []
            for i, t in enumerate(switch_points):
                segments_out.append({
                    "segment_id": i,
                    "start": t - 0.05,
                    "end": t + 0.05,
                    "switch_type": "speaker"
                })

            manifest.append({
                "path": out_path,
                "audio_name": output,
                "segments": segments_out
            })

            pbar.update(1)

        except Exception as e:
            print(f"[SKIP] {idx}: {e}")
            pbar.update(1)

# =====================
# 主函数
# =====================

async def main():
    data = load_json(INPUT_JSON)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    pbar = tqdm(total=len(data))

    tasks = [
        process_item(item, i, sem, pbar)
        for i, item in enumerate(data)
    ]

    await asyncio.gather(*tasks)

    out_manifest = os.path.join(BASE_OUT_DIR, "manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Saved:", out_manifest)

if __name__ == "__main__":
    asyncio.run(main())