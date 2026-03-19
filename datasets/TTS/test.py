import asyncio
import edge_tts
import os
import json

OUT_DIR = "datasets/voice_test_all"
os.makedirs(OUT_DIR, exist_ok=True)

VOICE_MAP = {
    "0": "zh-CN-XiaoxiaoNeural",
    "1": "zh-CN-YunxiNeural",
    "2": "zh-CN-YunjianNeural",
    "3": "zh-CN-YunyangNeural",
    "4": "en-US-JennyNeural",
    "5": "en-US-AriaNeural",
    "6": "en-US-GuyNeural",
    "7": "en-US-EricNeural",
    "8": "en-GB-SoniaNeural",
    "9": "en-GB-RyanNeural",
    "10": "en-AU-NatashaNeural",
    "11": "en-AU-WilliamNeural",
    "12": "en-CA-ClaraNeural",
    "13": "en-CA-LiamNeural",
    "14": "en-IN-NeerjaNeural",
    "15": "en-IN-PrabhatNeural",
}

TEXT = "今天我们先 run the experiment，然后再分析结果。"

async def generate(idx, voice):
    out_path = os.path.join(OUT_DIR, f"{idx}.mp3")

    communicate = edge_tts.Communicate(
        text=TEXT,
        voice=voice
    )

    await communicate.save(out_path)
    print(f"{idx} -> {voice}")

async def main():
    # 保存映射（方便你对照听）
    with open(os.path.join(OUT_DIR, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(VOICE_MAP, f, ensure_ascii=False, indent=2)

    tasks = []
    for idx, voice in VOICE_MAP.items():
        tasks.append(generate(idx, voice))

    await asyncio.gather(*tasks)

    print("Done. Check:", OUT_DIR)

if __name__ == "__main__":
    asyncio.run(main())