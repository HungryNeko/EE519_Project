import asyncio
from pathlib import Path

import edge_tts
import torchaudio

# Output folder for all generated wav files.
OUTPUT_DIR = Path("preprocess/generated_batch")

# Fill this list with your Chinese/English content later.
# Each item generates 3 files:
#   <id>_zh.wav, <id>_en.wav, <id>_mixed.wav
SYNTH_CASES = [
    {
        "id": "sample_001",
        "zh_text": "我是琳达，我非常喜欢学习",
        "en_text": "My name is Linda, and I absolutely love learning.",
        "zh_voice": "zh-CN-YunxiNeural",
        "en_voice": "en-US-JennyNeural",
    },
    {
        "id": "sample_002",
        "zh_text": "谢谢你琳达，奖励你一台红色本田",
        "en_text": "Thank you, Linda. Here's a red Honda for you as a reward.",
        "zh_voice": "zh-CN-YunxiNeural",
        "en_voice": "en-US-JennyNeural",
    },
    {
    "id": "sample_003",
    "zh_text": "这样可太好了，周末就可以和杨哥一起出去玩了",
    "en_text": "That's great! I can go out with Brother Yang this weekend.",
    "zh_voice": "zh-CN-YunxiNeural",
    "en_voice": "en-US-JennyNeural",
    },

]


async def tts_to_file(text: str, voice: str, out_wav: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(out_wav))


def mix_two_wavs(wav_a_path: Path, wav_b_path: Path, mixed_out_path: Path) -> None:
    wav_a, sr_a = torchaudio.load(str(wav_a_path))
    wav_b, sr_b = torchaudio.load(str(wav_b_path))

    if sr_a != sr_b:
        raise ValueError(f"Sample rate mismatch: {sr_a} vs {sr_b}")

    min_len = min(wav_a.shape[1], wav_b.shape[1])
    mixed = 0.5 * wav_a[:, :min_len] + 0.5 * wav_b[:, :min_len]
    torchaudio.save(str(mixed_out_path), mixed, sr_a)


async def generate_one_case(case: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    case_id = case["id"]
    zh_path = output_dir / f"{case_id}_zh.wav"
    en_path = output_dir / f"{case_id}_en.wav"
    mixed_path = output_dir / f"{case_id}_mixed.wav"

    await tts_to_file(case["zh_text"], case["zh_voice"], zh_path)
    await tts_to_file(case["en_text"], case["en_voice"], en_path)
    mix_two_wavs(zh_path, en_path, mixed_path)

    return zh_path, en_path, mixed_path


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SYNTH_CASES:
        print("No synthesis cases found. Please add items to SYNTH_CASES.")
        return

    saved_files: list[Path] = []
    for case in SYNTH_CASES:
        zh_path, en_path, mixed_path = await generate_one_case(case, OUTPUT_DIR)
        saved_files.extend([zh_path, en_path, mixed_path])

    print(f"Saved {len(saved_files)} files to: {OUTPUT_DIR.resolve()}")
    for p in saved_files:
        print(f" - {p}")


if __name__ == "__main__":
    asyncio.run(main())
