import asyncio
import edge_tts
import torchaudio
import torch
import io
import os

# =====================
# 工具：TTS 到内存（不落文件）
# =====================

async def tts_to_tensor(text, voice):
    communicate = edge_tts.Communicate(text=text, voice=voice)

    audio_bytes = bytearray()
    async for msg in communicate.stream():
        if msg.get("type") == "audio":
            audio_bytes.extend(msg["data"])

    buf = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(buf)

    # 转单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav, sr


# =====================
# 主逻辑：同一语言，不同文字，不同音色
# =====================

async def main():
    # 不同音色（同一语言：中文）
    voice_1 = "zh-CN-YunxiNeural"
    voice_2 = "zh-CN-XiaoxiaoNeural"

    wav_1, sr_1 = await tts_to_tensor(
        "你好，这是第一段中文语音。",
        voice_1
    )

    wav_2, sr_2 = await tts_to_tensor(
        "这是一段不同音色的中文语音。",
        voice_2
    )

    assert sr_1 == sr_2, "Sample rate mismatch"

    combined = torch.cat([wav_1, wav_2], dim=1)

    # 输出路径：preprocess 目录
    out_path = os.path.join("preprocess", "same_language_diff_voice.wav")
    os.makedirs("preprocess", exist_ok=True)

    torchaudio.save(out_path, combined, sr_1)
    print("Saved:", out_path)


if __name__ == "__main__":
    asyncio.run(main())
