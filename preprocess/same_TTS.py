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

    # 用 BytesIO 直接读成 tensor
    buf = io.BytesIO(audio_bytes)
    wav, sr = torchaudio.load(buf)

    # 转单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav, sr


# =====================
# 主逻辑：同一音色，两种语言，只存一个文件
# =====================

async def main():
    voice = "zh-CN-YunxiNeural"  # 同一个音色

    wav_zh, sr_zh = await tts_to_tensor(
        "你好，这是中文。",
        voice
    )

    wav_en, sr_en = await tts_to_tensor(
        "Hello, this is English.",
        voice
    )

    assert sr_zh == sr_en, "Sample rate mismatch"



    combined = torch.cat([wav_zh, wav_en], dim=1)

    # 输出路径：preprocess 目录
    out_path = os.path.join("preprocess", "same_voice_two_languages.wav")
    os.makedirs("preprocess", exist_ok=True)

    torchaudio.save(out_path, combined, sr_zh)
    print("Saved:", out_path)


if __name__ == "__main__":
    asyncio.run(main())
