import asyncio
import edge_tts
import torchaudio

# =====================
# 1. TTS：男女声
# =====================

async def tts(text, voice, out_wav):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_wav)

async def main():
    # 中文男声
    await tts(
        "你好，欢迎来到语音处理课程。",
        "zh-CN-YunxiNeural",
        "speaker_zh.wav"
    )

    # 英文女声
    await tts(
        "Hello, this is a multilingual speech processing demo.",
        "en-US-JennyNeural",
        "speaker_en.wav"
    )

asyncio.run(main())

# =====================
# 2. 混音（同时说话）
# =====================

wav1, sr1 = torchaudio.load("speaker_zh.wav")
wav2, sr2 = torchaudio.load("speaker_en.wav")
assert sr1 == sr2

min_len = min(wav1.shape[1], wav2.shape[1])
mixed = 0.5 * wav1[:, :min_len] + 0.5 * wav2[:, :min_len]

torchaudio.save("mixed.wav", mixed, sr1)

print("Saved:")
print(" - speaker_zh.wav (male)")
print(" - speaker_en.wav (female)")
print(" - mixed.wav")
