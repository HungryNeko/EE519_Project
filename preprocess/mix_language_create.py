import asyncio
import edge_tts
import torchaudio
import torch

# =====================
# 1. 用 Edge TTS 生成两种语言
# =====================

async def tts(text, voice, out_wav):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_wav)

async def main():
    # 同一句话，不同语言
    await tts(
        "你好，这是中文。",
        "zh-CN-YunxiNeural",
        "zh.wav"
    )

    await tts(
        "Hello, this is english.",
        "en-US-JennyNeural",
        "en.wav"
    )

asyncio.run(main())

# =====================
# 2. 拼接成一个音频文件
# =====================

wav_zh, sr_zh = torchaudio.load("zh.wav")
wav_en, sr_en = torchaudio.load("en.wav")

assert sr_zh == sr_en, "Sample rate mismatch"

# 转单声道（保险）
if wav_zh.shape[0] > 1:
    wav_zh = wav_zh.mean(dim=0, keepdim=True)
if wav_en.shape[0] > 1:
    wav_en = wav_en.mean(dim=0, keepdim=True)

# 中间加 0.3 秒静音，方便语言分段
silence = torch.zeros(1, int(0.3 * sr_zh))

combined = torch.cat([wav_zh, silence, wav_en], dim=1)

torchaudio.save("two_languages.wav", combined, sr_zh)

print("Saved: two_languages.wav")
