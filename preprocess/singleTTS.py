import asyncio
import edge_tts

async def tts(text, voice, out_wav):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(out_wav)

async def main():
    await tts(
        "My name is Alex. I am taking a speech AI course.",
        "en-US-JennyNeural",
        "speaker_en.wav"
    )

asyncio.run(main())

print("Saved: out.wav")
