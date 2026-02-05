import whisper
from langdetect import detect
from collections import defaultdict

model = whisper.load_model("large-v3")
audio_path = "preprocess/two_languages.wav"

result = model.transcribe(
    audio_path,
    task="transcribe",
    verbose=False
)

segments_by_language = defaultdict(list)

for seg in result["segments"]:
    text = seg["text"].strip()

    # 1️⃣ 优先用 Whisper 的 segment 语言
    lang = seg.get("language")

    # 2️⃣ 如果 Whisper 没给，就用 langdetect 兜底
    if not lang:
        try:
            lang = detect(text)
        except:
            lang = "unknown"

    segments_by_language[lang].append({
        "lang": lang,
        "start": seg["start"],
        "end": seg["end"],
        "text": text
    })

# 输出
for lang, segs in segments_by_language.items():
    print(f"\n=== Language: {lang} ===")
    for s in segs:
        print(f"[{s['lang']}] {s['start']:.2f}-{s['end']:.2f}  {s['text']}")
