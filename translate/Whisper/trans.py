import whisper
from langdetect import detect
from collections import defaultdict

# =====================
# 1. 加载官方 Whisper
# =====================

model = whisper.load_model("large-v3")

audio_path = "D:\Github\EE519_Project\preprocess\\two_languages.wav"  # 或 mixed_8k.wav


# =====================
# 2. 转写（自动切段）
# =====================

result = model.transcribe(
    audio_path,
    task="translate",   # 自动识别语言 → 翻译成英文
    verbose=False
)

print("Detected global language:", result["language"])
print()

# =====================
# 3. 按 segment 做语言识别
# =====================

segments_by_language = defaultdict(list)

for seg in result["segments"]:
    text = seg["text"].strip()
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    segments_by_language[lang].append({
        "start": seg["start"],
        "end": seg["end"],
        "text": text
    })

# =====================
# 4. 输出结果
# =====================

for lang, segs in segments_by_language.items():
    print(f"\n=== Language: {lang} ===")
    for s in segs:
        print(f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}")
