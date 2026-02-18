import torch
import whisper
import whisperx

AUDIO_FILE = r"samples\output_spk0.wav"
MODEL = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model(MODEL, device=DEVICE)
result = model.transcribe(AUDIO_FILE, verbose=False)

print("\n=== WHISPER GLOBAL INFO ===")
print("language:", result["language"])

print("\n=== WHISPER SEGMENTS ===")
for s in result["segments"]:
    print(s)

align_model, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=DEVICE
)

aligned = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    AUDIO_FILE,
    DEVICE
)

print("\n=== WHISPERX WORDS ===")
for seg in aligned["segments"]:
    for w in seg["words"]:
        print(w)
