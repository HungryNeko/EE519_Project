import torch
import whisper
import whisperx
import unicodedata

# ======================
# Config
# ======================
AUDIO_FILE = r"datasets\Corpus\adult\audio\test_split\AD15115.wav"
WHISPER_MODEL = "large-v3"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = "cpu"
# ======================
# Helper: language by script (word-level)
# ======================
def word_language(word: str) -> str:
    for ch in word:
        name = unicodedata.name(ch, "")
        if "DEVANAGARI" in name:
            return "hi"
        if "LATIN" in name:
            return "en"
    return "other"


# ======================
# 1. Transcribe with original Whisper (NO WhisperX ASR)
# ======================
print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

print("Running Whisper transcription...")
result = whisper_model.transcribe(
    AUDIO_FILE,
    verbose=False
)

# result["segments"]: sentence-level segments with timestamps
# result["language"]: detected language (global)


# ======================
# 2. Load WhisperX alignment model ONLY
# ======================
print("Loading WhisperX alignment model...")
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=DEVICE,
)

# ======================
# 3. Force alignment (word-level)
# ======================
print("Running WhisperX alignment...")
aligned = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    AUDIO_FILE,
    DEVICE,
)

# ======================
# 4. Word-level output
# ======================
print("\n=== WORD-LEVEL OUTPUT ===\n")

for seg in aligned["segments"]:
    for w in seg["words"]:
        word = w.get("word", "").strip()
        start = w.get("start", None)
        end = w.get("end", None)

        if not word or start is None or end is None:
            continue

        lang = word_language(word)

        print(f"{word}\t{lang}\t{start:.2f}\t{end:.2f}")
