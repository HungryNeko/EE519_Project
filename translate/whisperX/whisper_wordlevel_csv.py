from pathlib import Path
import csv
from typing import List, Set
import unicodedata

import torch
import whisper
import whisperx
from tqdm import tqdm


# ======================
# Fixed configuration
# ======================
CORPUS_ROOT = Path(r".\datasets\Corpus")
OUTPUT_CSV = CORPUS_ROOT / "whisper_word_segments.csv"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
MODEL_NAME = "large-v3"
DEVICE = "cpu"   # 强制 CPU（稳定）


# ======================
# Utils
# ======================
def collect_all_audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def read_processed_files(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()

    processed = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["path"])
    return processed


def word_language(word: str) -> str:
    for ch in word:
        name = unicodedata.name(ch, "")
        if "DEVANAGARI" in name:
            return "hi"
        if "LATIN" in name:
            return "en"
    return "other"


# ======================
# Core processing
# ======================
def process_single_audio(
    whisper_model,
    align_model,
    metadata,
    audio_path: Path,
    writer: csv.DictWriter,
):
    # 1. Whisper transcription (sentence-level)
    result = whisper_model.transcribe(
        str(audio_path),
        verbose=False,
    )

    # 2. WhisperX alignment (word-level)
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        str(audio_path),
        DEVICE,
    )

    # 3. Write word-level rows
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            start = w.get("start")
            end = w.get("end")

            if not word or start is None or end is None:
                continue

            writer.writerow({
                "path": str(audio_path),
                "language": word_language(word),
                "start": start,
                "end": end,
                "word": word,
            })


# ======================
# Main
# ======================
def main():
    if not CORPUS_ROOT.exists():
        raise SystemExit(f"Corpus root does not exist: {CORPUS_ROOT}")

    audio_files = collect_all_audio_files(CORPUS_ROOT)
    if not audio_files:
        raise SystemExit("No audio files found.")

    processed_files = read_processed_files(OUTPUT_CSV)
    to_process = [p for p in audio_files if str(p) not in processed_files]

    if not to_process:
        print("All audio files have already been processed.")
        return

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"New audio files: {len(to_process)}")

    # Load models ONCE
    print("Loading Whisper model...")
    whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)

    print("Loading WhisperX alignment model...")
    # alignment language will be inferred per file,
    # but model itself can be reused
    # we load a dummy one first
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device=DEVICE,
    )

    write_header = not OUTPUT_CSV.exists()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "language",
                "start",
                "end",
                "word",
            ],
        )

        if write_header:
            writer.writeheader()
            f.flush()

        for audio_path in tqdm(to_process, desc="Processing audio", unit="file"):
            process_single_audio(
                whisper_model,
                align_model,
                metadata,
                audio_path,
                writer,
            )
            f.flush()

    print(f"Done. Results appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
