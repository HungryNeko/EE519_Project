from pathlib import Path
import csv
from typing import List, Set

import torch
import whisper
from langdetect import detect
from tqdm import tqdm


# ===== Fixed configuration =====
CORPUS_ROOT = Path(r".\datasets\Corpus")
OUTPUT_CSV = CORPUS_ROOT / "whisper_segments.csv"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
MODEL_NAME = "large-v3"


def collect_all_audio_files(root: Path) -> List[Path]:
    """Recursively collect all audio files under corpus root."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def read_processed_files(csv_path: Path) -> Set[str]:
    """
    Read existing CSV and return a set of audio paths
    that have already been processed.
    """
    if not csv_path.exists():
        return set()

    processed = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "path" in row:
                processed.add(row["path"])
    return processed


def safe_detect_language(text: str) -> str:
    """Detect language, return raw label or 'unknown'."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def process_single_audio(
    model,
    audio_path: Path,
    writer: csv.DictWriter,
    use_fp16: bool,
):
    """
    Process one audio file.
    Merge consecutive segments with the same language
    and write one CSV row per language chunk.
    """

    transcribe = model.transcribe(
        str(audio_path),
        task="transcribe",
        verbose=False,
        fp16=use_fp16,
    )

    translate = model.transcribe(
        str(audio_path),
        task="translate",
        verbose=False,
        fp16=use_fp16,
    )

    merged_chunks = []

    for seg_o, seg_e in zip(transcribe["segments"], translate["segments"]):
        orig = seg_o["text"].strip()
        en = seg_e["text"].strip()

        if not orig and not en:
            continue

        try:
            lang = detect(orig)
        except Exception:
            lang = "unknown"

        start = seg_o["start"]
        end = seg_o["end"]

        # merge with previous chunk if same language
        if (
            merged_chunks
            and merged_chunks[-1]["language"] == lang
        ):
            merged_chunks[-1]["end"] = end
            merged_chunks[-1]["original_text"] += " " + orig
            merged_chunks[-1]["english_text"] += " " + en
        else:
            merged_chunks.append({
                "path": str(audio_path),
                "language": lang,
                "start": start,
                "end": end,
                "original_text": orig,
                "english_text": en,
            })

    # write chunks immediately
    for chunk in merged_chunks:
        writer.writerow(chunk)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"New audio files to process: {len(to_process)}")

    model = whisper.load_model(MODEL_NAME, device=device)

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
                "original_text",
                "english_text",
            ],
        )

        if write_header:
            writer.writeheader()
            f.flush()

        for audio_path in tqdm(to_process, desc="Processing audio files", unit="file"):
            process_single_audio(model, audio_path, writer, use_fp16)
            f.flush()  # ensure data is written after each audio file

    print(f"Done. Results appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
