import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import whisper
from langdetect import detect


class MultiAudioLanguageRebuilder:
    """
    Rebuild multi-speaker audio into language-wise streams
    while preserving global time order.
    """

    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_fp16 = device == "cuda"
        self.model = whisper.load_model(model_name, device=device)
        if self.model.device.type != self.device:
            print(
                f"Warning: requested device '{self.device}' but model is on "
                f"'{self.model.device.type}'."
            )

    def _safe_detect(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def process(self, audio_paths: List[str]):
        all_segments = []

        # Step 1: process each audio independently
        for speaker_id, audio_path in enumerate(audio_paths):
            transcribe_result = self.model.transcribe(
                audio_path,
                task="transcribe",
                verbose=False,
                fp16=self.use_fp16
            )

            translate_result = self.model.transcribe(
                audio_path,
                task="translate",
                verbose=False,
                fp16=self.use_fp16
            )

            # Step 2: collect aligned segments
            for seg_orig, seg_en in zip(
                transcribe_result["segments"],
                translate_result["segments"]
            ):
                orig_text = seg_orig["text"].strip()
                en_text = seg_en["text"].strip()

                if not orig_text or not en_text:
                    continue

                lang = self._safe_detect(orig_text)

                all_segments.append({
                    "start": seg_orig["start"],
                    "end": seg_orig["end"],
                    "lang": lang,
                    "orig_text": orig_text,
                    "en_text": en_text
                })

        # Step 3: global time sort
        all_segments.sort(key=lambda x: x["start"])
        return all_segments

    def process_file(self, audio_path: str):
        return self.process([audio_path])


def _collect_audio_files(audio_dir: str, exts: Iterable[str]):
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    files = [
        p for p in audio_path.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files)

def _load_processed_files(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    processed = set()
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get("file_path")
                if file_path:
                    processed.add(file_path)
    except Exception:
        return set()
    return processed


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Whisper language detection on a dataset.")
    parser.add_argument(
        "--data-dir",
        default=r"datasets\Corpus\adult\audio\test_split",
        help="Directory containing audio files."
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Force device. 'auto' uses CUDA if available."
    )
    parser.add_argument(
        "--output",
        default="language_detect.csv",
        help="CSV output path."
    )
    args = parser.parse_args()

    audio_files = _collect_audio_files(args.data_dir, exts=(".wav", ".mp3", ".flac", ".m4a"))
    if not audio_files:
        raise SystemExit(f"No audio files found in {args.data_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")
    device = None if args.device == "auto" else args.device
    translator = MultiAudioLanguageRebuilder(model_name=args.model, device=device)

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {translator.device}")
    print(f"Whisper model device: {translator.model.device}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(audio_files)} audio files in {args.data_dir}")
    print(f"Writing CSV to {output_path}")

    processed_files = _load_processed_files(output_path)
    if processed_files:
        print(f"Detected {len(processed_files)} processed files in CSV. Will skip them.")

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    write_header = not file_exists
    open_mode = "a" if file_exists else "w"

    with output_path.open(open_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["file_path", "language", "time_stamp", "content", "translation"])

        for p in audio_files:
            file_path = str(p)
            if file_path in processed_files:
                print(f"Skipping (already processed): {file_path}")
                continue
            print(f"Processing: {file_path}")
            segments = translator.process_file(file_path)
            for seg in segments:
                time_stamp = f"{seg['start']:.3f}-{seg['end']:.3f}"
                writer.writerow([
                    file_path,
                    seg["lang"],
                    time_stamp,
                    seg["orig_text"],
                    seg["en_text"],
                ])

