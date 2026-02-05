import argparse
from collections import defaultdict
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

        # Step 4: rebuild by language (preserve time order)
        lang_to_segments = defaultdict(list)

        for seg in all_segments:
            lang_to_segments[seg["lang"]].append(seg)

        # Step 5: output
        print("\n========== Final Output ==========")

        for lang, segs in lang_to_segments.items():
            print(f"\n--- Original Language: {lang} ---")

            print("\n[Original Text]")
            print(" ".join(s["orig_text"] for s in segs))

            print("\n[English Translation]")
            print(" ".join(s["en_text"] for s in segs))

    def process_files(self, audio_paths: List[str]):
        for audio_path in audio_paths:
            print(f"\n========== File: {audio_path} ==========")
            self.process([audio_path])


def _collect_audio_files(audio_dir: str, exts: Iterable[str]):
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    files = [
        p for p in audio_path.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test Whisper language detection on a dataset.")
    parser.add_argument(
        "--data-dir",
        default=r".\datasets\Corpus\adult\audio\test_split",
        help="Directory containing audio files."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Limit the number of files for a quick test. Use 0 or negative for all files."
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name."
    )
    args = parser.parse_args()

    audio_files = _collect_audio_files(args.data_dir, exts=(".wav", ".mp3", ".flac", ".m4a"))
    if args.max_files and args.max_files > 0:
        audio_files = audio_files[:args.max_files]

    if not audio_files:
        raise SystemExit(f"No audio files found in {args.data_dir}")

    print(f"Found {len(audio_files)} audio files in {args.data_dir}")
    translator = MultiAudioLanguageRebuilder(model_name=args.model)
    translator.process_files([str(p) for p in audio_files])

