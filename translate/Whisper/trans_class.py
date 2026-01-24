import whisper
from langdetect import detect
from collections import defaultdict
from typing import List


class MultiAudioLanguageRebuilder:
    """
    Rebuild multi-speaker audio into language-wise streams
    while preserving global time order.
    """

    def __init__(self, model_name: str = "large-v3"):
        self.model = whisper.load_model(model_name)

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
                verbose=False
            )

            translate_result = self.model.transcribe(
                audio_path,
                task="translate",
                verbose=False
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


if __name__=="__main__":
    audio_files = [
        "../../split_models/mossformer2/output_spk0.wav",
        "../../split_models/mossformer2/output_spk1.wav"
    ]

    translator = MultiAudioLanguageRebuilder()
    translator.process(audio_files)

