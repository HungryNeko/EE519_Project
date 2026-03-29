"""
Demo Pipeline: Two-Speaker Multilingual Transcription
======================================================
Input : mixed audio with two speakers (any sample rate)
Output: per-speaker transcript, with language-switch attribution fixed

Pipeline
--------
1. MossFormer2 speaker separation  →  spk0.wav / spk1.wav  (8 kHz)
2. WhisperX transcription          →  word-level timestamps + phoneme alignment
                                       + per-word language detection
3. Mixed-language segment check    →  TDNN same-speaker classifier
      same speaker  (code-switch)  →  keep text as-is
      diff speaker  (cross-talk)   →  timestamp-based word reassignment
4. Print final per-speaker transcript

Usage
-----
    python demo_pipeline.py path/to/mixed.wav
    python demo_pipeline.py path/to/mixed.wav --whisper-model small --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import librosa
import numpy as np
import soundfile as sf

# ── make project root importable ─────────────────────────────────────────────
# NOTE: append instead of insert(0) so that installed packages (e.g. the
# HuggingFace `datasets` library) are NOT shadowed by the local datasets/
# directory that lives in the project root.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from dl_model.final_model.model import TDNNPredictor  # TDNN same-speaker model

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MOSSFORMER_SR = 8000   # MossFormer2 requires 8 kHz input
WHISPER_SR    = 16000  # Whisper / TDNN require 16 kHz
MIN_SPAN_SEC  = 0.10   # minimum audio span length for TDNN (100 ms)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Speaker Separation (MossFormer2)
# ─────────────────────────────────────────────────────────────────────────────

def _import_modelscope():
    """
    Import modelscope while ensuring the local datasets/ folder in the project
    root does NOT shadow the HuggingFace `datasets` package that modelscope needs.

    Strategy: temporarily remove every sys.path entry that points to the project
    root (or the current working directory) before the import, then restore them.
    """
    import importlib
    import os

    # Entries to hide during the import
    _root_str  = str(_ROOT)
    _cwd_strs  = {"", ".", os.getcwd()}
    _hide = {p for p in sys.path if p in _cwd_strs or p == _root_str}

    saved_path = sys.path[:]
    for p in _hide:
        while p in sys.path:          # may appear more than once
            sys.path.remove(p)

    # Also evict any already-cached stub for the local datasets namespace pkg
    for key in list(sys.modules.keys()):
        if key == "datasets" or key.startswith("datasets."):
            del sys.modules[key]

    try:
        import resampy
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as exc:
        raise ImportError(
            "ModelScope and resampy are required for speaker separation.\n"
            "Install them in the active conda env:\n"
            "  pip install modelscope resampy"
        ) from exc
    finally:
        sys.path[:] = saved_path   # always restore

    return ms_pipeline, Tasks


def separate_speakers(audio_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Run MossFormer2 on *audio_path* and write spk0.wav / spk1.wav to output_dir.
    Returns (spk0_path, spk1_path).
    """
    ms_pipeline, Tasks = _import_modelscope()
    try:
        import resampy
    except ImportError as exc:
        raise ImportError("pip install resampy") from exc

    # Prepare mono 8 kHz audio
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MOSSFORMER_SR:
        audio = resampy.resample(audio, sr, MOSSFORMER_SR)

    prep_path = output_dir / "_prepared_8k.wav"
    sf.write(str(prep_path), audio, MOSSFORMER_SR)

    separator = ms_pipeline(
        Tasks.speech_separation,
        model="iic/speech_mossformer2_separation_temporal_8k",
    )
    result = separator(str(prep_path))

    paths: List[Path] = []
    for i, pcm in enumerate(result["output_pcm_list"]):
        p = output_dir / f"spk{i}.wav"
        sf.write(str(p), np.frombuffer(pcm, dtype=np.int16), MOSSFORMER_SR)
        paths.append(p)

    if len(paths) < 2:
        raise RuntimeError(f"MossFormer2 returned only {len(paths)} speaker(s); expected 2.")
    return paths[0], paths[1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – WhisperX Transcription with word timestamps + language tagging
# ─────────────────────────────────────────────────────────────────────────────

def _detect_lang_by_char(ch: str) -> str:
    """Classify a character's language by Unicode range."""
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:          # CJK unified
        return "zh"
    if 0x0900 <= o <= 0x097F:          # Devanagari (Hindi)
        return "hi"
    if "a" <= ch.lower() <= "z":
        return "en"
    return "other"


def _group_words_by_language(words: List[dict]) -> List[dict]:
    """
    Group a flat word list into consecutive same-language spans.
    Each span: {language, start, end, text, words}
    """
    spans: List[dict] = []
    cur: Optional[dict] = None

    def flush() -> None:
        nonlocal cur
        if cur:
            cur["text"] = " ".join(w["word"] for w in cur["words"])
            spans.append(cur)
            cur = None

    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        lang = _detect_lang_by_char(token[0])
        if lang == "other":
            flush()
            continue
        if cur is None or cur["language"] != lang:
            flush()
            cur = {"language": lang, "start": w["start"], "end": w["end"],
                   "text": "", "words": [w]}
        else:
            cur["end"] = w["end"]
            cur["words"].append(w)

    flush()
    return spans


def transcribe_speaker(wx_model, audio_path: Path, device: str = "cpu") -> List[dict]:
    """
    Transcribe one speaker's audio with WhisperX.

    WhisperX pipeline:
      1. transcribe()  – CTC-based fast ASR, returns segment-level text + language
      2. load_align_model() + align()  – phoneme-forced alignment for precise
         word timestamps (falls back to transcription timestamps on failure)

    Returns a list of segment dicts:
        {segment_id, start, end, text, language_spans}

    Each language_span: {language, start, end, text, words}
    Each word:          {word, start, end, score, language}
    """
    import whisperx

    # Load and resample to 16 kHz (MossFormer2 outputs 8 kHz)
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != WHISPER_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)

    # ── 1. Transcription ─────────────────────────────────────────────────────
    result = wx_model.transcribe(audio, batch_size=8)
    detected_lang = result.get("language", "zh")

    # ── 2. Phoneme alignment for precise word timestamps ─────────────────────
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        aligned = whisperx.align(
            result["segments"], align_model, align_meta, audio, device,
            return_char_alignments=False,
        )
        segments_src = aligned["segments"]
    except Exception:
        # Alignment may fail for unsupported languages or very short clips;
        # fall back to raw transcription timestamps.
        segments_src = result["segments"]

    # ── 3. Build output segments ──────────────────────────────────────────────
    segments_out: List[dict] = []

    for i, seg in enumerate(segments_src):
        t0 = float(seg.get("start", 0.0))
        t1 = float(seg.get("end", t0))

        # Collect word-level entries from the WhisperX output
        words: List[dict] = []
        for w in seg.get("words", []):
            w_start = w.get("start")
            w_end   = w.get("end")
            token   = w.get("word", "").strip()
            if not token or w_start is None or w_end is None:
                continue
            words.append({
                "word":     token,
                "start":    float(w_start),
                "end":      float(w_end),
                "score":    float(w.get("score", 0.0)),
                "language": _detect_lang_by_char(token[0]),
            })

        spans = _group_words_by_language(words)
        if not spans:
            spans = [{
                "language": detected_lang,
                "start": t0, "end": t1,
                "text": seg.get("text", "").strip(),
                "words": [],
            }]

        segments_out.append({
            "segment_id": i,
            "start": t0, "end": t1,
            "text": seg.get("text", "").strip(),
            "language_spans": spans,
        })

    return segments_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – TDNN same-speaker check on mixed-language segments
# ─────────────────────────────────────────────────────────────────────────────

def _load_window(audio_path: Path, t0: float, t1: float) -> np.ndarray:
    """Load a time window from a wav file, resampled to WHISPER_SR (16 kHz)."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != WHISPER_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)
    a = max(0, int(t0 * WHISPER_SR))
    b = min(len(audio), int(t1 * WHISPER_SR))
    chunk = audio[a:b]
    # Pad to minimum 100 ms so TDNN has enough context
    min_len = int(MIN_SPAN_SEC * WHISPER_SR)
    if len(chunk) < min_len:
        chunk = np.pad(chunk, (0, min_len - len(chunk)))
    return chunk


def check_mixed_segments(
    segments: List[dict],
    audio_path: Path,
    predictor: TDNNPredictor,
) -> List[dict]:
    """
    For each segment with adjacent spans of *different* language, run the TDNN.

    Returns a list of check results:
        {segment_id, pair_idx, span_a, span_b, is_same_speaker, confidence}

    is_same_speaker=True  → code-switch (one speaker, two languages)
    is_same_speaker=False → cross-talk  (different speakers bled together)
    """
    results: List[dict] = []

    for seg in segments:
        spans = seg["language_spans"]
        for k in range(len(spans) - 1):
            sa, sb = spans[k], spans[k + 1]
            if sa["language"] == sb["language"]:
                continue  # no language switch, skip

            wav_a = _load_window(audio_path, sa["start"], sa["end"])
            wav_b = _load_window(audio_path, sb["start"], sb["end"])

            is_same, conf = predictor.predict(wav_a, wav_b)
            results.append({
                "segment_id": seg["segment_id"],
                "pair_idx":   k,
                "span_a":     sa,
                "span_b":     sb,
                "is_same_speaker": is_same,
                "confidence":      conf,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Fix cross-talk using Whisper timestamps (non-LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_all_words(segments: List[dict]) -> List[dict]:
    words: List[dict] = []
    for seg in segments:
        for span in seg["language_spans"]:
            for w in span.get("words", []):
                words.append(w)
    return sorted(words, key=lambda x: x["start"])


def _overlapping_indices(
    word_list: List[dict],
    t0: float,
    t1: float,
    margin: float = 0.12,
) -> Set[int]:
    """Return indices in word_list whose time range overlaps [t0-margin, t1+margin]."""
    return {
        i for i, w in enumerate(word_list)
        if w["start"] < t1 + margin and w["end"] > t0 - margin
    }


def fix_crosstalk(
    spk0_segments: List[dict],
    spk1_segments: List[dict],
    cross0: List[dict],
    cross1: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """
    Remove cross-talk words from each speaker's word list.

    Algorithm (non-LLM, timestamp-based):
    ──────────────────────────────────────
    For every language-switch pair flagged as 'different speaker' in spkX:
      • The 'foreign' span (the one that doesn't belong to spkX) is a bleed-through
        of the other speaker captured by the separator.
      • We check whether the *other* speaker's transcript already contains words
        at the same timestamp range (overlap > margin).
      • If yes  →  the foreign span is a duplicate; drop it from spkX.
      • If no   →  there is no alternative source; keep it in spkX to avoid
                   silent gaps (the separator may have assigned it here solely).
    """
    spk0_words = _extract_all_words(spk0_segments)
    spk1_words = _extract_all_words(spk1_segments)

    drop0: Set[int] = set()   # indices in spk0_words to remove
    drop1: Set[int] = set()   # indices in spk1_words to remove

    def _dominant_lang(word_list: List[dict], t0: float, t1: float) -> str:
        """Return the most common language among words in [t0, t1]."""
        langs: Dict[str, int] = {}
        for w in word_list:
            if w["start"] >= t0 and w["end"] <= t1 + 0.05:
                langs[w.get("language", "other")] = langs.get(w.get("language", "other"), 0) + 1
        return max(langs, key=langs.get) if langs else "other"

    for cr in cross0:
        if cr["is_same_speaker"]:
            continue  # genuine code-switch, keep both spans
        for span in (cr["span_a"], cr["span_b"]):
            t0, t1 = span["start"], span["end"]
            # Check whether spk1 already covers these timestamps
            if _overlapping_indices(spk1_words, t0, t1):
                # spk1 has content here → this is bleed-through in spk0, drop it
                drop0.update(_overlapping_indices(spk0_words, t0, t1))

    for cr in cross1:
        if cr["is_same_speaker"]:
            continue
        for span in (cr["span_a"], cr["span_b"]):
            t0, t1 = span["start"], span["end"]
            if _overlapping_indices(spk0_words, t0, t1):
                drop1.update(_overlapping_indices(spk1_words, t0, t1))

    spk0_final = [w for i, w in enumerate(spk0_words) if i not in drop0]
    spk1_final = [w for i, w in enumerate(spk1_words) if i not in drop1]
    return spk0_final, spk1_final


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Format output
# ─────────────────────────────────────────────────────────────────────────────

def _format_transcript(words: List[dict], label: str) -> str:
    """Render a word list as a readable transcript string."""
    if not words:
        return f"[{label}]: (no speech detected)"
    # Merge consecutive same-language runs with a separator
    lines: List[str] = []
    cur_lang = words[0].get("language", "?")
    buf: List[str] = []

    for w in words:
        lang = w.get("language", "?")
        if lang != cur_lang:
            lines.append(f"<{cur_lang}> {' '.join(buf)}")
            buf = [w["word"]]
            cur_lang = lang
        else:
            buf.append(w["word"])
    if buf:
        lines.append(f"<{cur_lang}> {' '.join(buf)}")

    return f"[{label}]:\n" + "\n".join(f"  {ln}" for ln in lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    output_dir: str = "demo/demo_output",
    whisper_model_name: str = "large-v3",
    device: str = "cpu",
    tdnn_weight: Optional[str] = None,
    ) -> None:
    import whisperx

    audio_path_obj = Path(audio_path).resolve()
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path_obj}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # WhisperX compute type: float16 on CUDA, int8 on CPU (faster than float32)
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n{'='*60}")
    print(f"  Two-Speaker Transcription Demo")
    print(f"  Input : {audio_path_obj.name}")
    print(f"  Device: {device}  |  WhisperX: {whisper_model_name}  |  compute: {compute_type}")
    print(f"{'='*60}\n")

    # ── 1. Speaker Separation ─────────────────────────────────────────────
    print("[1/4] Separating speakers  (MossFormer2) ...")
    spk0_path, spk1_path = separate_speakers(audio_path_obj, out_dir)
    print(f"      spk0 → {spk0_path}")
    print(f"      spk1 → {spk1_path}\n")

    # ── 2. Transcription ──────────────────────────────────────────────────
    print(f"[2/4] Transcribing with WhisperX ({whisper_model_name}) ...")
    wx_model = whisperx.load_model(whisper_model_name, device=device,
                                   compute_type=compute_type)

    spk0_segs = transcribe_speaker(wx_model, spk0_path, device=device)
    spk1_segs = transcribe_speaker(wx_model, spk1_path, device=device)

    n_mixed0 = sum(1 for s in spk0_segs if len(s["language_spans"]) > 1)
    n_mixed1 = sum(1 for s in spk1_segs if len(s["language_spans"]) > 1)
    print(f"      spk0: {len(spk0_segs)} segments, {n_mixed0} with language switches")
    print(f"      spk1: {len(spk1_segs)} segments, {n_mixed1} with language switches\n")

    # ── 3. TDNN same-speaker check ────────────────────────────────────────
    print("[3/4] Checking mixed-language spans with TDNN ...")
    default_weight = "dl_model/final_model/tdnn_full_best_acc.pth"
    tdnn_kwargs: dict = {
        "device": device,
        "weight_path": tdnn_weight if tdnn_weight else default_weight,
    }

    predictor = TDNNPredictor(**tdnn_kwargs)

    cross0 = check_mixed_segments(spk0_segs, spk0_path, predictor)
    cross1 = check_mixed_segments(spk1_segs, spk1_path, predictor)

    print("\n      TDNN Prediction Details:")
    print("      ----------------------------------------")

    for label, results in (("spk0", cross0), ("spk1", cross1)):
        if not results:
            print(f"      {label}: (no mixed-language pairs)")
            continue

        for r in results:
            is_same = r["is_same_speaker"]
            conf = r["confidence"]

            verdict = "SAME speaker (code-switch → KEEP)" if is_same \
                    else "DIFFERENT speakers (cross-talk → FIX)"

            print(f"""
        [{label}] Segment {r['segment_id']} | Pair {r['pair_idx']}
            Languages : {r['span_a']['language']} → {r['span_b']['language']}
            Time      : {r['span_a']['start']:.2f}s - {r['span_b']['end']:.2f}s
            Prediction: {is_same}
            Confidence: {conf:.4f}
            Result    : {verdict}
            ----------------------------------------
            """)       

    if not cross0 and not cross1:
        print("      (no mixed-language segments found)")
    print()

    # ── 4. Fix cross-talk ─────────────────────────────────────────────────
    print("[4/4] Reassigning cross-talk words by timestamp overlap ...")
    spk0_words, spk1_words = fix_crosstalk(spk0_segs, spk1_segs, cross0, cross1)

    dropped0 = len(_extract_all_words(spk0_segs)) - len(spk0_words)
    dropped1 = len(_extract_all_words(spk1_segs)) - len(spk1_words)
    print(f"      spk0: removed {dropped0} cross-talk word(s)")
    print(f"      spk1: removed {dropped1} cross-talk word(s)\n")

    # ── 5. Print final transcript ─────────────────────────────────────────
    print(f"{'='*60}")
    print("  FINAL TRANSCRIPT")
    print(f"{'='*60}")
    print(_format_transcript(spk0_words, "Speaker 0"))
    print()
    print(_format_transcript(spk1_words, "Speaker 1"))
    print(f"{'='*60}\n")
    final_text = (
        _format_transcript(spk0_words, "Speaker 0")
        + "\n\n"
        + _format_transcript(spk1_words, "Speaker 1")
    )

    with open(out_dir / "final_transcript.txt", "w") as f:
        f.write(final_text)
        


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end two-speaker multilingual transcription demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to the mixed two-speaker audio file (wav/mp3/flac/…).",
    )
    parser.add_argument(
        "--output-dir",
        default="demo/demo_output",
        help="Directory for intermediate separated audio files.",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper and TDNN inference.",
    )
    parser.add_argument(
        "--tdnn-weight",
        default=None,
        type=str,
        help=(
            "Path to TDNN checkpoint (.pth).  "
            "Defaults to dl_model/final_model/tdnn_full_best_acc.pth."
        ),
    )
    args = parser.parse_args()

    run_pipeline(
        audio_path=args.audio,
        output_dir=args.output_dir,
        whisper_model_name=args.whisper_model,
        device=args.device,
        tdnn_weight=args.tdnn_weight,
    )
