import argparse
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

MODEL_ID = "iic/speech_mossformer2_separation_temporal_8k"


def prepare_audio(input_path: Path, prepared_dir: Path, target_sr: int = 8000) -> Path:
    """Read audio, convert to mono, and resample to target sample rate."""
    audio, sr = sf.read(str(input_path))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)

    prepared_dir.mkdir(parents=True, exist_ok=True)
    out_path = prepared_dir / f"{input_path.stem}_{target_sr}hz.wav"
    sf.write(str(out_path), audio, target_sr)
    return out_path


def collect_inputs(project_root: Path, input_wav: str | None, input_dir: str, pattern: str) -> list[Path]:
    if input_wav:
        p = Path(input_wav).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input wav not found: {p}")
        return [p]

    dir_path = Path(input_dir).expanduser().resolve()
    if dir_path.exists():
        matches = sorted(dir_path.glob(pattern))
        if matches:
            return matches

    fallback = (project_root / "mixed.wav").resolve()
    if fallback.exists():
        return [fallback]

    raise FileNotFoundError(
        f"No input found. Checked: dir={dir_path} pattern={pattern}, fallback={fallback}"
    )


def run_one_file(mixed_path: Path, separator, output_dir: Path, target_sr: int = 8000) -> list[Path]:
    prepared = prepare_audio(mixed_path, output_dir / "_prepared", target_sr=target_sr)
    result = separator(str(prepared))

    sample_out_dir = output_dir / mixed_path.stem
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, signal in enumerate(result["output_pcm_list"]):
        out_path = sample_out_dir / f"spk{i}.wav"
        sf.write(str(out_path), np.frombuffer(signal, dtype=np.int16), target_sr)
        saved.append(out_path)
    return saved


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run MossFormer2 speech separation.")
    parser.add_argument(
        "--input_wav",
        type=str,
        default=None,
        help="Single mixed wav path. If set, only this file is processed.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(project_root / "preprocess" / "generated_batch"),
        help="Directory containing mixed wav files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_mixed.wav",
        help="Glob pattern for batch mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "split_models" / "mossformer2" / "outputs"),
        help="Directory for separated outputs.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=8000,
        help="Target sample rate for model input.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help="ModelScope model id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    inputs = collect_inputs(project_root, args.input_wav, args.input_dir, args.pattern)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = pipeline(Tasks.speech_separation, model=args.model_id)

    print(f"Found {len(inputs)} input file(s).")
    all_saved: list[Path] = []
    for idx, mixed_path in enumerate(inputs, start=1):
        print(f"[{idx}/{len(inputs)}] Processing: {mixed_path}")
        saved = run_one_file(mixed_path, separator, output_dir, target_sr=args.target_sr)
        all_saved.extend(saved)

    print("Saved separated files:")
    for p in all_saved:
        print(f" - {p}")


if __name__ == "__main__":
    main()
