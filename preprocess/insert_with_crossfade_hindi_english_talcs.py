import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

import insert_with_crossfade as base


DEFAULT_TASKS = [
    {
        "name": "hindi_hi_into_talcs_en",
        "source_json": Path("datasets/Hindi-English_train/whisper_segment_Hindi-English_train.json"),
        "target_json": Path("datasets/TALCS_corpus/whisper_segment_TALCS_corpus.json"),
        "source_language": "hi",
        "target_language": "en",
    },
    {
        "name": "talcs_zh_into_hindi_en",
        "source_json": Path("datasets/TALCS_corpus/whisper_segment_TALCS_corpus.json"),
        "target_json": Path("datasets/Hindi-English_train/whisper_segment_Hindi-English_train.json"),
        "source_language": "zh",
        "target_language": "en",
    },
    {
        "name": "hindi_en_into_talcs_zh",
        "source_json": Path("datasets/Hindi-English_train/whisper_segment_Hindi-English_train.json"),
        "target_json": Path("datasets/TALCS_corpus/whisper_segment_TALCS_corpus.json"),
        "source_language": "en",
        "target_language": "zh",
    },
    {
        "name": "talcs_en_into_hindi_hi",
        "source_json": Path("datasets/TALCS_corpus/whisper_segment_TALCS_corpus.json"),
        "target_json": Path("datasets/Hindi-English_train/whisper_segment_Hindi-English_train.json"),
        "source_language": "en",
        "target_language": "hi",
    },
]


def build_language_windows(record: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    for seg in record.get("segments", []):
        if not isinstance(seg, dict):
            continue
        for span in seg.get("language_spans", []):
            if not isinstance(span, dict) or span.get("language") != language:
                continue
            try:
                start = float(span.get("start", 0.0))
                end = float(span.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            windows.append(
                {
                    "segment_id": seg.get("segment_id"),
                    "start": start,
                    "end": end,
                    "text": span.get("text", ""),
                    "scores": seg.get("scores", {}),
                    "language_spans": [span],
                }
            )
    windows.sort(key=lambda x: (x["start"], x["end"]))
    return windows


def build_language_candidates(records: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        windows = build_language_windows(record, language)
        if not windows:
            continue
        normalized = dict(record)
        normalized["whisper_language"] = language
        normalized["segments"] = windows
        normalized["language_window_count"] = len(windows)
        normalized["language_total_sec"] = round(sum(w["end"] - w["start"] for w in windows), 6)
        out.append(normalized)
    return out


def choose_insert_index_from_segments(
    target_segments: Any,
    target_len: int,
    target_sr: int,
    rng: random.Random,
) -> int:
    target_sec = target_len / target_sr
    windows = base.segment_windows(target_segments, target_sec, min_sec=0.05)
    if not windows:
        return rng.randint(0, target_len)

    for _ in range(20):
        start_sec, end_sec = windows[rng.randrange(len(windows))]
        if end_sec <= start_sec:
            continue
        insert_sec = rng.uniform(start_sec, end_sec)
        return max(0, min(int(round(insert_sec * target_sr)), target_len))

    return rng.randint(0, target_len)


def choose_insert_segment_from_language_windows(
    source_records: List[Dict[str, Any]],
    source_json_path: Path,
    project_root: Path,
    target_sr: int,
    clip_sec: float,
    rng: random.Random,
    max_tries: int,
) -> Optional[Dict[str, Any]]:
    if not source_records:
        return None

    for _ in range(max_tries):
        src_rec = source_records[rng.randrange(len(source_records))]
        src_raw_path = src_rec.get("path")
        if not isinstance(src_raw_path, str):
            continue

        src_path = base.resolve_audio_path(src_raw_path, project_root, source_json_path)
        if src_path is None:
            continue

        try:
            source_wav, source_sr = base.load_audio_mono(src_path)
        except (RuntimeError, OSError, ValueError):
            continue

        if source_wav.size == 0:
            continue

        if source_sr != target_sr:
            source_wav = base.resample_audio(source_wav, source_sr, target_sr)
            source_sr = target_sr

        total_sec = source_wav.size / source_sr
        windows = base.segment_windows(src_rec.get("segments", []), total_sec, min_sec=clip_sec)
        if not windows:
            continue

        for _ in range(20):
            win_start, win_end = windows[rng.randrange(len(windows))]
            max_start = win_end - clip_sec
            if max_start < win_start:
                continue
            clip_start_sec = rng.uniform(win_start, max_start) if max_start > win_start else win_start
            clip_end_sec = clip_start_sec + clip_sec
            s0 = int(round(clip_start_sec * source_sr))
            s1 = int(round(clip_end_sec * source_sr))
            clip = source_wav[s0:s1].copy()
            if clip.size == 0 or base.rms(clip) <= 1e-5:
                continue

            return {
                "clip": clip.astype(np.float32),
                "source_record": src_rec,
                "source_path": src_path,
                "source_start_sec": s0 / source_sr,
                "source_end_sec": s1 / source_sr,
                "pick_method": "language_spans",
            }

    return None


def process_one_pair(
    task: Dict[str, Any],
    project_root: Path,
    output_root: Path,
    rng: random.Random,
    insert_min_sec: float,
    insert_max_sec: float,
    crossfade_ms: float,
    noise_mix: float,
    noise_window_sec: float,
    max_gain_db: float,
    max_items_per_pair: Optional[int],
    max_source_tries: int,
) -> Dict[str, Any]:
    name = str(task["name"])
    source_json_path = (project_root / Path(task["source_json"])).resolve()
    target_json_path = (project_root / Path(task["target_json"])).resolve()
    source_language = str(task["source_language"]).lower()
    target_language = str(task["target_language"]).lower()

    source_records_all = base.load_manifest(source_json_path)
    target_records_all = base.load_manifest(target_json_path)
    source_records = build_language_candidates(source_records_all, source_language)
    target_records = build_language_candidates(target_records_all, target_language)

    pair_dir = output_root / name
    audio_dir = pair_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_json = pair_dir / "mixed_manifest.json"

    mixed_records: List[Dict[str, Any]] = []
    stats = {
        "task": name,
        "source_json": base.format_path(source_json_path, project_root),
        "target_json": base.format_path(target_json_path, project_root),
        "source_language_filter": source_language,
        "target_language_filter": target_language,
        "source_total_before_filter": len(source_records_all),
        "target_total_before_filter": len(target_records_all),
        "source_total": len(source_records),
        "target_total": len(target_records),
        "processed": 0,
        "saved": 0,
        "skip_short_target": 0,
        "skip_no_target_audio": 0,
        "skip_no_source_clip": 0,
        "errors": 0,
        "output_json": base.format_path(out_json, project_root),
    }

    limit = len(target_records) if max_items_per_pair is None else min(len(target_records), max_items_per_pair)
    for i in range(limit):
        stats["processed"] += 1
        tgt_rec = target_records[i]
        tgt_raw_path = tgt_rec.get("path")
        if not isinstance(tgt_raw_path, str):
            stats["skip_no_target_audio"] += 1
            continue

        tgt_path = base.resolve_audio_path(tgt_raw_path, project_root, target_json_path)
        if tgt_path is None:
            stats["skip_no_target_audio"] += 1
            continue

        try:
            target_wav, target_sr = base.load_audio_mono(tgt_path)
        except (RuntimeError, OSError, ValueError):
            stats["skip_no_target_audio"] += 1
            continue

        if target_wav.size == 0:
            stats["skip_no_target_audio"] += 1
            continue

        target_sec = target_wav.size / target_sr
        if 3.0 <= target_sec <= 5.0:
            insert_sec = insert_min_sec
        elif target_sec > 5.0:
            insert_sec = insert_max_sec
        else:
            stats["skip_short_target"] += 1
            continue

        if insert_sec > target_sec * 0.5:
            stats["skip_short_target"] += 1
            continue

        source_pick = choose_insert_segment_from_language_windows(
            source_records=source_records,
            source_json_path=source_json_path,
            project_root=project_root,
            target_sr=target_sr,
            clip_sec=insert_sec,
            rng=rng,
            max_tries=max_source_tries,
        )
        if source_pick is None:
            stats["skip_no_source_clip"] += 1
            continue

        try:
            insert_seg = source_pick["clip"]
            max_insert_samples = max(1, target_wav.size // 2)
            if insert_seg.size > max_insert_samples:
                insert_seg = insert_seg[:max_insert_samples].copy()
            min_insert_samples = max(1, int(np.ceil(insert_sec * target_sr)))
            if insert_seg.size < min_insert_samples:
                stats["skip_no_source_clip"] += 1
                continue

            insert_idx = choose_insert_index_from_segments(
                target_segments=tgt_rec.get("segments", []),
                target_len=target_wav.size,
                target_sr=target_sr,
                rng=rng,
            )

            local_ref_len = min(insert_seg.size, target_wav.size)
            ref_start = max(0, insert_idx - local_ref_len // 2)
            ref_end = min(target_wav.size, ref_start + local_ref_len)
            local_ref = target_wav[ref_start:ref_end]
            if local_ref.size == 0:
                local_ref = target_wav

            insert_seg = base.match_rms(insert_seg, base.rms(local_ref), max_gain_db=max_gain_db)

            if noise_mix > 0.0:
                noise_track = base.extract_noise_track(
                    target=target_wav,
                    sr=target_sr,
                    center_idx=insert_idx,
                    need_len=insert_seg.size,
                    noise_window_sec=noise_window_sec,
                )
                insert_seg = insert_seg + noise_mix * noise_track

            crossfade_samples = max(0, int(round(crossfade_ms * target_sr / 1000.0)))
            mixed = base.insert_with_crossfade(
                target=target_wav,
                insert_seg=insert_seg,
                insert_idx=insert_idx,
                crossfade_samples=crossfade_samples,
            )
            mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

            out_wav = audio_dir / f"{i:06d}_{Path(tgt_path).stem}.wav"
            sf.write(str(out_wav), mixed, target_sr)

            src_rec = source_pick["source_record"]
            mixed_records.append(
                {
                    "path": base.format_path(out_wav, project_root),
                    "target_original_path": tgt_raw_path,
                    "insert_source_path": src_rec.get("path"),
                    "target_whisper_language": tgt_rec.get("whisper_language"),
                    "insert_source_whisper_language": src_rec.get("whisper_language"),
                    "insert_at_sec": round(insert_idx / target_sr, 6),
                    "insert_duration_sec": round(insert_seg.size / target_sr, 6),
                    "source_clip_start_sec": (
                        round(float(source_pick["source_start_sec"]), 6)
                        if source_pick["source_start_sec"] is not None
                        else None
                    ),
                    "source_clip_end_sec": (
                        round(float(source_pick["source_end_sec"]), 6)
                        if source_pick["source_end_sec"] is not None
                        else None
                    ),
                    "source_pick_method": source_pick["pick_method"],
                    "crossfade_sec": round(crossfade_samples / target_sr, 6),
                    "sample_rate": int(target_sr),
                }
            )
            stats["saved"] += 1
        except Exception:
            stats["errors"] += 1

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(mixed_records, f, ensure_ascii=False, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create TALCS + Hindi-English crossfade insertions with language-constrained tasks: "
            "hi<->en and zh<->en."
        )
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/crossfade_hindi_english_talcs"))
    parser.add_argument("--insert-min-sec", type=float, default=1.3)
    parser.add_argument("--insert-max-sec", type=float, default=2.0)
    parser.add_argument("--crossfade-ms", type=float, default=80.0)
    parser.add_argument("--noise-mix", type=float, default=0.0)
    parser.add_argument("--noise-window-sec", type=float, default=2.0)
    parser.add_argument("--max-gain-db", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items-per-pair", type=int, default=None)
    parser.add_argument("--max-source-tries", type=int, default=60)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--clean-output",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Delete existing output-root before generation (default: True).",
        )
    else:
        clean_group = parser.add_mutually_exclusive_group()
        clean_group.add_argument("--clean-output", dest="clean_output", action="store_true")
        clean_group.add_argument("--no-clean-output", dest="clean_output", action="store_false")
        parser.set_defaults(clean_output=True)
    parser.add_argument("--clean-retries", type=int, default=8)
    parser.add_argument("--clean-wait-sec", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.insert_min_sec <= 0 or args.insert_max_sec <= 0:
        raise ValueError("insert-min-sec and insert-max-sec must be positive.")
    if args.insert_min_sec > args.insert_max_sec:
        raise ValueError("insert-min-sec must be <= insert-max-sec.")
    if args.clean_retries <= 0:
        raise ValueError("clean-retries must be positive.")
    if args.clean_wait_sec < 0:
        raise ValueError("clean-wait-sec must be >= 0.")

    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root if args.output_root.is_absolute() else (project_root / args.output_root)
    if output_root.exists() and args.clean_output:
        try:
            base.clean_output_directory(
                output_root=output_root,
                project_root=project_root,
                retries=int(args.clean_retries),
                wait_sec=float(args.clean_wait_sec),
            )
        except RuntimeError as exc:
            raise SystemExit(f"[Error] {exc}") from None
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    all_stats: List[Dict[str, Any]] = []

    for task in DEFAULT_TASKS:
        print(f"[Task] {task['name']}")
        stats = process_one_pair(
            task=task,
            project_root=project_root,
            output_root=output_root,
            rng=rng,
            insert_min_sec=float(args.insert_min_sec),
            insert_max_sec=float(args.insert_max_sec),
            crossfade_ms=float(args.crossfade_ms),
            noise_mix=float(np.clip(args.noise_mix, 0.0, 1.0)),
            noise_window_sec=float(args.noise_window_sec),
            max_gain_db=float(args.max_gain_db),
            max_items_per_pair=args.max_items_per_pair,
            max_source_tries=max(1, int(args.max_source_tries)),
        )
        all_stats.append(stats)
        print(
            "  "
            + f"source_total={stats['source_total']} "
            + f"target_total={stats['target_total']} "
            + f"processed={stats['processed']} "
            + f"saved={stats['saved']} "
            + f"skip_short_target={stats['skip_short_target']} "
            + f"skip_no_target_audio={stats['skip_no_target_audio']} "
            + f"skip_no_source_clip={stats['skip_no_source_clip']} "
            + f"errors={stats['errors']}"
        )
        print(f"  output_json={stats['output_json']}")

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"[Done] Summary saved to {base.format_path(summary_path, project_root)}")


if __name__ == "__main__":
    main()
