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
        "name": "switchlingua_hi_into_seame_en",
        "source_json": Path("datasets/switchlingua_audio_2000/whisper_segment_switchlingua_audio_2000_en_hi_only.json"),
        "target_json": Path("datasets/seame_dev_sge_cn_en_2000/whisper_segment_seame_dev_sge_cn_en_2000_en_zh_only.json"),
        "source_language": "hi",
        "target_language": "en",
    },
    {
        "name": "seame_zh_into_switchlingua_en",
        "source_json": Path("datasets/seame_dev_sge_cn_en_2000/whisper_segment_seame_dev_sge_cn_en_2000_en_zh_only.json"),
        "target_json": Path("datasets/switchlingua_audio_2000/whisper_segment_switchlingua_audio_2000_en_hi_only.json"),
        "source_language": "zh",
        "target_language": "en",
    },
    {
        "name": "switchlingua_en_into_seame_zh",
        "source_json": Path("datasets/switchlingua_audio_2000/whisper_segment_switchlingua_audio_2000_en_hi_only.json"),
        "target_json": Path("datasets/seame_dev_sge_cn_en_2000/whisper_segment_seame_dev_sge_cn_en_2000_en_zh_only.json"),
        "source_language": "en",
        "target_language": "zh",
    },
    {
        "name": "seame_en_into_switchlingua_hi",
        "source_json": Path("datasets/seame_dev_sge_cn_en_2000/whisper_segment_seame_dev_sge_cn_en_2000_en_zh_only.json"),
        "target_json": Path("datasets/switchlingua_audio_2000/whisper_segment_switchlingua_audio_2000_en_hi_only.json"),
        "source_language": "en",
        "target_language": "hi",
    },
]


def filter_records_by_language(records: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    return [record for record in records if str(record.get("whisper_language") or "").lower() == language]


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
    source_records = filter_records_by_language(source_records_all, source_language)
    target_records = filter_records_by_language(target_records_all, target_language)

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

        source_pick = base.choose_insert_segment(
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

            insert_idx = rng.randint(0, target_wav.size)

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
            "Create switchlingua/seame crossfade insertions with language-constrained tasks: "
            "hi<->en and zh<->en, preserving the original crossfade generation logic."
        )
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/crossfade_switchlingua_seame"))
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
