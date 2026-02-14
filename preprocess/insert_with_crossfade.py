import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


EPS = 1e-9

DEFAULT_TASKS = [
    {
        "name": "corpus_en_into_ascend_zh",
        "source_json": Path("datasets/Corpus/corpus_en_language.json"),
        "target_json": Path("datasets/ascend/ascend_zh_language.json"),
    },
    {
        "name": "corpus_hi_into_ascend_en",
        "source_json": Path("datasets/Corpus/corpus_hi_language.json"),
        "target_json": Path("datasets/ascend/ascend_en_language.json"),
    },
    {
        "name": "ascend_en_into_corpus_hi",
        "source_json": Path("datasets/ascend/ascend_en_language.json"),
        "target_json": Path("datasets/Corpus/corpus_hi_language.json"),
    },
    {
        "name": "ascend_zh_into_corpus_en",
        "source_json": Path("datasets/ascend/ascend_zh_language.json"),
        "target_json": Path("datasets/Corpus/corpus_en_language.json"),
    },
]


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = signal.resample_poly(x, up=up, down=down)
    return y.astype(np.float32)


def to_samples(sec: float, sr: int, max_len: int) -> int:
    return max(0, min(int(round(sec * sr)), max_len))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + EPS))


def match_rms(src: np.ndarray, ref_rms: float, max_gain_db: float = 12.0) -> np.ndarray:
    src_r = rms(src)
    gain = ref_rms / (src_r + EPS)
    max_gain = 10 ** (max_gain_db / 20.0)
    min_gain = 1.0 / max_gain
    gain = float(np.clip(gain, min_gain, max_gain))
    return (src * gain).astype(np.float32)


def frame_rms(signal_1d: np.ndarray, frame_len: int, hop_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if signal_1d.size < frame_len:
        return np.empty((0, frame_len), dtype=np.float32), np.empty((0,), dtype=np.float32)
    starts = np.arange(0, signal_1d.size - frame_len + 1, hop_len)
    frames = np.stack([signal_1d[s : s + frame_len] for s in starts], axis=0)
    rms_values = np.sqrt(np.mean(frames * frames, axis=1) + EPS).astype(np.float32)
    return frames.astype(np.float32), rms_values


def extract_noise_track(
    target: np.ndarray,
    sr: int,
    center_idx: int,
    need_len: int,
    noise_window_sec: float = 2.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    noise_percentile: float = 25.0,
) -> np.ndarray:
    if need_len <= 0:
        return np.zeros((0,), dtype=np.float32)

    total_len = target.size
    win_len = max(1, int(round(noise_window_sec * sr)))
    frame_len = max(1, int(round(frame_ms * sr / 1000.0)))
    hop_len = max(1, int(round(hop_ms * sr / 1000.0)))

    start = max(0, center_idx - win_len // 2)
    end = min(total_len, start + win_len)
    context = target[start:end]
    if context.size < frame_len:
        return np.zeros((need_len,), dtype=np.float32)

    frames, rms_values = frame_rms(context, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros((need_len,), dtype=np.float32)

    threshold = np.quantile(rms_values, noise_percentile / 100.0)
    selected = frames[rms_values <= threshold]
    if selected.shape[0] == 0:
        argmin_idx = int(np.argmin(rms_values))
        selected = frames[argmin_idx : argmin_idx + 1]

    noise = selected.reshape(-1)
    if noise.size < need_len:
        repeats = int(np.ceil(need_len / noise.size))
        noise = np.tile(noise, repeats)
    return noise[:need_len].astype(np.float32)


def insert_with_crossfade(
    target: np.ndarray,
    insert_seg: np.ndarray,
    insert_idx: int,
    crossfade_samples: int,
) -> np.ndarray:
    target_len = target.size
    insert_idx = max(0, min(insert_idx, target_len))

    prefix = target[:insert_idx]
    suffix = target[insert_idx:]
    ins = insert_seg

    if crossfade_samples <= 0:
        return np.concatenate([prefix, ins, suffix], axis=0).astype(np.float32)

    prefix_len = prefix.size
    suffix_len = suffix.size
    ins_len = ins.size

    start_cf = min(crossfade_samples, prefix_len, ins_len)
    end_cf = min(crossfade_samples, suffix_len, ins_len - start_cf)

    if start_cf + end_cf > ins_len:
        overflow = start_cf + end_cf - ins_len
        end_cf = max(0, end_cf - overflow)
    if start_cf + end_cf > ins_len:
        overflow = start_cf + end_cf - ins_len
        start_cf = max(0, start_cf - overflow)

    parts: List[np.ndarray] = []

    if start_cf > 0:
        parts.append(prefix[:-start_cf])
        fade_in = np.linspace(0.0, 1.0, start_cf, dtype=np.float32)
        fade_out = 1.0 - fade_in
        blend_start = prefix[-start_cf:] * fade_out + ins[:start_cf] * fade_in
        parts.append(blend_start)
    else:
        parts.append(prefix)

    mid_start = start_cf
    mid_end = ins_len - end_cf
    if mid_end > mid_start:
        parts.append(ins[mid_start:mid_end])

    if end_cf > 0:
        fade_in = np.linspace(0.0, 1.0, end_cf, dtype=np.float32)
        fade_out = 1.0 - fade_in
        blend_end = ins[-end_cf:] * fade_out + suffix[:end_cf] * fade_in
        parts.append(blend_end)
        parts.append(suffix[end_cf:])
    else:
        parts.append(suffix)

    return np.concatenate(parts, axis=0).astype(np.float32)


def load_manifest(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list: {json_path}")
    return data


def normalize_dataset_tail(raw_path: str) -> Optional[Path]:
    norm = raw_path.replace("\\", "/")
    lower = norm.lower()
    marker = "/datasets/"
    idx = lower.find(marker)
    if idx >= 0:
        return Path(norm[idx + 1 :])
    if lower.startswith("datasets/"):
        return Path(norm)
    return None


def resolve_audio_path(raw_path: str, project_root: Path, json_path: Path) -> Optional[Path]:
    raw_path = str(raw_path).strip()
    if not raw_path:
        return None

    raw = Path(raw_path)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(project_root / raw)
        candidates.append(json_path.parent / raw)

    norm_rel = Path(raw_path.replace("\\", "/").lstrip("./"))
    candidates.append(project_root / norm_rel)
    candidates.append(json_path.parent / norm_rel)

    ds_tail = normalize_dataset_tail(raw_path)
    if ds_tail is not None:
        candidates.append(project_root / ds_tail)

    seen = set()
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.exists():
                return c
        except OSError:
            continue
    return None


def format_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def segment_windows(record_segments: Any, total_sec: float, min_sec: float) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    if not isinstance(record_segments, list):
        return windows

    for seg in record_segments:
        if not isinstance(seg, dict):
            continue
        try:
            start = max(0.0, float(seg.get("start", 0.0)))
            end = min(total_sec, float(seg.get("end", 0.0)))
        except (TypeError, ValueError):
            continue
        if end - start >= min_sec:
            windows.append((start, end))
    return windows


def choose_voiced_from_segments(
    source: np.ndarray,
    sr: int,
    segments: Any,
    clip_sec: float,
    rng: random.Random,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    total_sec = source.size / sr
    windows = segment_windows(segments, total_sec, clip_sec)
    if not windows:
        return None, None, None

    for _ in range(20):
        w0, w1 = windows[rng.randrange(len(windows))]
        max_start = w1 - clip_sec
        if max_start < w0:
            continue
        start_sec = rng.uniform(w0, max_start) if max_start > w0 else w0
        end_sec = start_sec + clip_sec
        s0 = to_samples(start_sec, sr, source.size)
        s1 = to_samples(end_sec, sr, source.size)
        if s1 <= s0:
            continue
        clip = source[s0:s1].copy()
        if rms(clip) > 1e-5:
            return clip, s0 / sr, s1 / sr
    return None, None, None


def choose_voiced_by_energy(
    source: np.ndarray,
    sr: int,
    clip_sec: float,
    rng: random.Random,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    need = int(round(clip_sec * sr))
    if need <= 0:
        return None, None, None
    if source.size < need:
        return None, None, None

    frame_len = max(1, int(round(25.0 * sr / 1000.0)))
    hop_len = max(1, int(round(10.0 * sr / 1000.0)))
    _, rms_values = frame_rms(source, frame_len, hop_len)
    if rms_values.size == 0:
        return None, None, None

    threshold = max(
        float(np.quantile(rms_values, 0.65)),
        float(np.mean(rms_values) * 1.1),
        1e-5,
    )
    hot = np.flatnonzero(rms_values >= threshold).tolist()
    if not hot:
        hot = [int(np.argmax(rms_values))]
    rng.shuffle(hot)

    for idx in hot[:80]:
        center = idx * hop_len + frame_len // 2
        s0 = center - need // 2
        s0 = max(0, min(s0, source.size - need))
        s1 = s0 + need
        clip = source[s0:s1].copy()
        if clip.size == need and rms(clip) >= max(1e-5, threshold * 0.45):
            return clip, s0 / sr, s1 / sr

    return None, None, None


def choose_insert_segment(
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

        src_path = resolve_audio_path(src_raw_path, project_root, source_json_path)
        if src_path is None:
            continue

        try:
            source_wav, source_sr = load_audio_mono(src_path)
        except (RuntimeError, OSError, ValueError):
            continue

        if source_wav.size == 0:
            continue

        if source_sr != target_sr:
            source_wav = resample_audio(source_wav, source_sr, target_sr)
            source_sr = target_sr

        clip, start_sec, end_sec = choose_voiced_from_segments(
            source=source_wav,
            sr=source_sr,
            segments=src_rec.get("segments", []),
            clip_sec=clip_sec,
            rng=rng,
        )
        method = "segments"
        if clip is None:
            clip, start_sec, end_sec = choose_voiced_by_energy(
                source=source_wav,
                sr=source_sr,
                clip_sec=clip_sec,
                rng=rng,
            )
            method = "energy"

        if clip is None or clip.size == 0:
            continue

        return {
            "clip": clip.astype(np.float32),
            "source_record": src_rec,
            "source_path": src_path,
            "source_start_sec": start_sec,
            "source_end_sec": end_sec,
            "pick_method": method,
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

    source_records = load_manifest(source_json_path)
    target_records = load_manifest(target_json_path)

    pair_dir = output_root / name
    audio_dir = pair_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_json = pair_dir / "mixed_manifest.json"

    mixed_records: List[Dict[str, Any]] = []
    stats = {
        "task": name,
        "source_json": format_path(source_json_path, project_root),
        "target_json": format_path(target_json_path, project_root),
        "target_total": len(target_records),
        "processed": 0,
        "saved": 0,
        "skip_short_target": 0,
        "skip_no_target_audio": 0,
        "skip_no_source_clip": 0,
        "errors": 0,
        "output_json": format_path(out_json, project_root),
    }

    limit = len(target_records) if max_items_per_pair is None else min(len(target_records), max_items_per_pair)
    for i in range(limit):
        stats["processed"] += 1
        tgt_rec = target_records[i]
        tgt_raw_path = tgt_rec.get("path")
        if not isinstance(tgt_raw_path, str):
            stats["skip_no_target_audio"] += 1
            continue

        tgt_path = resolve_audio_path(tgt_raw_path, project_root, target_json_path)
        if tgt_path is None:
            stats["skip_no_target_audio"] += 1
            continue

        try:
            target_wav, target_sr = load_audio_mono(tgt_path)
        except (RuntimeError, OSError, ValueError):
            stats["skip_no_target_audio"] += 1
            continue

        if target_wav.size == 0:
            stats["skip_no_target_audio"] += 1
            continue

        target_sec = target_wav.size / target_sr
        max_allowed = min(insert_max_sec, target_sec * 0.5)
        if max_allowed < insert_min_sec:
            stats["skip_short_target"] += 1
            continue

        insert_sec = rng.uniform(insert_min_sec, max_allowed)
        source_pick = choose_insert_segment(
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
            min_insert_samples = max(1, int(math.ceil(insert_min_sec * target_sr)))
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

            insert_seg = match_rms(insert_seg, rms(local_ref), max_gain_db=max_gain_db)

            if noise_mix > 0.0:
                noise_track = extract_noise_track(
                    target=target_wav,
                    sr=target_sr,
                    center_idx=insert_idx,
                    need_len=insert_seg.size,
                    noise_window_sec=noise_window_sec,
                )
                insert_seg = insert_seg + noise_mix * noise_track

            crossfade_samples = max(0, int(round(crossfade_ms * target_sr / 1000.0)))
            mixed = insert_with_crossfade(
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
                    "path": format_path(out_wav, project_root),
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

    pair_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(mixed_records, f, ensure_ascii=False, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch cross-language insertion with random 0.2~0.5s voiced clips and crossfade. "
            "Runs 4 fixed tasks by default."
        )
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/crossfade_insertions"))
    parser.add_argument("--insert-min-sec", type=float, default=0.2)
    parser.add_argument("--insert-max-sec", type=float, default=0.5)
    parser.add_argument("--crossfade-ms", type=float, default=80.0)
    parser.add_argument("--noise-mix", type=float, default=0.0)
    parser.add_argument("--noise-window-sec", type=float, default=2.0)
    parser.add_argument("--max-gain-db", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=519)
    parser.add_argument("--max-items-per-pair", type=int, default=None)
    parser.add_argument("--max-source-tries", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.insert_min_sec <= 0 or args.insert_max_sec <= 0:
        raise ValueError("insert-min-sec and insert-max-sec must be positive.")
    if args.insert_min_sec > args.insert_max_sec:
        raise ValueError("insert-min-sec must be <= insert-max-sec.")

    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root if args.output_root.is_absolute() else (project_root / args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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
    print(f"[Done] Summary saved to {format_path(summary_path, project_root)}")


if __name__ == "__main__":
    main()
