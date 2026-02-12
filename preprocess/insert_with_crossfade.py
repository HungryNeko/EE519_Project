import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy import signal


EPS = 1e-9


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
        selected = frames[np.argmin(rms_values) : np.argmin(rms_values) + 1]

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

    parts = []

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert a segment from one wav file into another with crossfade and ambient-noise matching."
    )
    parser.add_argument("--target", type=Path, default=Path("speaker_zh.wav"), help="Target wav to receive insertion.")
    parser.add_argument(
        "--insert-source",
        type=Path,
        default=Path("speaker_en.wav"),
        help="Source wav that provides inserted segment.",
    )
    parser.add_argument("--out", type=Path, default=Path("preprocess/inserted_output.wav"), help="Output wav path.")
    parser.add_argument(
        "--insert-at-sec",
        type=float,
        default=None,
        help="Insert position in target, in seconds. Default: middle of target.",
    )
    parser.add_argument(
        "--insert-start-sec",
        type=float,
        default=0.0,
        help="Start time of segment in insert-source (seconds).",
    )
    parser.add_argument(
        "--insert-end-sec",
        type=float,
        default=None,
        help="End time of segment in insert-source (seconds). Default: to source end.",
    )
    parser.add_argument("--crossfade-ms", type=float, default=80.0, help="Crossfade duration in milliseconds.")
    parser.add_argument(
        "--noise-window-sec",
        type=float,
        default=2.0,
        help="Context window around insertion point for noise-profile extraction.",
    )
    parser.add_argument(
        "--noise-mix",
        type=float,
        default=0.18,
        help="Amount of extracted ambient noise mixed into inserted segment (0~1).",
    )
    args = parser.parse_args()

    target, sr_t = load_audio_mono(args.target)
    source, sr_s = load_audio_mono(args.insert_source)

    if sr_s != sr_t:
        source = resample_audio(source, sr_s, sr_t)

    src_len = source.size
    s0 = to_samples(args.insert_start_sec, sr_t, src_len)
    s1 = src_len if args.insert_end_sec is None else to_samples(args.insert_end_sec, sr_t, src_len)
    if s1 <= s0:
        raise ValueError("Invalid insert segment: insert-end-sec must be greater than insert-start-sec.")
    insert_seg = source[s0:s1].copy()

    tgt_len = target.size
    if args.insert_at_sec is None:
        insert_idx = tgt_len // 2
    else:
        insert_idx = to_samples(args.insert_at_sec, sr_t, tgt_len)

    local_ref_len = min(insert_seg.size, tgt_len)
    ref_start = max(0, insert_idx - local_ref_len // 2)
    ref_end = min(tgt_len, ref_start + local_ref_len)
    local_ref = target[ref_start:ref_end]
    if local_ref.size == 0:
        local_ref = target

    insert_seg = match_rms(insert_seg, rms(local_ref))

    noise_mix = float(np.clip(args.noise_mix, 0.0, 1.0))
    if noise_mix > 0.0:
        noise_track = extract_noise_track(
            target=target,
            sr=sr_t,
            center_idx=insert_idx,
            need_len=insert_seg.size,
            noise_window_sec=args.noise_window_sec,
        )
        insert_seg = insert_seg + noise_mix * noise_track

    crossfade_samples = max(0, int(round(args.crossfade_ms * sr_t / 1000.0)))
    mixed = insert_with_crossfade(
        target=target,
        insert_seg=insert_seg,
        insert_idx=insert_idx,
        crossfade_samples=crossfade_samples,
    )
    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), mixed, sr_t)

    print(f"Saved: {args.out}")
    print(f"Sample rate: {sr_t}")
    print(f"Target duration: {target.size / sr_t:.3f} sec")
    print(f"Inserted segment duration: {insert_seg.size / sr_t:.3f} sec")
    print(f"Output duration: {mixed.size / sr_t:.3f} sec")
    print(f"Insert point in target: {insert_idx / sr_t:.3f} sec")
    print(f"Crossfade: {crossfade_samples / sr_t:.3f} sec")


if __name__ == "__main__":
    main()
