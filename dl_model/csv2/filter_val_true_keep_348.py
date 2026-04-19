import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile


TRUE_TOKENS = {"1", "true", "t", "yes", "y"}
FALSE_TOKENS = {"0", "false", "f", "no", "n"}


@dataclass
class RowInfo:
    csv_position: int
    source_index: int
    is_switch: bool
    row: dict


@dataclass
class EdgeStats:
    head_rms_db: float
    tail_rms_db: float
    head_active_ratio: float
    tail_active_ratio: float

    @property
    def min_rms_db(self) -> float:
        return min(self.head_rms_db, self.tail_rms_db)

    @property
    def min_active_ratio(self) -> float:
        return min(self.head_active_ratio, self.tail_active_ratio)


def parse_bool_label(value: str) -> bool:
    text = str(value).strip().lower()
    if text in TRUE_TOKENS:
        return True
    if text in FALSE_TOKENS:
        return False
    raise ValueError(f"Unsupported is_switch value: {value}")


def as_float_mono(wav: np.ndarray) -> np.ndarray:
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        wav = wav.astype(np.float32) / float(max(abs(info.min), abs(info.max)))
    else:
        wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)


def rms_db(signal: np.ndarray) -> float:
    if signal.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))
    return 20.0 * np.log10(rms + 1e-12)


def frame_rms_db(region: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))

    if region.size < frame_len:
        return np.array([rms_db(region)], dtype=np.float32)

    vals = []
    for i in range(0, region.size - frame_len + 1, hop_len):
        vals.append(rms_db(region[i : i + frame_len]))
    return np.asarray(vals, dtype=np.float32)


def compute_edge_stats(
    wav: np.ndarray,
    sr: int,
    edge_sec: float,
    speech_db_threshold: float,
) -> EdgeStats:
    edge_samples = max(1, int(round(sr * edge_sec)))
    head = wav[:edge_samples]
    tail = wav[-edge_samples:]

    head_db = rms_db(head)
    tail_db = rms_db(tail)

    head_frames = frame_rms_db(head, sr)
    tail_frames = frame_rms_db(tail, sr)
    head_active = float(np.mean(head_frames > speech_db_threshold))
    tail_active = float(np.mean(tail_frames > speech_db_threshold))

    return EdgeStats(
        head_rms_db=head_db,
        tail_rms_db=tail_db,
        head_active_ratio=head_active,
        tail_active_ratio=tail_active,
    )


def quality_score(stats: EdgeStats) -> float:
    # Higher is better: require both sides energetic and active.
    return stats.min_rms_db + 20.0 * stats.min_active_ratio


def passes_gate(stats: EdgeStats, min_edge_rms_db: float, min_active_ratio: float) -> bool:
    return stats.min_rms_db >= min_edge_rms_db and stats.min_active_ratio >= min_active_ratio


def load_rows(input_csv: Path):
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for pos, row in enumerate(reader, start=1):
            source_index = int(row["test_row_index"]) if row.get("test_row_index") else pos
            rows.append(
                RowInfo(
                    csv_position=pos,
                    source_index=source_index,
                    is_switch=parse_bool_label(row["is_switch"]),
                    row=dict(row),
                )
            )
    return fieldnames, rows


def main():
    parser = argparse.ArgumentParser(
        description="Filter val set: keep all False and keep top-K True by edge speech quality."
    )
    parser.add_argument(
        "--input-csv",
        default="datasets/train_test2/val_segments_audio_package_500.csv",
    )
    parser.add_argument(
        "--input-audio-dir",
        default="datasets/train_test2/val",
    )
    parser.add_argument(
        "--output-csv",
        default="datasets/train_test2/val_segments_audio_package_500_balanced.csv",
    )
    parser.add_argument(
        "--output-audio-dir",
        default="datasets/train_test2/val_balanced",
    )
    parser.add_argument("--target-true", type=int, default=348)
    parser.add_argument("--edge-sec", type=float, default=1.0)
    parser.add_argument("--speech-db-threshold", type=float, default=-45.0)
    parser.add_argument("--min-edge-rms-db", type=float, default=-42.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.08)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    input_audio_dir = Path(args.input_audio_dir)
    output_csv = Path(args.output_csv)
    output_audio_dir = Path(args.output_audio_dir)

    fieldnames, rows = load_rows(input_csv)
    true_rows = [r for r in rows if r.is_switch]
    false_rows = [r for r in rows if not r.is_switch]

    scored_true = []
    skipped_missing = 0
    for idx, info in enumerate(true_rows, start=1):
        wav_path = input_audio_dir / f"{info.source_index}.wav"
        if not wav_path.exists():
            skipped_missing += 1
            continue

        sr, wav = wavfile.read(str(wav_path))
        wav = as_float_mono(wav)
        stats = compute_edge_stats(
            wav=wav,
            sr=sr,
            edge_sec=args.edge_sec,
            speech_db_threshold=args.speech_db_threshold,
        )
        scored_true.append(
            {
                "info": info,
                "stats": stats,
                "quality": quality_score(stats),
                "pass_gate": passes_gate(
                    stats,
                    min_edge_rms_db=args.min_edge_rms_db,
                    min_active_ratio=args.min_active_ratio,
                ),
            }
        )
        if idx % 5000 == 0:
            print(f"Scored True clips: {idx}/{len(true_rows)}")

    pass_list = [x for x in scored_true if x["pass_gate"]]
    fail_list = [x for x in scored_true if not x["pass_gate"]]

    rank_key = lambda x: (
        -x["quality"],
        -x["stats"].min_active_ratio,
        -x["stats"].min_rms_db,
        x["info"].csv_position,
    )
    pass_list.sort(key=rank_key)
    fail_list.sort(key=rank_key)

    selected_true = pass_list[: args.target_true]
    if len(selected_true) < args.target_true:
        need = args.target_true - len(selected_true)
        selected_true.extend(fail_list[:need])

    selected_true_infos = [x["info"] for x in selected_true]
    selected_set = set((x.csv_position, x.source_index) for x in selected_true_infos)

    # Keep all False plus selected True, then preserve source order.
    selected_rows = [r for r in rows if (not r.is_switch) or ((r.csv_position, r.source_index) in selected_set)]
    selected_rows.sort(key=lambda x: x.csv_position)

    output_audio_dir.mkdir(parents=True, exist_ok=True)
    for p in output_audio_dir.glob("*.wav"):
        p.unlink()

    if "test_row_index" not in fieldnames:
        fieldnames = ["test_row_index"] + fieldnames
    if "is_switch" not in fieldnames:
        fieldnames.append("is_switch")
    if "split" not in fieldnames:
        fieldnames.append("split")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    copy_missing = 0
    for new_idx, info in enumerate(selected_rows, start=1):
        src = input_audio_dir / f"{info.source_index}.wav"
        dst = output_audio_dir / f"{new_idx}.wav"
        if not src.exists():
            copy_missing += 1
            continue
        shutil.copy2(src, dst)

        new_row = dict(info.row)
        new_row["test_row_index"] = str(new_idx)
        new_row["is_switch"] = "True" if info.is_switch else "False"
        new_row["split"] = "val"
        rows_out.append(new_row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    final_true = sum(1 for r in rows_out if parse_bool_label(r["is_switch"]))
    final_false = len(rows_out) - final_true

    print(f"Input rows: {len(rows)} (True={len(true_rows)}, False={len(false_rows)})")
    print(f"Scored True rows: {len(scored_true)} | gate-pass={len(pass_list)} | gate-fail={len(fail_list)}")
    print(f"Selected True rows: {len(selected_true_infos)}")
    print(f"Output rows: {len(rows_out)} (True={final_true}, False={final_false})")
    print(f"Skipped missing while scoring: {skipped_missing}")
    print(f"Missing while copying: {copy_missing}")
    print(f"Output CSV: {output_csv.as_posix()}")
    print(f"Output audio dir: {output_audio_dir.as_posix()}")


if __name__ == "__main__":
    main()
