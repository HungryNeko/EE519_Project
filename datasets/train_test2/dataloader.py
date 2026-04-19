from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.dataloader import (
    DistillationPairDataset,
    collate_audio_pairs,
    extract_time_window,
    split_pair_from_full_clip,
)
from baseline.common import load_audio


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_CSV = DEFAULT_DATASET_DIR / "compare_train_val_test_manifest.csv"


def parse_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return 1
    if text in {"0", "false", "f", "no"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def _to_int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(text)


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def build_samples_from_manifest(
    split: Optional[str],
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    dataset_root: Path = DEFAULT_DATASET_DIR,
    target_sr: int = 16000,
    include_time_windows: bool = False,
) -> List[dict]:
    """Load samples from one unified manifest csv.

    Args:
        split: one of {"train", "val", "test"}; None loads all rows.
        manifest_csv: unified csv path.
        dataset_root: root used to resolve `audio_rel_path`.
        target_sr: audio sample rate for loading.
        include_time_windows: whether to attach left/right/switch timestamps
            when those columns exist.
    """
    split_key = None if split is None else str(split).strip().lower()
    if split_key not in {None, "train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    samples: List[dict] = []
    with open(manifest_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = str(row.get("compare_split", "")).strip().lower()
            if split_key is not None and row_split != split_key:
                continue

            audio_rel = row.get("audio_rel_path", "").strip()
            if audio_rel == "":
                continue

            audio_file = (dataset_root / audio_rel).resolve()
            if not audio_file.exists():
                continue

            sample = {
                "audio_path": row.get("source_audio_path", row.get("audio_rel_path", "")),
                "audio_file": str(audio_file),
                "label": parse_label(row.get("label", "")),
                "target_sr": int(target_sr),
                "compare_split": row_split,
                "audio_rel_path": audio_rel,
            }

            source_row_index = _to_int_or_none(row.get("source_row_index"))
            if source_row_index is not None:
                sample["source_row_index"] = source_row_index

            test_row_index = _to_int_or_none(row.get("test_row_index"))
            if test_row_index is not None:
                sample["test_row_index"] = test_row_index

            if include_time_windows:
                left_start = _to_float_or_none(row.get("left_start"))
                left_end = _to_float_or_none(row.get("left_end"))
                switch_time = _to_float_or_none(row.get("switch_time"))
                right_start = _to_float_or_none(row.get("right_start"))
                right_end = _to_float_or_none(row.get("right_end"))

                if left_start is not None and left_end is not None:
                    sample["left_start"] = left_start
                    sample["left_end"] = left_end
                if right_start is not None and right_end is not None:
                    sample["right_start"] = right_start
                    sample["right_end"] = right_end
                if switch_time is not None:
                    sample["switch_time"] = switch_time

            samples.append(sample)

    return samples


def build_train_samples(
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    dataset_root: Path = DEFAULT_DATASET_DIR,
    target_sr: int = 16000,
    include_time_windows: bool = False,
) -> List[dict]:
    return build_samples_from_manifest(
        "train",
        manifest_csv=manifest_csv,
        dataset_root=dataset_root,
        target_sr=target_sr,
        include_time_windows=include_time_windows,
    )


def build_val_samples(
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    dataset_root: Path = DEFAULT_DATASET_DIR,
    target_sr: int = 16000,
    include_time_windows: bool = False,
) -> List[dict]:
    return build_samples_from_manifest(
        "val",
        manifest_csv=manifest_csv,
        dataset_root=dataset_root,
        target_sr=target_sr,
        include_time_windows=include_time_windows,
    )


def build_test_samples(
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    dataset_root: Path = DEFAULT_DATASET_DIR,
    target_sr: int = 16000,
    include_time_windows: bool = False,
) -> List[dict]:
    return build_samples_from_manifest(
        "test",
        manifest_csv=manifest_csv,
        dataset_root=dataset_root,
        target_sr=target_sr,
        include_time_windows=include_time_windows,
    )


def build_train_val_test_samples(
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    dataset_root: Path = DEFAULT_DATASET_DIR,
    target_sr: int = 16000,
    include_time_windows: bool = False,
) -> Dict[str, List[dict]]:
    return {
        "train": build_train_samples(
            manifest_csv=manifest_csv,
            dataset_root=dataset_root,
            target_sr=target_sr,
            include_time_windows=include_time_windows,
        ),
        "val": build_val_samples(
            manifest_csv=manifest_csv,
            dataset_root=dataset_root,
            target_sr=target_sr,
            include_time_windows=include_time_windows,
        ),
        "test": build_test_samples(
            manifest_csv=manifest_csv,
            dataset_root=dataset_root,
            target_sr=target_sr,
            include_time_windows=include_time_windows,
        ),
    }


def set_half_duration(samples: Iterable[dict], half_duration: float) -> None:
    for sample in samples:
        sample["half_duration"] = float(half_duration)


def assign_random_half_durations(
    samples: Iterable[dict],
    min_half_duration: float = 1.0,
    max_half_duration: float = 2.0,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    for sample in samples:
        sample["half_duration"] = rng.uniform(min_half_duration, max_half_duration)
        sample.pop("left_audio", None)
        sample.pop("right_audio", None)


def assign_teacher_prob_from_labels(samples: Iterable[dict]) -> None:
    for sample in samples:
        sample["teacher_prob"] = float(sample["label"])


def preload_audio_pairs(samples: List[dict], limit: Optional[int] = None) -> None:
    count = len(samples) if limit is None else min(limit, len(samples))
    for sample in samples[:count]:
        if "left_audio" in sample and "right_audio" in sample:
            continue

        wav, _ = load_audio(Path(sample["audio_file"]), sr=int(sample.get("target_sr", 16000)))
        sr = int(sample.get("target_sr", 16000))

        if "switch_time" in sample and "half_duration" in sample:
            switch_time = float(sample["switch_time"])
            half_duration = float(sample["half_duration"])
            left = extract_time_window(wav, sr, switch_time - half_duration, switch_time)
            right = extract_time_window(wav, sr, switch_time, switch_time + half_duration)
        elif all(key in sample for key in ("left_start", "left_end", "right_start", "right_end")):
            left = extract_time_window(wav, sr, float(sample["left_start"]), float(sample["left_end"]))
            right = extract_time_window(wav, sr, float(sample["right_start"]), float(sample["right_end"]))
        else:
            half_duration = float(sample.get("half_duration", 4.0))
            left, right = split_pair_from_full_clip(wav, half_duration, sr)

        sample["left_audio"] = left
        sample["right_audio"] = right


def create_loader(
    samples: List[dict],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DistillationPairDataset(samples)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_audio_pairs,
    )


def build_loaders_from_manifest(
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    dataset_root: Path = DEFAULT_DATASET_DIR,
    target_sr: int = 16000,
    batch_size: int = 64,
    num_workers: int = 0,
    include_time_windows: bool = False,
) -> Dict[str, DataLoader]:
    splits = build_train_val_test_samples(
        manifest_csv=manifest_csv,
        dataset_root=dataset_root,
        target_sr=target_sr,
        include_time_windows=include_time_windows,
    )

    for split_name in ("train", "val", "test"):
        if len(splits[split_name]) == 0:
            raise RuntimeError(f"No samples loaded for split: {split_name}")

    return {
        "train": create_loader(
            splits["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": create_loader(
            splits["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": create_loader(
            splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


if __name__ == "__main__":
    splits = build_train_val_test_samples()
    for name, samples in splits.items():
        print(f"{name}: {len(samples)}")
