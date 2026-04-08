import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from baseline.common import AudioPathResolver, load_audio


def split_pair_from_full_clip(wav, half_duration_seconds, sr):
    half_samples = int(half_duration_seconds * sr)
    mid = len(wav) // 2
    left_start = max(0, mid - half_samples)
    left_end = mid
    right_start = mid
    right_end = min(len(wav), mid + half_samples)
    left = wav[left_start:left_end].astype("float32")
    right = wav[right_start:right_end].astype("float32")
    return left, right


def extract_time_window(wav, sr, start_time, end_time):
    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    length = max(1, end_i - start_i)
    out = wav.new_zeros(length) if hasattr(wav, "new_zeros") else None
    if out is None:
        import numpy as np

        out = np.zeros(length, dtype="float32")

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)
    if src_end <= src_start:
        return out

    dst_start = src_start - start_i
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = wav[src_start:src_end]
    return out.astype("float32")


def parse_bool_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return 1
    if text in {"0", "false", "f", "no"}:
        return 0
    raise ValueError(f"Unsupported is_switch value: {value}")


def build_samples_from_old_all(csv_path: Path, train_audio_dir: Path, test_audio_dir: Path, target_sr=16000):
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            split = row["split"].strip().lower()
            audio_dir = train_audio_dir if split == "train" else test_audio_dir
            audio_file = audio_dir / f"{idx + 1}.wav"
            if not audio_file.exists():
                continue
            samples.append(
                {
                    "audio_path": row["audio_path"],
                    "source_index": idx + 1,
                    "source_split": split,
                    "audio_file": str(audio_file),
                    "label": parse_bool_label(row["is_switch"]),
                    "target_sr": target_sr,
                }
            )
    return samples


def build_samples_from_new_extracted(csv_path: Path, test_audio_dir: Path, target_sr=16000, split="test"):
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row_split = row.get("split", "").strip().lower()
            if split and row_split and row_split != split:
                continue
            file_index = int(row["test_row_index"]) if row.get("test_row_index") else idx + 1
            audio_file = test_audio_dir / f"{file_index}.wav"
            if not audio_file.exists():
                continue
            samples.append(
                {
                    "audio_path": row["audio_path"],
                    "test_row_index": file_index,
                    "audio_file": str(audio_file),
                    "label": parse_bool_label(row["is_switch"]),
                    "target_sr": target_sr,
                }
            )
    return samples


def build_samples_from_timestamp_csv(csv_path: Path, root: Path, target_sr=16000, split="test"):
    resolver = AudioPathResolver(root)
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row_split = row.get("split", "").strip().lower()
            if split and row_split and row_split != split:
                continue
            resolved = resolver.resolve(row["audio_path"], row.get("audio_abs_path"))
            if resolved is None:
                continue
            samples.append(
                {
                    "audio_path": row["audio_path"],
                    "test_row_index": int(row["test_row_index"]) if row.get("test_row_index") else idx + 1,
                    "audio_file": str(resolved),
                    "label": parse_bool_label(row["is_switch"]),
                    "target_sr": target_sr,
                    "left_start": float(row["left_start"]),
                    "left_end": float(row["left_end"]),
                    "switch_time": float(row["switch_time"]),
                    "right_start": float(row["right_start"]),
                    "right_end": float(row["right_end"]),
                }
            )
    return samples


class DistillationPairDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        if "left_audio" not in sample or "right_audio" not in sample:
            wav, _ = load_audio(Path(sample["audio_file"]), sr=int(sample.get("target_sr", 16000)))
            sr = sample.get("target_sr", 16000)
            if "switch_time" in sample and "half_duration" in sample:
                switch_time = float(sample["switch_time"])
                half_duration = float(sample["half_duration"])
                left = extract_time_window(wav, sr, switch_time - half_duration, switch_time)
                right = extract_time_window(wav, sr, switch_time, switch_time + half_duration)
            elif all(key in sample for key in ("left_start", "left_end", "right_start", "right_end")):
                left = extract_time_window(wav, sr, float(sample["left_start"]), float(sample["left_end"]))
                right = extract_time_window(wav, sr, float(sample["right_start"]), float(sample["right_end"]))
            else:
                half_duration = sample.get("half_duration", 4.0)
                left, right = split_pair_from_full_clip(wav, half_duration, sr)
            sample["left_audio"] = left
            sample["right_audio"] = right
        return sample


def collate_audio_pairs(batch):
    left_lengths = [len(item["left_audio"]) for item in batch]
    right_lengths = [len(item["right_audio"]) for item in batch]
    max_left = max(left_lengths)
    max_right = max(right_lengths)

    left_batch = torch.zeros(len(batch), max_left, dtype=torch.float32)
    right_batch = torch.zeros(len(batch), max_right, dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    teacher_probs = torch.tensor([item["teacher_prob"] for item in batch], dtype=torch.float32)

    for i, item in enumerate(batch):
        left = torch.from_numpy(item["left_audio"])
        right = torch.from_numpy(item["right_audio"])
        left_batch[i, : left.numel()] = left
        right_batch[i, : right.numel()] = right

    return {
        "left_audio": left_batch,
        "right_audio": right_batch,
        "labels": labels,
        "teacher_prob": teacher_probs,
    }


def preload_audio_pairs(samples, limit=None):
    count = len(samples) if limit is None else min(limit, len(samples))
    for sample in samples[:count]:
        if "left_audio" in sample and "right_audio" in sample:
            continue
        wav, _ = load_audio(Path(sample["audio_file"]), sr=int(sample.get("target_sr", 16000)))
        half_duration = sample.get("half_duration", 4.0)
        sr = sample.get("target_sr", 16000)
        left, right = split_pair_from_full_clip(wav, half_duration, sr)
        sample["left_audio"] = left
        sample["right_audio"] = right


def assign_random_half_durations(samples, min_half_duration=1.0, max_half_duration=2.0, seed=42):
    rng = random.Random(seed)
    for sample in samples:
        sample["half_duration"] = rng.uniform(min_half_duration, max_half_duration)
        sample.pop("left_audio", None)
        sample.pop("right_audio", None)
