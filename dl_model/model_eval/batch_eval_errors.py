"""
Batch evaluate the final_model TDNN predictor on CSV label files.

This script follows the old `dl_model/old/speechbrain_ablation` style:
- Read CSV rows and resolve audio paths
- Load full 2-second audio and split into left/right halves
- Predict with `TDNNPredictor`
- Copy wrong 2-second audio files into `dl_model/model_eval/error/`

No error CSV summary is created.
"""

import csv
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.common import load_audio
from dl_model.final_model.model import TDNNPredictor
from dl_model.old.speechbrain_ablation.shared import parse_bool_label, split_pair_from_full_clip


def resolve_audio_path(row, root: Path, test_dir: Path):
    audio_path = str(row.get("audio_path", "") or "").strip()
    if audio_path:
        candidate = Path(audio_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
        if test_dir is not None:
            fallback = test_dir / candidate.name
            if fallback.exists():
                return fallback

    if test_dir is not None:
        index_keys = [row.get("test_row_index"), row.get("csv_index"), row.get("id"), row.get("index")]
        for key in index_keys:
            if key is None:
                continue
            try:
                idx = int(str(key).strip())
            except ValueError:
                continue
            candidate = test_dir / f"{idx}.wav"
            if candidate.exists():
                return candidate

    return None


def build_error_filename(csv_name: str, sample_index: int, audio_path: Path):
    return f"{csv_name}_{sample_index:05d}_{audio_path.name}"


def evaluate_single_csv(csv_path: Path, test_dir: Path, error_root: Path, predictor, threshold: float, sr: int):
    print(f"\nEvaluating CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    test_rows = [(i + 1, row) for i, row in enumerate(all_rows) if row.get("split", "").strip().lower() == "test"]

    print(f"  Test rows: {len(test_rows)}")

    all_scores = []
    all_preds = []
    all_labels = []
    errors = 0

    csv_name = csv_path.stem

    for absolute_index, row in test_rows:
        audio_file = test_dir / f"{absolute_index}.wav"
        if not audio_file.exists():
            print(f"Warning: audio file not found: {audio_file}")
            continue

        wav_full, _ = load_audio(audio_file, sr=sr)
        left_audio, right_audio = split_pair_from_full_clip(wav_full)

        is_switch_pred, confidence = predictor.predict(left_audio, right_audio)
        score_float = float(confidence)
        pred_label = 1 if score_float > threshold else 0

        is_switch_val = row.get("is_switch")
        if is_switch_val is None:
            raise ValueError(f"Missing is_switch for row {absolute_index} in {csv_path}")
        true_label = parse_bool_label(is_switch_val)

        all_preds.append(pred_label)
        all_labels.append(true_label)
        all_scores.append(score_float)

        if pred_label != true_label:
            dest_name = f"{csv_name}_{absolute_index:05d}.wav"
            dest_path = error_root / dest_name
            shutil.copy2(audio_file, dest_path)
            errors += 1

    if all_labels:
        total = len(all_labels)
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        accuracy = correct / total
        print(f"  Total evaluated: {total}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
        print(f"  Errors copied: {errors} into {error_root}")
    else:
        print("  No evaluated rows found.")

    return errors


def main():
    repo_root = Path(__file__).resolve().parents[2]
    error_root = repo_root / "dl_model/model_eval/error"
    error_root.mkdir(parents=True, exist_ok=True)

    # Hardcoded paths
    csvs = [
        ("dl_model/baseline_train_test_segments.csv", "datasets/mlp_train/test"),
        ("dl_model/baseline_train_test_segments_switchlingua_seame.csv", "datasets/baseline_switchlingua_seame_testset/test"),
    ]
    weight_path = "dl_model/final_model/tdnn_full_best_acc.pth"
    threshold = 0.0
    sr = 16000

    print(f"repo_root: {repo_root}")
    print(f"error_root: {error_root}")
    print(f"weight: {weight_path}")

    predictor = TDNNPredictor(device="cpu", weight_path=str(repo_root / weight_path))

    total_errors = 0
    for csv_arg, test_dir_arg in csvs:
        csv_path = repo_root / csv_arg
        test_dir = repo_root / test_dir_arg
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        total_errors += evaluate_single_csv(csv_path, test_dir, error_root, predictor, threshold, sr)

    print(f"\nBatch complete. Total error files copied: {total_errors}")


if __name__ == "__main__":
    main()
