from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from baseline.common import compute_metrics, load_eval_samples, preload_segment_pairs
from baseline.pyannote_wespeaker_voxceleb_resnet34_lm import (
    PyannoteWeSpeakerVoxCelebResnet34LMBaseline,
)
from baseline.pure_sincnet import PureSincNetBaseline
# from baseline.microsoft_wavlm_base_plus_sv import MicrosoftWavLMBasePlusSVBaseline
from baseline.distilled_mel_tdnn import DistilledMelTDNNBaseline
# from baseline.resemblyzer_ge2e import ResemblyzerGE2EBaseline
# from baseline.speechbrain_ecapa import SpeechBrainECAPABaseline
# from baseline.speechbrain_xvector import SpeechBrainXVectorBaseline
# from baseline.wespeaker_english import WeSpeakerEnglishBaseline


MODEL_REGISTRY = {
    "pure_sincnet": PureSincNetBaseline,
    # "pyannote_wespeaker_voxceleb_resnet34_lm": PyannoteWeSpeakerVoxCelebResnet34LMBaseline,
    # "microsoft_wavlm_base_plus_sv": MicrosoftWavLMBasePlusSVBaseline,
    # "speechbrain_ecapa": SpeechBrainECAPABaseline,
    # "speechbrain_xvector": SpeechBrainXVectorBaseline,
    # "resemblyzer_ge2e": ResemblyzerGE2EBaseline,
    # "wespeaker_english": WeSpeakerEnglishBaseline,
    "distilled_mel_tdnn": DistilledMelTDNNBaseline,
    # "project_mlp_whisper": ProjectMLPWhisperBaseline,
}


def select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_predictions_csv(output_path: Path, prediction_rows):
    fieldnames = [
        "test_row_index",
        "audio_path",
        "label",
        "prediction",
        "correct",
        "same_speaker_score",
        "raw_score",
        "runtime_ms",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)


def normalize_summary_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_summary_csv(output_path: Path, summary_rows):
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: normalize_summary_value(value) for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate selected same-speaker baselines on test_baseline_segments.csv."
    )
    parser.add_argument(
        "--csv",
        default="dl_model/baseline_train_test_segments_switchlingua_seame.csv",
        help="CSV containing segment windows and labels",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline/results",
        help="Where to write summary and per-model predictions",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which baselines to run",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of available rows to evaluate",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Run inference on the selected device when the model implementation supports it",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for models that require authenticated access",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / args.csv
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_start = time.perf_counter()
    samples, dataset_report = load_eval_samples(csv_path, root=repo_root, max_samples=args.max_samples)
    pairs = preload_segment_pairs(samples, target_sr=16000)
    dataset_time_s = time.perf_counter() - dataset_start

    print(f"csv rows             : {dataset_report.csv_rows}")
    print(f"available rows       : {dataset_report.available_rows}")
    print(f"missing rows         : {dataset_report.missing_rows}")
    print(f"positives / negatives: {dataset_report.positives} / {dataset_report.negatives}")
    print(f"dataset prep time(s) : {dataset_time_s:.4f}")

    if dataset_report.missing_examples:
        print("missing examples     :")
        for path in dataset_report.missing_examples[:5]:
            print(f"  - {path}")

    if len(pairs) == 0:
        raise RuntimeError("No available audio rows were found for evaluation.")

    if dataset_report.positives == 0 or dataset_report.negatives == 0:
        print("warning              : only one class is available locally, so accuracy is not representative.")

    device = select_device(args.device)
    print(f"device               : {device}")

    summary_rows = []
    for model_name in args.models:
        model_class = MODEL_REGISTRY[model_name]
        print(f"\n=== Running {model_name} ===")

        init_start = time.perf_counter()
        init_kwargs = {
            "device": device,
            "cache_dir": cache_dir,
        }
        if model_name == "pyannote_wespeaker_voxceleb_resnet34_lm":
            init_kwargs["hf_token"] = args.hf_token

        try:
            model = model_class(**init_kwargs)
        except Exception as exc:
            init_time_s = time.perf_counter() - init_start
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"failed to initialize  : {error_message}")
            summary_rows.append(
                {
                    "model": model_name,
                    "sample_count": 0,
                    "positives": 0,
                    "negatives": 0,
                    "accuracy": None,
                    "positive_accuracy": None,
                    "negative_accuracy": None,
                    "precision": None,
                    "recall": None,
                    "specificity": None,
                    "f1": None,
                    "balanced_accuracy": None,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "init_time_s": init_time_s,
                    "inference_time_s": 0.0,
                    "total_model_time_s": init_time_s,
                    "avg_inference_ms": None,
                    "dataset_available_rows": dataset_report.available_rows,
                    "dataset_missing_rows": dataset_report.missing_rows,
                    "error": error_message,
                }
            )
            continue

        init_time_s = time.perf_counter() - init_start

        labels = []
        predictions = []
        prediction_rows = []
        inference_time_s = 0.0

        for pair in pairs:
            step_start = time.perf_counter()
            result = model.predict(pair.left_audio, pair.right_audio, pair.sample_rate)
            step_time_s = time.perf_counter() - step_start
            inference_time_s += step_time_s

            label = pair.sample.label
            prediction = int(result.prediction)
            labels.append(label)
            predictions.append(prediction)
            prediction_rows.append(
                {
                    "test_row_index": pair.sample.test_row_index,
                    "audio_path": pair.sample.audio_path,
                    "label": label,
                    "prediction": prediction,
                    "correct": int(label == prediction),
                    "same_speaker_score": f"{result.same_speaker_score:.6f}",
                    "raw_score": f"{result.raw_score:.6f}",
                    "runtime_ms": f"{step_time_s * 1000.0:.4f}",
                }
            )

        metrics = compute_metrics(labels, predictions)
        total_model_time_s = init_time_s + inference_time_s
        avg_inference_ms = inference_time_s * 1000.0 / len(pairs)

        summary = {
            "model": model_name,
            "sample_count": metrics["sample_count"],
            "positives": metrics["positives"],
            "negatives": metrics["negatives"],
            "accuracy": metrics["accuracy"],
            "positive_accuracy": metrics["positive_accuracy"],
            "negative_accuracy": metrics["negative_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "init_time_s": init_time_s,
            "inference_time_s": inference_time_s,
            "total_model_time_s": total_model_time_s,
            "avg_inference_ms": avg_inference_ms,
            "dataset_available_rows": dataset_report.available_rows,
            "dataset_missing_rows": dataset_report.missing_rows,
            "error": "",
        }
        summary_rows.append(summary)

        write_predictions_csv(output_dir / f"{model_name}_predictions.csv", prediction_rows)
        print(
            f"accuracy={summary['accuracy'] if summary['accuracy'] is not None else 'NA'} "
            f"avg_inference_ms={avg_inference_ms:.4f} total_time_s={total_model_time_s:.4f}"
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"\nsummary csv          : {(output_dir / 'summary.csv').resolve()}")
    print(f"summary json         : {(output_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
