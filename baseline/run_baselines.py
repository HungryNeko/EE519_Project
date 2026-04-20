from __future__ import annotations

import argparse
import csv
import gc
import importlib
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.common import compute_metrics
from datasets.train_test2.dataloader import (
    build_train_val_test_samples,
    preload_audio_pairs,
    set_half_duration,
)


# Keep only official pretrained models.
# SpeechBrain models are default and prioritized for stability.
MODEL_SPECS = {
    "speechbrain_ecapa": {
        "module": "baseline.speechbrain_ecapa",
        "class": "SpeechBrainECAPABaseline",
        "optional": False,
    },
    "speechbrain_xvector": {
        "module": "baseline.speechbrain_xvector",
        "class": "SpeechBrainXVectorBaseline",
        "optional": False,
    },
    "resemblyzer_ge2e": {
        "module": "baseline.resemblyzer_ge2e",
        "class": "ResemblyzerGE2EBaseline",
        "optional": False,
    },
    "microsoft_wavlm_base_plus_sv": {
        "module": "baseline.microsoft_wavlm_base_plus_sv",
        "class": "MicrosoftWavLMBasePlusSVBaseline",
        "optional": False,
    },
    "pyannote_wespeaker_voxceleb_resnet34_lm": {
        "module": "baseline.pyannote_wespeaker_voxceleb_resnet34_lm",
        "class": "PyannoteWeSpeakerVoxCelebResnet34LMBaseline",
        "optional": False,
    },
    "wespeaker_english": {
        "module": "baseline.wespeaker_english",
        "class": "WeSpeakerEnglishBaseline",
        "optional": False,
    },
}

DEFAULT_MODELS = [name for name, spec in MODEL_SPECS.items() if not spec["optional"]]
OPTIONAL_MODELS = [name for name, spec in MODEL_SPECS.items() if spec["optional"]]

OFFICIAL_SOURCE = {
    "speechbrain_ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "speechbrain_xvector": "speechbrain/spkrec-xvect-voxceleb",
    "resemblyzer_ge2e": "resemblyzer/GE2E",
    "microsoft_wavlm_base_plus_sv": "microsoft/wavlm-base-plus-sv",
    "pyannote_wespeaker_voxceleb_resnet34_lm": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "wespeaker_english": "wespeaker_nuaazs:english",
}

RUN_FIELDNAMES = [
    "model",
    "run",
    "select_metric",
    "best_epoch",
    "epochs_trained",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_acc_at_best",
    "train_f1_at_best",
    "train_precision_at_best",
    "train_recall_at_best",
    "train_loss_at_best",
    "train_err_at_best",
    "val_acc_at_best",
    "val_f1_at_best",
    "val_precision_at_best",
    "val_recall_at_best",
    "val_loss_at_best",
    "val_err_at_best",
    "test_acc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_loss",
    "test_err",
    "test_sample_count",
    "test_time_seconds",
    "train_time_seconds",
    "total_time_seconds",
    "student_ms",
    "best_checkpoint",
]

AGG_FIELDNAMES = [
    "model",
    "runs",
    "test_acc_mean",
    "test_acc_std",
    "test_f1_mean",
    "test_f1_std",
    "test_err_mean",
    "test_err_std",
    "test_loss_mean",
    "test_loss_std",
    "val_acc_mean",
    "val_acc_std",
    "val_f1_mean",
    "val_f1_std",
    "val_err_mean",
    "val_err_std",
    "val_loss_mean",
    "val_loss_std",
    "train_time_seconds_mean",
    "train_time_seconds_std",
    "test_time_seconds_mean",
    "test_time_seconds_std",
    "total_time_seconds_mean",
    "total_time_seconds_std",
    "student_ms_mean",
    "student_ms_std",
]


def import_model_class(model_name: str):
    spec = MODEL_SPECS[model_name]
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["class"])


def select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def error_rate(acc_value):
    if acc_value is None:
        return None
    return 1.0 - float(acc_value)


def mean_std(values):
    if not values:
        return None, None
    mean_val = sum(values) / len(values)
    var_val = sum((x - mean_val) ** 2 for x in values) / len(values)
    return mean_val, var_val**0.5


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def evaluate_split(model, samples, sample_rate: int, desc: str = ""):
    labels = []
    predictions = []
    start_t = time.perf_counter()
    iterator = tqdm(samples, desc=desc, leave=False) if desc else samples
    for sample in iterator:
        pred = model.predict(
            sample["left_audio"],
            sample["right_audio"],
            int(sample.get("target_sr", sample_rate)),
        )
        labels.append(int(sample["label"]))
        predictions.append(int(pred.prediction))
    elapsed = time.perf_counter() - start_t
    metrics = compute_metrics(labels, predictions)
    return metrics, elapsed


def aggregate_model_rows(model_name, rows):
    def collect(key):
        return [row[key] for row in rows if row.get(key) is not None]

    test_acc_mean, test_acc_std = mean_std(collect("test_acc"))
    test_f1_mean, test_f1_std = mean_std(collect("test_f1"))
    test_err_mean, test_err_std = mean_std(collect("test_err"))
    test_loss_mean, test_loss_std = mean_std(collect("test_loss"))
    val_acc_mean, val_acc_std = mean_std(collect("val_acc_at_best"))
    val_f1_mean, val_f1_std = mean_std(collect("val_f1_at_best"))
    val_err_mean, val_err_std = mean_std(collect("val_err_at_best"))
    val_loss_mean, val_loss_std = mean_std(collect("val_loss_at_best"))
    train_time_mean, train_time_std = mean_std(collect("train_time_seconds"))
    test_time_mean, test_time_std = mean_std(collect("test_time_seconds"))
    total_time_mean, total_time_std = mean_std(collect("total_time_seconds"))
    student_ms_mean, student_ms_std = mean_std(collect("student_ms"))

    return {
        "model": model_name,
        "runs": len(rows),
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
        "test_f1_mean": test_f1_mean,
        "test_f1_std": test_f1_std,
        "test_err_mean": test_err_mean,
        "test_err_std": test_err_std,
        "test_loss_mean": test_loss_mean,
        "test_loss_std": test_loss_std,
        "val_acc_mean": val_acc_mean,
        "val_acc_std": val_acc_std,
        "val_f1_mean": val_f1_mean,
        "val_f1_std": val_f1_std,
        "val_err_mean": val_err_mean,
        "val_err_std": val_err_std,
        "val_loss_mean": val_loss_mean,
        "val_loss_std": val_loss_std,
        "train_time_seconds_mean": train_time_mean,
        "train_time_seconds_std": train_time_std,
        "test_time_seconds_mean": test_time_mean,
        "test_time_seconds_std": test_time_std,
        "total_time_seconds_mean": total_time_mean,
        "total_time_seconds_std": total_time_std,
        "student_ms_mean": student_ms_mean,
        "student_ms_std": student_ms_std,
    }


def build_run_row(model_name, train_metrics, train_time, val_metrics, val_time, test_metrics, test_time, split_sizes):
    train_acc = train_metrics.get("accuracy")
    val_acc = val_metrics.get("accuracy")
    test_acc = test_metrics.get("accuracy")

    train_eval_time = train_time + val_time
    total_time = train_eval_time + test_time
    test_count = int(test_metrics.get("sample_count") or 0)
    student_ms = (test_time * 1000.0 / test_count) if test_count > 0 else None

    return {
        "model": model_name,
        "run": 1,
        "select_metric": "f1",
        "best_epoch": 0,
        "epochs_trained": 0,
        "train_samples": split_sizes["train"],
        "val_samples": split_sizes["val"],
        "test_samples": split_sizes["test"],
        "train_acc_at_best": train_acc,
        "train_f1_at_best": train_metrics.get("f1"),
        "train_precision_at_best": train_metrics.get("precision"),
        "train_recall_at_best": train_metrics.get("recall"),
        "train_loss_at_best": None,
        "train_err_at_best": error_rate(train_acc),
        "val_acc_at_best": val_acc,
        "val_f1_at_best": val_metrics.get("f1"),
        "val_precision_at_best": val_metrics.get("precision"),
        "val_recall_at_best": val_metrics.get("recall"),
        "val_loss_at_best": None,
        "val_err_at_best": error_rate(val_acc),
        "test_acc": test_acc,
        "test_f1": test_metrics.get("f1"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_loss": None,
        "test_err": error_rate(test_acc),
        "test_sample_count": test_count,
        "test_time_seconds": test_time,
        "train_time_seconds": train_eval_time,
        "total_time_seconds": total_time,
        "student_ms": student_ms,
        "best_checkpoint": OFFICIAL_SOURCE.get(model_name, "official_pretrained"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only official pretrained baseline evaluation on train/val/test splits. "
            "Outputs compare-style runs/aggregate CSV files."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_SPECS.keys()),
        help="Official pretrained baselines to evaluate.",
    )
    parser.add_argument(
        "--include-optional-official",
        action="store_true",
        help="Append optional official models (WavLM/Pyannote/WeSpeaker).",
    )
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--half-duration", type=float, default=2.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--cache-dir", default="baseline/output_official/model_cache")
    parser.add_argument("--summary-csv", default="baseline/output_official/summary_manifest_runs.csv")
    parser.add_argument("--summary-agg-csv", default="baseline/output_official/summary_manifest_aggregate.csv")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.include_optional_official:
        for model_name in OPTIONAL_MODELS:
            if model_name not in args.models:
                args.models.append(model_name)

    root = Path(__file__).resolve().parents[1]
    cache_dir = root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / args.manifest_csv
    dataset_root = root / args.dataset_root
    splits = build_train_val_test_samples(
        manifest_csv=manifest_path,
        dataset_root=dataset_root,
        target_sr=args.sr,
        include_time_windows=False,
    )

    for split_name in ("train", "val", "test"):
        set_half_duration(splits[split_name], args.half_duration)
        preload_audio_pairs(splits[split_name], limit=None)

    split_sizes = {k: len(v) for k, v in splits.items()}
    if min(split_sizes.values()) == 0:
        raise RuntimeError(
            f"Empty split detected: train={split_sizes['train']}, "
            f"val={split_sizes['val']}, test={split_sizes['test']}"
        )

    device = select_device(args.device)
    print(f"device               : {device}")
    print(f"manifest_csv         : {manifest_path}")
    print(
        "samples (train/val/test): "
        f"{split_sizes['train']}/{split_sizes['val']}/{split_sizes['test']}"
    )
    print(f"models               : {', '.join(args.models)}")
    print("mode                 : inference only (no training)")

    run_rows = []
    per_model_rows = {}

    for model_name in args.models:
        print(f"\n=== Evaluating {model_name} ===")
        init_kwargs = {"device": device, "cache_dir": cache_dir}
        if model_name == "pyannote_wespeaker_voxceleb_resnet34_lm":
            init_kwargs["hf_token"] = args.hf_token

        model = None
        try:
            model_class = import_model_class(model_name)
            model = model_class(**init_kwargs)

            train_metrics, train_time = evaluate_split(
                model,
                splits["train"],
                args.sr,
                desc=f"{model_name} train",
            )
            val_metrics, val_time = evaluate_split(
                model,
                splits["val"],
                args.sr,
                desc=f"{model_name} val",
            )
            test_metrics, test_time = evaluate_split(
                model,
                splits["test"],
                args.sr,
                desc=f"{model_name} test",
            )

            row = build_run_row(
                model_name=model_name,
                train_metrics=train_metrics,
                train_time=train_time,
                val_metrics=val_metrics,
                val_time=val_time,
                test_metrics=test_metrics,
                test_time=test_time,
                split_sizes=split_sizes,
            )
            run_rows.append(row)
            per_model_rows.setdefault(model_name, []).append(row)

            print(
                f"test_acc={row['test_acc']:.4f} test_f1={row['test_f1']:.4f} "
                f"test_time_s={row['test_time_seconds']:.3f}"
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"[warn] {model_name} failed: {type(exc).__name__}: {exc}")
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate_rows = [
        aggregate_model_rows(model_name, rows)
        for model_name, rows in per_model_rows.items()
        if rows
    ]

    summary_csv_path = root / args.summary_csv
    summary_agg_csv_path = root / args.summary_agg_csv
    write_csv(summary_csv_path, run_rows, RUN_FIELDNAMES)
    write_csv(summary_agg_csv_path, aggregate_rows, AGG_FIELDNAMES)

    print(f"\nRun-level CSV        : {summary_csv_path}")
    print(f"Aggregate CSV        : {summary_agg_csv_path}")


if __name__ == "__main__":
    main()
