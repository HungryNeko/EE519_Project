from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.run_baselines import OFFICIAL_SOURCE, MODEL_SPECS, import_model_class, select_device
from datasets.train_test2.dataloader import build_train_val_test_samples, preload_audio_pairs, set_half_duration


DEFAULT_B1_MODELS = [
    "speechbrain_ecapa",
    "speechbrain_xvector",
    "resemblyzer_ge2e",
    "wespeaker_english",
]


def load_models_from_runs_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    models = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            model_name = str(row.get("model", "")).strip()
            if model_name and model_name in MODEL_SPECS and model_name not in models:
                models.append(model_name)
    return models


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_eer(labels: list[int], scores: list[float]) -> tuple[float, float]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("empty labels/scores")

    positives = sum(1 for x in labels if int(x) == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires both positive and negative samples")

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    points = [(0.0, 1.0, float("inf"))]

    tp = 0
    fp = 0
    idx = 0
    while idx < len(pairs):
        threshold = float(pairs[idx][0])
        while idx < len(pairs) and float(pairs[idx][0]) == threshold:
            if int(pairs[idx][1]) == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        far = fp / negatives
        frr = (positives - tp) / positives
        points.append((far, frr, threshold))

    best_far, best_frr, best_thr = min(points, key=lambda x: abs(x[0] - x[1]))
    eer_nearest = 0.5 * (best_far + best_frr)

    eer_interp = None
    thr_interp = best_thr
    for i in range(len(points) - 1):
        far1, frr1, thr1 = points[i]
        far2, frr2, thr2 = points[i + 1]
        diff1 = far1 - frr1
        diff2 = far2 - frr2
        if diff1 == 0.0:
            eer_interp = far1
            thr_interp = thr1
            break
        if diff2 == 0.0:
            eer_interp = far2
            thr_interp = thr2
            break
        if diff1 * diff2 < 0:
            alpha = diff1 / (diff1 - diff2)
            eer_interp = far1 + alpha * (far2 - far1)
            if math.isfinite(thr1) and math.isfinite(thr2):
                thr_interp = thr1 + alpha * (thr2 - thr1)
            else:
                thr_interp = thr2
            break

    if eer_interp is None:
        return float(eer_nearest), float(best_thr)
    return float(eer_interp), float(thr_interp)


def evaluate_model_eer(model, samples: list[dict], score_type: str, desc: str) -> tuple[float, float, int, float]:
    labels = []
    scores = []
    t0 = time.perf_counter()
    iterator = tqdm(samples, desc=desc, leave=False) if desc else samples
    for sample in iterator:
        pred = model.predict(
            sample["left_audio"],
            sample["right_audio"],
            int(sample.get("target_sr", 16000)),
        )
        labels.append(int(sample["label"]))
        if score_type == "raw":
            scores.append(float(pred.raw_score))
        else:
            scores.append(float(pred.same_speaker_score))
    elapsed = time.perf_counter() - t0
    eer, thr = compute_eer(labels, scores)
    return eer, thr, len(labels), elapsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate official baseline models with EER only (speaker-verification style threshold sweep). "
            "Defaults to B1 models from baseline/output_official/summary_manifest_runs.csv when available."
        )
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS.keys()), default=None)
    parser.add_argument("--reference-runs-csv", default="baseline/output_official/summary_manifest_runs.csv")
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to evaluate. Ignored when --split is set.",
    )
    parser.add_argument("--half-duration", type=float, default=2.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache-dir", default="baseline/output_official/model_cache")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--score-type", default="same", choices=["same", "raw"])
    parser.add_argument("--output-csv", default="baseline/output_official/eer_baseline_output_official.csv")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    reference_runs_path = root / args.reference_runs_csv

    if args.models:
        models = list(args.models)
    else:
        models = load_models_from_runs_csv(reference_runs_path)
        if not models:
            models = list(DEFAULT_B1_MODELS)

    device = select_device(args.device)
    cache_dir = root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = build_train_val_test_samples(
        manifest_csv=root / args.manifest_csv,
        dataset_root=root / args.dataset_root,
        target_sr=args.sr,
        include_time_windows=False,
    )
    if args.split is not None:
        eval_splits = [args.split]
    else:
        eval_splits = []
        for split_name in args.splits:
            if split_name not in eval_splits:
                eval_splits.append(split_name)

    split_samples = {}
    for split_name in eval_splits:
        samples = splits[split_name]
        if not samples:
            raise RuntimeError(f"Split '{split_name}' is empty.")
        set_half_duration(samples, args.half_duration)
        preload_audio_pairs(samples, limit=None)
        split_samples[split_name] = samples

    print(f"device               : {device}")
    print(f"splits               : {', '.join(eval_splits)}")
    print(
        "samples              : "
        + ", ".join(f"{split_name}={len(split_samples[split_name])}" for split_name in eval_splits)
    )
    print(f"half_duration        : {args.half_duration}")
    print(f"models               : {', '.join(models)}")
    print(f"score_type           : {args.score_type}")

    rows = []
    for model_name in models:
        print(f"\n=== EER {model_name} ===")
        init_kwargs = {"device": device, "cache_dir": cache_dir}
        if model_name == "pyannote_wespeaker_voxceleb_resnet34_lm":
            init_kwargs["hf_token"] = args.hf_token

        model = None
        try:
            model_class = import_model_class(model_name)
            model = model_class(**init_kwargs)

            for split_name in eval_splits:
                try:
                    eer, threshold, sample_count, eval_seconds = evaluate_model_eer(
                        model=model,
                        samples=split_samples[split_name],
                        score_type=args.score_type,
                        desc=f"{model_name} {split_name}",
                    )
                    row = {
                        "experiment": "b1_baseline_output_official",
                        "model": model_name,
                        "run": 1,
                        "split": split_name,
                        "half_duration": float(args.half_duration),
                        "score_type": args.score_type,
                        "eer": eer,
                        "eer_threshold": threshold,
                        "sample_count": sample_count,
                        "eval_time_seconds": eval_seconds,
                        "best_checkpoint": OFFICIAL_SOURCE.get(model_name, "official_pretrained"),
                    }
                    rows.append(row)
                    print(
                        f"{model_name},{split_name},"
                        f"eer={eer:.6f},threshold={threshold:.6f},samples={sample_count}"
                    )
                except Exception as split_exc:
                    if args.fail_fast:
                        raise
                    print(
                        f"[warn] {model_name} on split={split_name} failed: "
                        f"{type(split_exc).__name__}: {split_exc}"
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

    if not rows:
        raise RuntimeError("No baseline model was evaluated successfully.")

    output_csv = root / args.output_csv
    write_csv(output_csv, rows)
    print(f"\nSaved EER CSV         : {output_csv}")


if __name__ == "__main__":
    main()
