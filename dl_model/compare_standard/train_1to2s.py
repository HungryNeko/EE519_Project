import argparse
import csv
import importlib
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.compare.shared import augment_waveforms, benchmark_student, evaluate_student, set_seed
from dl_model.dataloader import DistillationPairDataset, collate_audio_pairs
from datasets.train_test2.dataloader import (
    assign_random_half_durations,
    assign_teacher_prob_from_labels,
    build_train_val_test_samples,
    preload_audio_pairs,
)


MODEL_MODULES = {
    "tdnn": "dl_model.compare_standard.model_official_tdnn",
    "ecapatdnn": "dl_model.compare_standard.model_official_ecapatdnn",
    "resnet": "dl_model.compare_standard.model_official_resnet",
}


def import_builder(model_name):
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model


def to_float(value):
    if value is None:
        return None
    return float(value)


def fmt_metric(value):
    if value is None:
        return "nan"
    return f"{value:.4f}"


def error_rate(acc):
    if acc is None:
        return None
    return 1.0 - float(acc)


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_loader(samples, batch_size, shuffle, num_workers):
    return DataLoader(
        DistillationPairDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_pairs,
    )


def normalize_duration_list(values):
    out = []
    seen = set()
    for value in values:
        duration = float(value)
        key = round(duration, 6)
        if key in seen:
            continue
        out.append(duration)
        seen.add(key)
    return out


def duration_tag(duration):
    return str(float(duration)).replace(".", "p").replace("-", "m")


def set_fixed_half_duration(samples, half_duration):
    value = float(half_duration)
    for sample in samples:
        sample["half_duration"] = value
        sample.pop("left_audio", None)
        sample.pop("right_audio", None)


def evaluate_test_durations(model, args, test_samples, device, ce_loss_fn, model_name):
    duration_rows = []
    duration_fields = {}
    total_eval_seconds = 0.0

    for half_duration in args.test_half_durations:
        set_fixed_half_duration(test_samples, half_duration)
        test_loader = build_loader(test_samples, args.batch_size, False, args.num_workers)
        t0 = time.perf_counter()
        metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
        elapsed = time.perf_counter() - t0
        total_eval_seconds += elapsed

        acc = to_float(metrics.get("accuracy"))
        f1 = to_float(metrics.get("f1"))
        precision = to_float(metrics.get("precision"))
        recall = to_float(metrics.get("recall"))
        loss = to_float(metrics.get("loss"))
        err = error_rate(acc)
        sample_count = metrics.get("sample_count")

        print(
            f"{model_name} | test@{half_duration:.1f}s "
            f"acc={fmt_metric(acc)} f1={fmt_metric(f1)} loss={fmt_metric(loss)}"
        )

        tag = duration_tag(half_duration)
        duration_fields[f"test_{tag}_acc"] = acc
        duration_fields[f"test_{tag}_f1"] = f1
        duration_fields[f"test_{tag}_precision"] = precision
        duration_fields[f"test_{tag}_recall"] = recall
        duration_fields[f"test_{tag}_loss"] = loss
        duration_fields[f"test_{tag}_err"] = err
        duration_fields[f"test_{tag}_time_seconds"] = elapsed
        duration_fields[f"test_{tag}_sample_count"] = sample_count

        duration_rows.append(
            {
                "model": model_name,
                "test_half_duration": float(half_duration),
                "test_acc": acc,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "test_loss": loss,
                "test_err": err,
                "test_sample_count": sample_count,
                "test_time_seconds": elapsed,
            }
        )

    return duration_fields, duration_rows, total_eval_seconds


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
    test_time_total_mean, test_time_total_std = mean_std(collect("test_time_total_seconds"))
    total_time_mean, total_time_std = mean_std(collect("total_time_seconds"))
    student_ms_mean, student_ms_std = mean_std(collect("student_ms"))

    row = {
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
        "test_time_total_seconds_mean": test_time_total_mean,
        "test_time_total_seconds_std": test_time_total_std,
        "total_time_seconds_mean": total_time_mean,
        "total_time_seconds_std": total_time_std,
        "student_ms_mean": student_ms_mean,
        "student_ms_std": student_ms_std,
    }

    duration_keys = sorted(
        key
        for key in rows[0].keys()
        if key.startswith("test_") and key.endswith(("_acc", "_f1", "_err", "_loss", "_time_seconds"))
    )
    for key in duration_keys:
        mean, std = mean_std(collect(key))
        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std
    return row


def aggregate_duration_rows(duration_rows):
    groups = {}
    for row in duration_rows:
        key = (row["model"], float(row["test_half_duration"]))
        groups.setdefault(key, []).append(row)

    out = []
    for (model_name, half_duration), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        def collect(key):
            return [item[key] for item in rows if item.get(key) is not None]

        test_acc_mean, test_acc_std = mean_std(collect("test_acc"))
        test_f1_mean, test_f1_std = mean_std(collect("test_f1"))
        test_err_mean, test_err_std = mean_std(collect("test_err"))
        test_loss_mean, test_loss_std = mean_std(collect("test_loss"))
        test_time_mean, test_time_std = mean_std(collect("test_time_seconds"))

        out.append(
            {
                "model": model_name,
                "test_half_duration": half_duration,
                "runs": len(rows),
                "test_acc_mean": test_acc_mean,
                "test_acc_std": test_acc_std,
                "test_f1_mean": test_f1_mean,
                "test_f1_std": test_f1_std,
                "test_err_mean": test_err_mean,
                "test_err_std": test_err_std,
                "test_loss_mean": test_loss_mean,
                "test_loss_std": test_loss_std,
                "test_time_seconds_mean": test_time_mean,
                "test_time_seconds_std": test_time_std,
            }
        )
    return out


def train_one_model(model_name, args, train_samples, val_samples, test_samples, device, run_index):
    train_start_time = time.perf_counter()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_run{run_index}" if args.repeat > 1 else ""
    best_checkpoint_path = checkpoint_dir / f"{model_name}{suffix}_best_val_{args.select_metric}.pth"

    model = import_builder(model_name)(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_epoch = 0
    best_val_score = -1.0
    best_val_acc = -1.0
    best_snapshot = None
    epochs_trained = 0
    no_improve = 0

    print(f"\n=== Training {model_name} (run {run_index}/{args.repeat}) ===")
    for epoch in range(1, args.epochs + 1):
        epochs_trained = epoch

        assign_random_half_durations(
            train_samples,
            min_half_duration=args.train_min_half_duration,
            max_half_duration=args.train_max_half_duration,
            seed=args.seed + run_index * 10000 + epoch,
        )
        train_loader = build_loader(train_samples, args.batch_size, True, args.num_workers)

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"{model_name} {epoch}/{args.epochs}")
        for batch in pbar:
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"].to(device)

            if args.waveform_aug:
                left = augment_waveforms(left)
                right = augment_waveforms(right)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(left, right)
                loss = ce_loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / max(train_count, 1)
        assign_random_half_durations(
            val_samples,
            min_half_duration=args.val_min_half_duration,
            max_half_duration=args.val_max_half_duration,
            seed=args.seed + run_index * 10000 + epoch + 500000,
        )
        val_loader = build_loader(val_samples, args.batch_size, False, args.num_workers)
        train_metrics = evaluate_student(model, train_loader, device, ce_loss_fn, use_tta_swap=False)
        val_metrics = evaluate_student(model, val_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
        scheduler.step()

        current_val_acc = to_float(val_metrics.get("accuracy"))
        current_val_f1 = to_float(val_metrics.get("f1"))
        if args.select_metric == "acc":
            current_score = current_val_acc if current_val_acc is not None else -1.0
        else:
            current_score = current_val_f1 if current_val_f1 is not None else -1.0

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={fmt_metric(to_float(train_metrics.get('accuracy')))} "
            f"train_f1={fmt_metric(to_float(train_metrics.get('f1')))} "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={fmt_metric(to_float(val_metrics.get('accuracy')))} "
            f"val_f1={fmt_metric(to_float(val_metrics.get('f1')))} "
            f"val_loss={fmt_metric(to_float(val_metrics.get('loss')))}"
        )

        improved = current_score > best_val_score or (
            abs(current_score - best_val_score) < 1e-8
            and (current_val_acc if current_val_acc is not None else -1.0) > best_val_acc
        )
        if improved:
            best_val_score = current_score
            best_val_acc = current_val_acc if current_val_acc is not None else -1.0
            best_epoch = epoch
            no_improve = 0
            best_snapshot = {
                "train_acc": to_float(train_metrics.get("accuracy")),
                "train_f1": to_float(train_metrics.get("f1")),
                "train_precision": to_float(train_metrics.get("precision")),
                "train_recall": to_float(train_metrics.get("recall")),
                "train_loss": to_float(train_metrics.get("loss")),
                "val_acc": to_float(val_metrics.get("accuracy")),
                "val_f1": to_float(val_metrics.get("f1")),
                "val_precision": to_float(val_metrics.get("precision")),
                "val_recall": to_float(val_metrics.get("recall")),
                "val_loss": to_float(val_metrics.get("loss")),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "selection_metric": args.select_metric,
                },
                best_checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping for {model_name} at epoch {epoch}.")
                break

    train_seconds = time.perf_counter() - train_start_time

    if best_checkpoint_path.exists():
        payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state_dict)

    test_duration_fields, test_duration_rows, test_eval_total_seconds = evaluate_test_durations(
        model=model,
        args=args,
        test_samples=test_samples,
        device=device,
        ce_loss_fn=ce_loss_fn,
        model_name=model_name,
    )

    primary_duration = float(args.primary_test_half_duration)
    if round(primary_duration, 6) not in {round(x, 6) for x in args.test_half_durations}:
        primary_duration = float(args.test_half_durations[0])
    primary_tag = duration_tag(primary_duration)

    set_fixed_half_duration(test_samples, args.benchmark_half_duration)
    preload_audio_pairs(test_samples, limit=args.benchmark_samples)
    student_ms = benchmark_student(model, test_samples, device=device, limit=args.benchmark_samples)

    test_acc = test_duration_fields.get(f"test_{primary_tag}_acc")
    test_f1 = test_duration_fields.get(f"test_{primary_tag}_f1")
    test_precision = test_duration_fields.get(f"test_{primary_tag}_precision")
    test_recall = test_duration_fields.get(f"test_{primary_tag}_recall")
    test_loss = test_duration_fields.get(f"test_{primary_tag}_loss")
    test_err = test_duration_fields.get(f"test_{primary_tag}_err")
    test_time_seconds = test_duration_fields.get(f"test_{primary_tag}_time_seconds")
    test_sample_count = test_duration_fields.get(f"test_{primary_tag}_sample_count")

    total_time_seconds = train_seconds + test_eval_total_seconds
    best_snapshot = best_snapshot or {}
    train_acc = best_snapshot.get("train_acc")
    val_acc = best_snapshot.get("val_acc")

    run_row = {
        "model": model_name,
        "run": run_index,
        "select_metric": args.select_metric,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_half_duration_mode": "random_uniform",
        "train_half_duration_min": float(args.train_min_half_duration),
        "train_half_duration_max": float(args.train_max_half_duration),
        "val_half_duration_mode": "random_uniform",
        "val_half_duration_min": float(args.val_min_half_duration),
        "val_half_duration_max": float(args.val_max_half_duration),
        "primary_test_half_duration": float(primary_duration),
        "test_half_durations": ",".join(f"{x:.2f}" for x in args.test_half_durations),
        "train_acc_at_best": train_acc,
        "train_f1_at_best": best_snapshot.get("train_f1"),
        "train_precision_at_best": best_snapshot.get("train_precision"),
        "train_recall_at_best": best_snapshot.get("train_recall"),
        "train_loss_at_best": best_snapshot.get("train_loss"),
        "train_err_at_best": error_rate(train_acc),
        "val_acc_at_best": val_acc,
        "val_f1_at_best": best_snapshot.get("val_f1"),
        "val_precision_at_best": best_snapshot.get("val_precision"),
        "val_recall_at_best": best_snapshot.get("val_recall"),
        "val_loss_at_best": best_snapshot.get("val_loss"),
        "val_err_at_best": error_rate(val_acc),
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_loss": test_loss,
        "test_err": test_err,
        "test_sample_count": test_sample_count,
        "test_time_seconds": test_time_seconds,
        "test_time_total_seconds": test_eval_total_seconds,
        "train_time_seconds": train_seconds,
        "total_time_seconds": total_time_seconds,
        "student_ms": student_ms,
        "best_checkpoint": str(best_checkpoint_path),
    }
    run_row.update(test_duration_fields)
    for row in test_duration_rows:
        row["run"] = run_index
    return run_row, test_duration_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Official compare trainer with random 1-2s half-duration training augmentation; "
            "validation on fixed duration and test report on multiple durations."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_MODULES.keys()), choices=list(MODEL_MODULES.keys()))
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--checkpoint-dir", default="dl_model/compare_standard/output_official/checkpoints_1to2s")
    parser.add_argument("--summary-csv", default="dl_model/compare_standard/output_official/results_runs_1to2s.csv")
    parser.add_argument(
        "--summary-agg-csv",
        default="dl_model/compare_standard/output_official/results_aggregate_1to2s.csv",
    )
    parser.add_argument(
        "--test-duration-csv",
        default="dl_model/compare_standard/output_official/results_test_duration_runs_1to2s.csv",
    )
    parser.add_argument(
        "--test-duration-agg-csv",
        default="dl_model/compare_standard/output_official/results_test_duration_aggregate_1to2s.csv",
    )
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--student-channels", type=int, nargs="+", default=[128, 192, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--waveform-aug", action="store_true", default=False)
    parser.add_argument("--no-waveform-aug", dest="waveform_aug", action="store_false")
    parser.add_argument("--eval-tta-swap", action="store_true", default=False)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument("--benchmark-half-duration", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train-min-half-duration", type=float, default=1.0)
    parser.add_argument("--train-max-half-duration", type=float, default=2.0)
    parser.add_argument("--val-min-half-duration", type=float, default=1.0)
    parser.add_argument("--val-max-half-duration", type=float, default=2.0)
    parser.add_argument("--test-half-durations", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--primary-test-half-duration", type=float, default=1.5)
    parser.add_argument("--select-metric", default="f1", choices=["f1", "acc"])
    args = parser.parse_args()

    if args.train_min_half_duration <= 0 or args.train_max_half_duration <= 0:
        raise ValueError("train half-duration bounds must be > 0.")
    if args.train_min_half_duration > args.train_max_half_duration:
        raise ValueError("train_min_half_duration must be <= train_max_half_duration.")
    if args.val_min_half_duration <= 0 or args.val_max_half_duration <= 0:
        raise ValueError("val half-duration bounds must be > 0.")
    if args.val_min_half_duration > args.val_max_half_duration:
        raise ValueError("val_min_half_duration must be <= val_max_half_duration.")
    args.test_half_durations = normalize_duration_list(args.test_half_durations)
    if not args.test_half_durations:
        raise ValueError("test_half_durations cannot be empty.")

    root = Path(__file__).resolve().parents[2]
    manifest_path = root / args.manifest_csv
    dataset_root = root / args.dataset_root

    splits = build_train_val_test_samples(
        manifest_csv=manifest_path,
        dataset_root=dataset_root,
        target_sr=args.sr,
        include_time_windows=False,
    )
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError(
            f"Loaded empty split(s): train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

    assign_teacher_prob_from_labels(train_samples)
    assign_teacher_prob_from_labels(val_samples)
    assign_teacher_prob_from_labels(test_samples)

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Test samples: {len(test_samples)}")
    print(f"Models: {args.models}")
    print(
        f"Train half-duration: random U[{args.train_min_half_duration:.2f}, {args.train_max_half_duration:.2f}] | "
        f"Val: random U[{args.val_min_half_duration:.2f}, {args.val_max_half_duration:.2f}] | "
        f"Test: {[round(x, 2) for x in args.test_half_durations]}"
    )

    run_rows = []
    aggregate_rows = []
    duration_rows = []
    for model_name in args.models:
        model_rows = []
        for run_index in range(1, args.repeat + 1):
            set_seed(args.seed + run_index - 1)
            run_row, run_duration_rows = train_one_model(
                model_name=model_name,
                args=args,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                device=device,
                run_index=run_index,
            )
            run_rows.append(run_row)
            model_rows.append(run_row)
            duration_rows.extend(run_duration_rows)
        aggregate_rows.append(aggregate_model_rows(model_name, model_rows))

    duration_agg_rows = aggregate_duration_rows(duration_rows)

    summary_csv_path = root / args.summary_csv
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_csv_path, run_rows)

    summary_agg_csv_path = root / args.summary_agg_csv
    summary_agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_agg_csv_path, aggregate_rows)

    test_duration_csv_path = root / args.test_duration_csv
    test_duration_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(test_duration_csv_path, duration_rows)

    test_duration_agg_csv_path = root / args.test_duration_agg_csv
    test_duration_agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(test_duration_agg_csv_path, duration_agg_rows)

    print("\nDone.")
    print(f"Run-level CSV: {summary_csv_path}")
    print(f"Aggregate CSV: {summary_agg_csv_path}")
    print(f"Test-duration run CSV: {test_duration_csv_path}")
    print(f"Test-duration aggregate CSV: {test_duration_agg_csv_path}")


if __name__ == "__main__":
    main()
