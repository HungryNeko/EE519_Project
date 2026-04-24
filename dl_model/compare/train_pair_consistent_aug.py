import argparse
import csv
import importlib
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.dataloader import DistillationPairDataset, collate_audio_pairs
from dl_model.compare.shared import (
    benchmark_student,
    evaluate_student,
    load_soft_labels,
    set_seed,
    soft_distill_loss,
)
from datasets.train_test2.dataloader import (
    assign_teacher_prob_from_labels,
    build_train_val_test_samples,
    preload_audio_pairs,
    set_half_duration,
)


MODEL_MODULES = {
    "tdnn": "dl_model.compare.model_tdnn",
    "final_model": "dl_model.compare.model_final_model",
    "ecapatdnn": "dl_model.compare.model_escapetdnn",
    "redimnet": "dl_model.compare.model_redimnet",
    "sincnet": "dl_model.compare.model_sincnet",
    # "sincnet_tdnn": "dl_model.compare.model_sincnet_tdnn",
}


def import_builder(model_name):
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


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


def _sample_aug_params(args, device):
    apply_speed = bool(torch.rand(1, device=device).item() < args.aug_speed_p)
    speed = 1.0
    if apply_speed:
        speed = torch.empty(1, device=device).uniform_(args.aug_speed_min, args.aug_speed_max).item()
    gain = torch.empty(1, device=device).uniform_(args.aug_gain_low, args.aug_gain_high).item()
    noise_std = torch.empty(1, device=device).uniform_(0.0, args.aug_noise_std).item()
    flip = bool(torch.rand(1, device=device).item() < args.aug_polarity_flip_p)
    return {
        "speed": speed,
        "gain": gain,
        "noise_std": noise_std,
        "flip": flip,
    }


def _time_scale_then_restore(wav_1d, speed):
    if abs(speed - 1.0) < 1e-4:
        return wav_1d
    src_len = int(wav_1d.shape[0])
    dst_len = max(8, int(round(src_len / speed)))
    x = wav_1d.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=dst_len, mode="linear", align_corners=False)
    x = F.interpolate(x, size=src_len, mode="linear", align_corners=False)
    return x.squeeze(0).squeeze(0)


def _apply_aug_params(wav_1d, params):
    out = wav_1d
    out = _time_scale_then_restore(out, params["speed"])
    out = out * params["gain"]
    if params["flip"]:
        out = -out
    if params["noise_std"] > 0.0:
        out = out + torch.randn_like(out) * params["noise_std"]
    return out.clamp_(-1.0, 1.0)


def augment_waveform_pairs(left, right, labels, args):
    left_out = left.clone()
    right_out = right.clone()
    same_speaker_mask = labels.eq(args.same_speaker_label)

    for idx in range(left.size(0)):
        if bool(same_speaker_mask[idx].item()):
            # switch=True -> same speaker: enforce the same augmentation params.
            shared_params = _sample_aug_params(args, left.device)
            left_out[idx] = _apply_aug_params(left_out[idx], shared_params)
            right_out[idx] = _apply_aug_params(right_out[idx], shared_params)
        else:
            left_out[idx] = _apply_aug_params(left_out[idx], _sample_aug_params(args, left.device))
            right_out[idx] = _apply_aug_params(right_out[idx], _sample_aug_params(args, left.device))

    return left_out, right_out


def train_one_model(model_name, args, train_samples, val_samples, test_samples, device, run_index):
    train_start_time = time.perf_counter()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_run{run_index}" if args.repeat > 1 else ""
    best_checkpoint_path = checkpoint_dir / f"{model_name}{suffix}_best_val_{args.select_metric}.pth"

    train_loader = build_loader(train_samples, args.batch_size, True, args.num_workers)
    val_loader = build_loader(val_samples, args.batch_size, False, args.num_workers)
    test_loader = build_loader(test_samples, args.batch_size, False, args.num_workers)

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
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"{model_name} {epoch}/{args.epochs}")
        for batch in pbar:
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"].to(device)
            teacher_prob = batch["teacher_prob"].to(device)

            if args.waveform_aug:
                left, right = augment_waveform_pairs(left, right, labels, args)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(left, right)
                hard_loss = ce_loss_fn(logits, labels)
                distill_loss = soft_distill_loss(logits, teacher_prob, args.temperature)
                loss = args.alpha * hard_loss + (1.0 - args.alpha) * distill_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / max(train_count, 1)
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
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
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
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)

    test_t0 = time.perf_counter()
    test_metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
    test_seconds = time.perf_counter() - test_t0
    total_time_seconds = train_seconds + test_seconds
    print(
        f"{model_name} stop@epoch {epochs_trained} | "
        f"test_acc={fmt_metric(to_float(test_metrics.get('accuracy')))} "
        f"test_f1={fmt_metric(to_float(test_metrics.get('f1')))} "
        f"test_loss={fmt_metric(to_float(test_metrics.get('loss')))}"
    )
    # benchmark_student expects precomputed left_audio/right_audio on raw sample dicts.
    preload_audio_pairs(test_samples, limit=args.benchmark_samples)
    student_ms = benchmark_student(model, test_samples, device=device, limit=args.benchmark_samples)

    best_snapshot = best_snapshot or {}
    test_acc = to_float(test_metrics.get("accuracy"))
    val_acc = best_snapshot.get("val_acc")
    train_acc = best_snapshot.get("train_acc")

    return {
        "model": model_name,
        "run": run_index,
        "select_metric": args.select_metric,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
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
        "test_f1": to_float(test_metrics.get("f1")),
        "test_precision": to_float(test_metrics.get("precision")),
        "test_recall": to_float(test_metrics.get("recall")),
        "test_loss": to_float(test_metrics.get("loss")),
        "test_err": error_rate(test_acc),
        "test_sample_count": test_metrics.get("sample_count"),
        "test_time_seconds": test_seconds,
        "train_time_seconds": train_seconds,
        "total_time_seconds": total_time_seconds,
        "student_ms": student_ms,
        "best_checkpoint": str(best_checkpoint_path),
    }


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


def main():
    parser = argparse.ArgumentParser(
        description="Train compare models with train/val/test from a unified manifest, select best by val, then report test in CSV."
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_MODULES.keys()), choices=list(MODEL_MODULES.keys()))
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--soft-labels-cache", default="dl_model/checkpoints/speechbrain_soft_labels_old_all_eval_new.pt")
    parser.add_argument("--checkpoint-dir", default="dl_model/compare/output/checkpoints_manifest")
    parser.add_argument("--summary-csv", default="dl_model/compare/output/summary_manifest_runs.csv")
    parser.add_argument("--summary-agg-csv", default="dl_model/compare/output/summary_manifest_aggregate.csv")
    parser.add_argument("--student-device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--student-channels", type=int, nargs="+", default=[128, 192, 256, 256])
    parser.add_argument("--ecapa-channels", type=int, default=256)
    parser.add_argument("--redimnet-channels", type=int, default=48)
    parser.add_argument("--sinc-channels", type=int, default=80)
    parser.add_argument("--final-model-weight-path", default="dl_model/final_model/sincnet_best_acc.pth")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--time-mask-max", type=int, default=12)
    parser.add_argument("--freq-mask-max", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--waveform-aug", action="store_true", default=True)
    parser.add_argument("--no-waveform-aug", dest="waveform_aug", action="store_false")
    parser.add_argument("--same-speaker-label", type=int, default=1)
    parser.add_argument("--aug-gain-low", type=float, default=0.9)
    parser.add_argument("--aug-gain-high", type=float, default=1.1)
    parser.add_argument("--aug-noise-std", type=float, default=0.003)
    parser.add_argument("--aug-speed-p", type=float, default=0.25)
    parser.add_argument("--aug-speed-min", type=float, default=0.95)
    parser.add_argument("--aug-speed-max", type=float, default=1.05)
    parser.add_argument("--aug-polarity-flip-p", type=float, default=0.1)
    parser.add_argument("--eval-tta-swap", action="store_true", default=True)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--half-duration", type=float, default=2.0)
    parser.add_argument("--select-metric", default="f1", choices=["f1", "acc"])
    args = parser.parse_args()

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

    set_half_duration(train_samples, args.half_duration)
    set_half_duration(val_samples, args.half_duration)
    set_half_duration(test_samples, args.half_duration)

    cache_loaded = False
    try:
        load_soft_labels(root / args.soft_labels_cache, train_samples, test_samples)
        cache_loaded = True
    except Exception as e:
        print(f"[WARN] Failed to load soft labels cache, fallback to hard labels for train/test: {e}")
        assign_teacher_prob_from_labels(train_samples)
        assign_teacher_prob_from_labels(test_samples)
    assign_teacher_prob_from_labels(val_samples)

    preload_audio_pairs(train_samples, limit=args.benchmark_samples)

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError(
            f"Loaded empty split(s): train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

    print(f"Using device: {device}")
    print(
        f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | "
        f"Test samples: {len(test_samples)} | Soft labels cache loaded: {cache_loaded}"
    )

    run_rows = []
    aggregate_rows = []
    for model_name in args.models:
        model_rows = []
        for run_index in range(1, args.repeat + 1):
            set_seed(args.seed + run_index - 1)
            row = train_one_model(
                model_name=model_name,
                args=args,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                device=device,
                run_index=run_index,
            )
            run_rows.append(row)
            model_rows.append(row)
        aggregate_rows.append(aggregate_model_rows(model_name, model_rows))

    summary_csv_path = root / args.summary_csv
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_csv_path, run_rows)

    summary_agg_csv_path = root / args.summary_agg_csv
    summary_agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_agg_csv_path, aggregate_rows)

    print(f"\nRun-level CSV: {summary_csv_path}")
    print(f"Aggregate CSV: {summary_agg_csv_path}")


if __name__ == "__main__":
    main()
