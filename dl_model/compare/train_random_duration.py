import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.dataloader import (
    DistillationPairDataset,
    assign_random_half_durations,
    build_samples_from_new_extracted,
    build_samples_from_old_all,
    collate_audio_pairs,
    preload_audio_pairs,
)
from dl_model.compare.shared import (
    augment_waveforms,
    benchmark_student,
    evaluate_student,
    load_soft_labels,
    set_seed,
    soft_distill_loss,
)


MODEL_MODULES = {
    "tdnn": "dl_model.compare.model_tdnn",
    "final_model": "dl_model.compare.model_final_model",
    "escapetdnn": "dl_model.compare.model_escapetdnn",
    "ecapatdnn": "dl_model.compare.model_escapetdnn",
    "redimnet": "dl_model.compare.model_redimnet",
    "sincnet": "dl_model.compare.model_sincnet",
    "sincnet_tdnn": "dl_model.compare.model_sincnet_tdnn",
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


def build_duration_manifest(train_samples, test_samples):
    return {
        "train": [
            {
                "audio_path": sample.get("audio_path"),
                "source_index": sample.get("source_index"),
                "half_duration": sample.get("half_duration"),
            }
            for sample in train_samples
        ],
        "test": [
            {
                "audio_path": sample.get("audio_path"),
                "test_row_index": sample.get("test_row_index"),
                "half_duration": sample.get("half_duration"),
            }
            for sample in test_samples
        ],
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one_model(model_name, args, train_samples, test_samples, device, run_index):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_run{run_index}" if args.repeat > 1 else ""
    best_acc_path = checkpoint_dir / f"{model_name}{suffix}_best_acc.pth"
    best_f1_path = checkpoint_dir / f"{model_name}{suffix}_best_f1.pth"
    final_path = checkpoint_dir / f"{model_name}{suffix}_final.pth"

    train_dataset = DistillationPairDataset(train_samples)
    test_dataset = DistillationPairDataset(test_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )

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

    best_f1 = -1.0
    best_f1_acc = 0.0
    best_f1_epoch = 0
    best_acc = -1.0
    best_acc_f1 = -1.0
    best_acc_epoch = 0
    no_improve = 0
    best_acc_record = None
    best_f1_record = None

    print(f"\n=== Training {model_name} (run {run_index}/{args.repeat}) ===")
    for epoch in range(1, args.epochs + 1):
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
                left = augment_waveforms(left)
                right = augment_waveforms(right)

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
        test_metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['f1']:.4f} "
            f"test_loss={test_metrics['loss']:.4f}"
        )

        current_acc = test_metrics["accuracy"] if test_metrics["accuracy"] is not None else -1.0
        current_f1 = test_metrics["f1"] if test_metrics["f1"] is not None else -1.0

        improved_acc = current_acc > best_acc or (
            abs(current_acc - best_acc) < 1e-8 and current_f1 > best_acc_f1
        )
        if improved_acc:
            best_acc = current_acc
            best_acc_f1 = current_f1
            best_acc_epoch = epoch
            best_acc_record = {
                "model": model_name,
                "run": run_index,
                "epoch": epoch,
                "test_acc": float(current_acc),
                "test_f1": float(current_f1),
                "test_loss": float(test_metrics["loss"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "train_loss": float(train_loss),
                "args": vars(args),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_metrics": test_metrics,
                    "args": vars(args),
                    "model_name": model_name,
                    "selection_metric": "accuracy",
                },
                best_acc_path,
            )

        improved_f1 = current_f1 > best_f1 or (
            abs(current_f1 - best_f1) < 1e-8 and current_acc > best_f1_acc
        )
        if improved_f1:
            best_f1 = current_f1
            best_f1_acc = current_acc
            best_f1_epoch = epoch
            no_improve = 0
            best_f1_record = {
                "model": model_name,
                "run": run_index,
                "epoch": epoch,
                "test_acc": float(current_acc),
                "test_f1": float(current_f1),
                "test_loss": float(test_metrics["loss"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "train_loss": float(train_loss),
                "args": vars(args),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_metrics": test_metrics,
                    "args": vars(args),
                    "model_name": model_name,
                    "selection_metric": "f1",
                },
                best_f1_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping for {model_name} at epoch {epoch}.")
                break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_acc_epoch": best_acc_epoch,
            "best_f1_epoch": best_f1_epoch,
            "best_acc": best_acc,
            "best_f1": best_f1,
            "best_acc_f1": best_acc_f1,
            "best_f1_acc": best_f1_acc,
            "args": vars(args),
            "model_name": model_name,
        },
        final_path,
    )

    best_acc_record_path = checkpoint_dir / f"{model_name}{suffix}_best_acc_record.json"
    best_f1_record_path = checkpoint_dir / f"{model_name}{suffix}_best_f1_record.json"

    if best_acc_record is not None:
        with open(best_acc_record_path, "w", encoding="utf-8") as f:
            json.dump(best_acc_record, f, indent=2, ensure_ascii=False)

    if best_f1_record is not None:
        with open(best_f1_record_path, "w", encoding="utf-8") as f:
            json.dump(best_f1_record, f, indent=2, ensure_ascii=False)

    print(
        f"Finished {model_name} | "
        f"best_test_acc={best_acc:.4f} (epoch {best_acc_epoch}) | "
        f"best_test_f1={best_f1:.4f} (epoch {best_f1_epoch})"
    )

    student_ms = benchmark_student(model, test_dataset, device=device, limit=args.benchmark_samples)
    return {
        "model": model_name,
        "run": run_index,
        "best_acc_epoch": best_acc_epoch,
        "best_f1_epoch": best_f1_epoch,
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_acc_f1": best_acc_f1,
        "best_f1_acc": best_f1_acc,
        "student_ms": student_ms,
        "best_acc_checkpoint": str(best_acc_path),
        "best_f1_checkpoint": str(best_f1_path),
        "final_checkpoint": str(final_path),
        "best_acc_record": str(best_acc_record_path) if best_acc_record else None,
        "best_f1_record": str(best_f1_record_path) if best_f1_record else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare models with fixed random 1-2s segment durations.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_MODULES.keys()), choices=list(MODEL_MODULES.keys()))
    parser.add_argument("--old-csv", default="dl_model/baseline_train_test_segments.csv")
    parser.add_argument("--old-train-audio-dir", default="datasets/mlp_train/train")
    parser.add_argument("--old-test-audio-dir", default="datasets/mlp_train/test")
    parser.add_argument("--new-test-csv", default="dl_model/baseline_train_test_segments_switchlingua_seame.csv")
    parser.add_argument("--new-test-audio-dir", default="datasets/baseline_switchlingua_seame_testset/test")
    parser.add_argument("--soft-labels-cache", default="dl_model/checkpoints/speechbrain_soft_labels_old_all_eval_new.pt")
    parser.add_argument("--checkpoint-dir", default="dl_model/compare/output/checkpoints_random_duration")
    parser.add_argument("--summary-path", default="dl_model/compare/output/summary_random_duration.json")
    parser.add_argument("--duration-manifest-path", default="dl_model/compare/output/random_duration_manifest.json")
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
    parser.add_argument("--eval-tta-swap", action="store_true", default=True)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-half-duration", type=float, default=1.0)
    parser.add_argument("--max-half-duration", type=float, default=2.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_samples = build_samples_from_old_all(
        root / args.old_csv,
        root / args.old_train_audio_dir,
        root / args.old_test_audio_dir,
        target_sr=args.sr,
    )
    test_samples = build_samples_from_new_extracted(
        root / args.new_test_csv,
        root / args.new_test_audio_dir,
        target_sr=args.sr,
        split="test",
    )
    load_soft_labels(root / args.soft_labels_cache, train_samples, test_samples)

    assign_random_half_durations(
        train_samples,
        min_half_duration=args.min_half_duration,
        max_half_duration=args.max_half_duration,
        seed=args.seed,
    )
    assign_random_half_durations(
        test_samples,
        min_half_duration=args.min_half_duration,
        max_half_duration=args.max_half_duration,
        seed=args.seed + 1,
    )
    preload_audio_pairs(train_samples, limit=args.benchmark_samples)
    preload_audio_pairs(test_samples, limit=min(args.benchmark_samples, len(test_samples)))

    duration_manifest_path = root / args.duration_manifest_path
    duration_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(duration_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "min_half_duration": args.min_half_duration,
                "max_half_duration": args.max_half_duration,
                "manifest": build_duration_manifest(train_samples, test_samples),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_samples)} | Test samples: {len(test_samples)}")
    print(
        f"Random half duration range: [{args.min_half_duration:.2f}, {args.max_half_duration:.2f}] sec "
        f"with seed={args.seed}"
    )
    print(f"Saved duration manifest to: {duration_manifest_path}")

    summary_rows = []
    aggregate_rows = []
    for model_name in args.models:
        model_runs = []
        for run_index in range(1, args.repeat + 1):
            set_seed(args.seed + run_index - 1)
            result = train_one_model(model_name, args, train_samples, test_samples, device, run_index)
            summary_rows.append(result)
            model_runs.append(result)

        acc_mean, acc_std = mean_std([row["best_acc"] for row in model_runs])
        f1_mean, f1_std = mean_std([row["best_f1"] for row in model_runs])
        ms_values = [row["student_ms"] for row in model_runs if row["student_ms"] is not None]
        ms_mean, ms_std = mean_std(ms_values)
        aggregate_rows.append(
            {
                "model": model_name,
                "runs": len(model_runs),
                "best_acc_mean": acc_mean,
                "best_acc_std": acc_std,
                "best_f1_mean": f1_mean,
                "best_f1_std": f1_std,
                "student_ms_mean": ms_mean,
                "student_ms_std": ms_std,
            }
        )

    summary_path = root / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"runs": summary_rows, "aggregate": aggregate_rows}, f, indent=2, ensure_ascii=False)

    write_csv(summary_path.with_suffix(".csv"), summary_rows)
    write_csv(summary_path.with_name(summary_path.stem + "_aggregate.csv"), aggregate_rows)

    best_acc_data = []
    best_f1_data = []
    for row in summary_rows:
        if row.get("best_acc_record") and Path(row["best_acc_record"]).exists():
            with open(row["best_acc_record"], "r", encoding="utf-8") as f:
                best_acc_data.append(json.load(f))
        if row.get("best_f1_record") and Path(row["best_f1_record"]).exists():
            with open(row["best_f1_record"], "r", encoding="utf-8") as f:
                best_f1_data.append(json.load(f))

    if best_acc_data:
        write_csv(summary_path.with_name(summary_path.stem + "_best_acc.csv"), best_acc_data)
    if best_f1_data:
        write_csv(summary_path.with_name(summary_path.stem + "_best_f1.csv"), best_f1_data)

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
