import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.compare.model_final_model import build_model
from dl_model.compare.shared import (
    augment_waveforms,
    benchmark_student,
    evaluate_student,
    load_soft_labels,
    set_seed,
    soft_distill_loss,
)
from dl_model.dataloader import (
    DistillationPairDataset,
    build_samples_from_new_extracted,
    build_samples_from_old_all,
    collate_audio_pairs,
    preload_audio_pairs,
)


def assign_teacher_prob_from_labels(samples):
    for sample in samples:
        sample["teacher_prob"] = float(sample["label"])


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_and_eval(args, train_samples, val_samples, test_samples, device):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_path = checkpoint_dir / "final_model_best_val.pth"
    final_path = checkpoint_dir / "final_model_final.pth"
    best_val_record_path = checkpoint_dir / "final_model_best_val_record.json"
    test_record_path = checkpoint_dir / "final_model_test_record.json"

    train_loader = DataLoader(
        DistillationPairDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )
    val_loader = DataLoader(
        DistillationPairDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )
    test_loader = DataLoader(
        DistillationPairDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_val_epoch = 0
    best_val_metrics = None
    no_improve = 0

    print("\n=== Training final_model (select by val) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"final_model {epoch}/{args.epochs}")
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
        val_metrics = evaluate_student(model, val_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        current_val_f1 = val_metrics["f1"] if val_metrics["f1"] is not None else -1.0
        current_val_acc = val_metrics["accuracy"] if val_metrics["accuracy"] is not None else -1.0

        improved = current_val_f1 > best_val_f1 or (
            abs(current_val_f1 - best_val_f1) < 1e-8 and current_val_acc > best_val_acc
        )
        if improved:
            best_val_f1 = current_val_f1
            best_val_acc = current_val_acc
            best_val_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(model.state_dict(), best_val_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    torch.save(model.state_dict(), final_path)

    if best_val_path.exists():
        model.load_state_dict(torch.load(best_val_path, map_location=device))

    test_metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
    student_ms = benchmark_student(model, train_samples[: args.benchmark_samples], device)

    if best_val_metrics is not None:
        with open(best_val_record_path, "w", encoding="utf-8") as f:
            json.dump(best_val_metrics, f, indent=2)
    with open(test_record_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    result = {
        "model": "final_model",
        "best_val_epoch": best_val_epoch,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "test_acc_at_best_val": test_metrics.get("accuracy"),
        "test_f1_at_best_val": test_metrics.get("f1"),
        "test_precision_at_best_val": test_metrics.get("precision"),
        "test_recall_at_best_val": test_metrics.get("recall"),
        "test_loss_at_best_val": test_metrics.get("loss"),
        "student_ms": student_ms,
        "best_val_checkpoint": str(best_val_path),
        "final_checkpoint": str(final_path),
        "best_val_record": str(best_val_record_path) if best_val_metrics is not None else None,
        "test_record": str(test_record_path),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train final_model with train split, select best epoch on val split, then evaluate on test split."
    )
    parser.add_argument("--train-csv", default="dl_model/csv2/baseline_train_test_segments.csv")
    parser.add_argument("--train-train-audio-dir", default="datasets/train_test2/train")
    parser.add_argument("--train-test-audio-dir", default="datasets/train_test2/test")
    parser.add_argument("--val-csv", default="datasets/train_test2/val_segments_audio_package_500.csv")
    parser.add_argument("--val-audio-dir", default="datasets/train_test2/val")
    parser.add_argument("--test-csv", default="dl_model/csv2/baseline_train_test_segments_switchlingua_seame.csv")
    parser.add_argument("--test-audio-dir", default="datasets/train_test2/test")
    parser.add_argument("--soft-labels-cache", default="dl_model/checkpoints/speechbrain_soft_labels_old_all_eval_new.pt")
    parser.add_argument("--checkpoint-dir", default="dl_model/compare/output/checkpoints_final_model_val")
    parser.add_argument("--summary-path", default="dl_model/compare/output/summary_final_model_val.json")
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
    parser.add_argument(
        "--half-duration",
        type=float,
        default=2.0,
        help="Used by lazy waveform split; with 4s clips, 2.0 means 2s+2s.",
    )
    parser.add_argument("--val-split-name", default="val")
    parser.add_argument("--test-split-name", default="test")

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    set_seed(args.seed)

    train_samples = build_samples_from_old_all(
        root / args.train_csv,
        root / args.train_train_audio_dir,
        root / args.train_test_audio_dir,
        target_sr=args.sr,
    )
    val_samples = build_samples_from_new_extracted(
        root / args.val_csv,
        root / args.val_audio_dir,
        target_sr=args.sr,
        split=args.val_split_name,
    )
    test_samples = build_samples_from_new_extracted(
        root / args.test_csv,
        root / args.test_audio_dir,
        target_sr=args.sr,
        split=args.test_split_name,
    )

    for sample in train_samples:
        sample["half_duration"] = args.half_duration
    for sample in val_samples:
        sample["half_duration"] = args.half_duration
    for sample in test_samples:
        sample["half_duration"] = args.half_duration

    # Distillation cache is expected for train/test only; val uses label as fallback teacher prob.
    cache_loaded = False
    try:
        load_soft_labels(root / args.soft_labels_cache, train_samples, test_samples)
        cache_loaded = True
    except Exception as e:
        print(f"[WARN] Failed to load soft labels cache, fallback to hard labels as teacher probs: {e}")
        assign_teacher_prob_from_labels(train_samples)
        assign_teacher_prob_from_labels(test_samples)

    assign_teacher_prob_from_labels(val_samples)

    preload_audio_pairs(train_samples, limit=args.benchmark_samples)

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    if len(train_samples) == 0:
        raise RuntimeError("No train samples loaded.")
    if len(val_samples) == 0:
        raise RuntimeError("No val samples loaded.")
    if len(test_samples) == 0:
        raise RuntimeError("No test samples loaded.")

    print(f"Using device: {device}")
    print(
        f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | "
        f"Test samples: {len(test_samples)} | Soft labels cache loaded: {cache_loaded}"
    )

    result = train_and_eval(args, train_samples, val_samples, test_samples, device)

    summary_path = root / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)

    summary_csv_path = summary_path.with_suffix(".csv")
    write_csv(summary_csv_path, [result])

    print(f"\nDone. Summary JSON: {summary_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
