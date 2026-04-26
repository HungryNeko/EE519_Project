from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.common import compute_metrics, load_audio
from datasets.train_test2.dataloader import build_train_val_test_samples
from dl_model.compare.eval_eer_compare import load_checkpoint_state, normalize_model_name
from dl_model.dataloader import DistillationPairDataset, collate_audio_pairs, split_pair_from_full_clip


MODEL_MODULES = {
    "tdnn": "dl_model.compare.model_tdnn",
    "ecapatdnn": "dl_model.compare.model_escapetdnn",
    "escapetdnn": "dl_model.compare.model_escapetdnn",
    "redimnet": "dl_model.compare.model_redimnet",
    "sincnet": "dl_model.compare.model_sincnet",
}


def import_builder(model_name):
    if model_name == "final_model":
        from dl_model.compare.shared import SincNetPairStudent

        return lambda args: SincNetPairStudent(
            sample_rate=args.sr,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            sinc_channels=args.sinc_channels,
        )
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def fmt(value) -> str:
    value = safe_float(value)
    if value is None:
        return "nan"
    return f"{value:.4f}"


def frame_rms(wav: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)
    if wav.size < frame_samples:
        padded = np.zeros(frame_samples, dtype=np.float32)
        padded[: wav.size] = wav
        wav = padded
    starts = np.arange(0, wav.size - frame_samples + 1, hop_samples, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)

    power = wav.astype(np.float32) * wav.astype(np.float32)
    cumsum = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    frame_energy = cumsum[starts + frame_samples] - cumsum[starts]
    return np.sqrt((frame_energy / float(frame_samples)) + 1e-12).astype(np.float32)


def estimate_speech_seconds(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    relative_rms: float,
    min_rms: float,
) -> tuple[float, float, float]:
    frame_samples = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_samples = max(1, int(round(sr * hop_ms / 1000.0)))

    def one_side(wav: np.ndarray) -> float:
        rms = frame_rms(wav.astype(np.float32), frame_samples, hop_samples)
        if rms.size == 0:
            return 0.0
        threshold = max(float(min_rms), float(relative_rms) * float(np.max(rms)))
        voiced_frames = int(np.sum(rms >= threshold))
        return voiced_frames * hop_samples / float(sr)

    left_seconds = one_side(left)
    right_seconds = one_side(right)
    return left_seconds + right_seconds, left_seconds, right_seconds


def attach_audio_and_speech_lengths(samples: list[dict], split_name: str, args) -> list[dict]:
    records = []
    for idx, sample in enumerate(tqdm(samples, desc=f"scan {split_name}", unit="sample"), start=1):
        sr = int(sample.get("target_sr", args.sr))
        wav, _ = load_audio(Path(sample["audio_file"]), sr=sr)
        half_duration = float(args.half_duration)
        left, right = split_pair_from_full_clip(wav, half_duration, sr)

        speech_seconds, left_speech, right_speech = estimate_speech_seconds(
            left=left,
            right=right,
            sr=sr,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            relative_rms=args.relative_rms,
            min_rms=args.min_rms,
        )

        sample["left_audio"] = left
        sample["right_audio"] = right
        sample["teacher_prob"] = 0.5
        sample["speech_seconds"] = speech_seconds
        sample["left_speech_seconds"] = left_speech
        sample["right_speech_seconds"] = right_speech
        sample["half_duration"] = half_duration

        records.append(
            {
                "split": split_name,
                "split_index": idx,
                "label": int(sample["label"]),
                "audio_rel_path": sample.get("audio_rel_path", ""),
                "source_audio_path": sample.get("audio_path", ""),
                "speech_seconds": speech_seconds,
                "left_speech_seconds": left_speech,
                "right_speech_seconds": right_speech,
                "total_window_seconds": 2.0 * half_duration,
            }
        )
    return records


def select_top_balanced(samples: list[dict], fraction: float) -> list[dict]:
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[int(sample["label"])].append(sample)

    min_count = min(len(by_label[0]), len(by_label[1]))
    keep_per_label = max(1, int(math.floor(min_count * float(fraction))))

    selected = []
    for label in (0, 1):
        ranked = sorted(by_label[label], key=lambda x: float(x["speech_seconds"]), reverse=True)
        selected.extend(ranked[:keep_per_label])
    return sorted(selected, key=lambda x: (str(x.get("audio_rel_path", "")), int(x["label"])))


def build_loader(samples: list[dict], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        DistillationPairDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_pairs,
    )


def evaluate_model(model, samples: list[dict], device, args, desc: str) -> dict:
    loader = build_loader(samples, args.batch_size, args.num_workers)
    loss_fn = nn.CrossEntropyLoss()
    labels_all = []
    preds_all = []
    loss_sum = 0.0
    count = 0
    start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"].to(device)

            logits = model(left, right)
            if args.eval_tta_swap:
                logits = 0.5 * (logits + model(right, left))

            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_count = int(labels.numel())
            loss_sum += float(loss.item()) * batch_count
            count += batch_count
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())

    metrics = compute_metrics(labels_all, preds_all)
    metrics["loss"] = loss_sum / max(count, 1)
    metrics["eval_time_seconds"] = time.perf_counter() - start
    return metrics


def metric_delta(after, before):
    after = safe_float(after)
    before = safe_float(before)
    if after is None or before is None:
        return None
    return after - before


def summarize_subset(samples: list[dict]) -> dict:
    labels = [int(s["label"]) for s in samples]
    speech = [float(s["speech_seconds"]) for s in samples]
    return {
        "samples": len(samples),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "speech_seconds_mean": sum(speech) / max(len(speech), 1),
        "speech_seconds_min": min(speech) if speech else None,
        "speech_seconds_max": max(speech) if speech else None,
    }


def append_markdown_summary(md_path: Path, result_rows: list[dict], args) -> None:
    if not result_rows:
        return
    section_title = "## 7. Small Speech-Length Filter Check"

    lines = [
        "",
        section_title,
        "",
        (
            f"Filtered subset keeps the top {args.keep_fraction:.0%} longest estimated speech-duration "
            f"samples per label from val/test, preserving class balance. "
            f"Energy VAD: frame={args.frame_ms:g}ms, hop={args.hop_ms:g}ms, "
            f"threshold=max({args.min_rms:g}, {args.relative_rms:g}*max_rms)."
        ),
        "",
        "| Split | Model | Full ACC | Filtered ACC | Delta ACC | Full F1 | Filtered F1 | Delta F1 | Full Speech(s) | Filtered Speech(s) | Samples Full -> Filtered |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    split_order = {"val": 0, "test": 1}
    sorted_rows = sorted(
        result_rows,
        key=lambda r: (
            split_order.get(str(r.get("split", "")), 99),
            -(safe_float(r.get("filtered_acc")) or -1.0),
            str(r.get("model", "")),
        ),
    )
    for split_name in ("val", "test"):
        for row in [r for r in sorted_rows if r.get("split") == split_name]:
            lines.append(
                "| "
                f"{split_name} | {row['model']} | {fmt(row['full_acc'])} | {fmt(row['filtered_acc'])} | "
                f"{fmt(row['delta_acc'])} | {fmt(row['full_f1'])} | {fmt(row['filtered_f1'])} | "
                f"{fmt(row['delta_f1'])} | {fmt(row['full_speech_seconds_mean'])} | "
                f"{fmt(row['filtered_speech_seconds_mean'])} | "
                f"{row['full_samples']} -> {row['filtered_samples']} |"
            )

    lines.append("")
    existing = ""
    if md_path.exists():
        existing = md_path.read_text(encoding="utf-8")
        marker = "\n" + section_title
        pos = existing.find(marker)
        if pos >= 0:
            existing = existing[:pos].rstrip() + "\n"

    prefix = existing.rstrip() if existing else ""
    final_text = (prefix + "\n" if prefix else "") + "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_text)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan val/test samples by estimated speech length, keep the balanced top half, "
            "and evaluate compare/train.py checkpoints on full vs filtered data."
        )
    )
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--runs-csv", default="dl_model/compare/output/summary_manifest_runs.csv")
    parser.add_argument("--checkpoint-field", default="best_checkpoint")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--half-duration", type=float, default=2.0)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--student-device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-tta-swap", action="store_true", default=True)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--relative-rms", type=float, default=0.10)
    parser.add_argument("--min-rms", type=float, default=0.005)
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
    parser.add_argument("--scan-csv", default="dl_model/compare/speech_length_filter/output/speech_length_scan.csv")
    parser.add_argument("--result-csv", default="dl_model/compare/speech_length_filter/output/filtered_eval_results.csv")
    parser.add_argument(
        "--md-path",
        default="dl_model/compare/output/report_ppt_compare_filtered_summary_2026-04-20.md",
    )
    parser.add_argument("--append-md", action="store_true", default=True)
    parser.add_argument("--no-append-md", dest="append_md", action="store_false")
    parser.add_argument("--scan-only", action="store_true", help="Only write the speech-length scan CSV.")
    parser.add_argument("--md-only", action="store_true", help="Only rewrite the markdown section from result CSV.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    if args.md_only:
        result_csv = resolve_path(root, args.result_csv)
        if not result_csv.exists():
            raise FileNotFoundError(f"Result CSV not found: {result_csv}")
        append_markdown_summary(resolve_path(root, args.md_path), read_csv_rows(result_csv), args)
        print(f"md_updated : {resolve_path(root, args.md_path)}")
        return

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    splits = build_train_val_test_samples(
        manifest_csv=resolve_path(root, args.manifest_csv),
        dataset_root=resolve_path(root, args.dataset_root),
        target_sr=args.sr,
        include_time_windows=False,
    )

    eval_splits = {"val": splits["val"], "test": splits["test"]}
    scan_rows = []
    filtered_splits = {}
    subset_summary = {}
    for split_name, samples in eval_splits.items():
        scan_rows.extend(attach_audio_and_speech_lengths(samples, split_name, args))
        filtered_splits[split_name] = select_top_balanced(samples, args.keep_fraction)
        subset_summary[(split_name, "full")] = summarize_subset(samples)
        subset_summary[(split_name, "filtered")] = summarize_subset(filtered_splits[split_name])

    write_csv(resolve_path(root, args.scan_csv), scan_rows)
    if args.scan_only:
        print(f"scan_csv   : {resolve_path(root, args.scan_csv)}")
        print("scan_only  : true")
        return

    run_rows = read_csv_rows(resolve_path(root, args.runs_csv))
    if args.models:
        wanted = {normalize_model_name(x) for x in args.models}
        run_rows = [r for r in run_rows if normalize_model_name(r.get("model", "")) in wanted]

    result_rows = []
    for row in run_rows:
        model_name = normalize_model_name(row.get("model", ""))
        ckpt_raw = row.get(args.checkpoint_field, "")
        ckpt_path = resolve_path(root, ckpt_raw) if ckpt_raw else None
        if not ckpt_path or not ckpt_path.exists():
            print(f"[warn] skip {model_name}: checkpoint not found: {ckpt_raw}")
            continue

        try:
            state_dict, ckpt_model_name = load_checkpoint_state(ckpt_path, device)
            if ckpt_model_name:
                model_name = normalize_model_name(ckpt_model_name)
            if model_name not in MODEL_MODULES and model_name != "final_model":
                print(f"[warn] skip unsupported model: {model_name}")
                continue

            model = import_builder(model_name)(args).to(device)
            model.load_state_dict(state_dict, strict=True)

            for split_name in ("val", "test"):
                full_samples = eval_splits[split_name]
                filtered_samples = filtered_splits[split_name]
                full_metrics = evaluate_model(
                    model,
                    full_samples,
                    device,
                    args,
                    desc=f"{model_name} {split_name} full",
                )
                filtered_metrics = evaluate_model(
                    model,
                    filtered_samples,
                    device,
                    args,
                    desc=f"{model_name} {split_name} filtered",
                )
                full_info = subset_summary[(split_name, "full")]
                filtered_info = subset_summary[(split_name, "filtered")]

                out = {
                    "model": row.get("model", model_name),
                    "run": row.get("run", 1),
                    "split": split_name,
                    "half_duration": float(args.half_duration),
                    "keep_fraction": float(args.keep_fraction),
                    "full_samples": full_info["samples"],
                    "filtered_samples": filtered_info["samples"],
                    "full_positives": full_info["positives"],
                    "filtered_positives": filtered_info["positives"],
                    "full_negatives": full_info["negatives"],
                    "filtered_negatives": filtered_info["negatives"],
                    "full_speech_seconds_mean": full_info["speech_seconds_mean"],
                    "filtered_speech_seconds_mean": filtered_info["speech_seconds_mean"],
                    "full_acc": full_metrics.get("accuracy"),
                    "filtered_acc": filtered_metrics.get("accuracy"),
                    "delta_acc": metric_delta(filtered_metrics.get("accuracy"), full_metrics.get("accuracy")),
                    "full_f1": full_metrics.get("f1"),
                    "filtered_f1": filtered_metrics.get("f1"),
                    "delta_f1": metric_delta(filtered_metrics.get("f1"), full_metrics.get("f1")),
                    "full_precision": full_metrics.get("precision"),
                    "filtered_precision": filtered_metrics.get("precision"),
                    "delta_precision": metric_delta(filtered_metrics.get("precision"), full_metrics.get("precision")),
                    "full_recall": full_metrics.get("recall"),
                    "filtered_recall": filtered_metrics.get("recall"),
                    "delta_recall": metric_delta(filtered_metrics.get("recall"), full_metrics.get("recall")),
                    "full_loss": full_metrics.get("loss"),
                    "filtered_loss": filtered_metrics.get("loss"),
                    "delta_loss": metric_delta(filtered_metrics.get("loss"), full_metrics.get("loss")),
                    "full_eval_time_seconds": full_metrics.get("eval_time_seconds"),
                    "filtered_eval_time_seconds": filtered_metrics.get("eval_time_seconds"),
                    "best_checkpoint": ckpt_raw,
                }
                result_rows.append(out)
                print(
                    f"{model_name},{split_name}: "
                    f"acc {fmt(out['full_acc'])}->{fmt(out['filtered_acc'])} "
                    f"({fmt(out['delta_acc'])}), "
                    f"f1 {fmt(out['full_f1'])}->{fmt(out['filtered_f1'])} "
                    f"({fmt(out['delta_f1'])})"
                )
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"[warn] {model_name} failed: {type(exc).__name__}: {exc}")
        finally:
            if "model" in locals():
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result_csv = resolve_path(root, args.result_csv)
    write_csv(result_csv, result_rows)
    if args.append_md:
        append_markdown_summary(resolve_path(root, args.md_path), result_rows, args)

    print(f"scan_csv   : {resolve_path(root, args.scan_csv)}")
    print(f"result_csv : {result_csv}")
    if args.append_md:
        print(f"md_updated : {resolve_path(root, args.md_path)}")


if __name__ == "__main__":
    main()
