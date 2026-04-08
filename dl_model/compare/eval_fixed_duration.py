import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.dataloader import DistillationPairDataset, build_samples_from_new_extracted, collate_audio_pairs
from dl_model.compare.shared import evaluate_student, set_seed


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


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_summary_rows(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "runs" in payload:
        return payload["runs"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported summary format: {summary_path}")


def load_sidecar_args(record_path):
    if not record_path:
        return {}
    path = Path(record_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("args", {})


def load_checkpoint_payload(checkpoint_path, map_location):
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("args", {}), payload.get("model_name")
    if isinstance(payload, dict):
        return payload, {}, None
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_namespace(saved_args, fallback_args):
    merged = dict(fallback_args)
    merged.update(saved_args)
    return argparse.Namespace(**merged)


def resolve_checkpoint_field(kind):
    field = f"{kind}_checkpoint"
    if kind == "final":
        field = "final_checkpoint"
    return field


def resolve_record_field(kind):
    if kind == "final":
        return None
    return f"{kind}_record"


def build_test_loader(root, csv_rel_path, audio_rel_path, sr, half_duration, batch_size, num_workers):
    test_samples = build_samples_from_new_extracted(
        root / csv_rel_path,
        root / audio_rel_path,
        target_sr=sr,
        split="test",
    )
    for sample in test_samples:
        sample["half_duration"] = half_duration
        sample["teacher_prob"] = 0.5

    dataset = DistillationPairDataset(test_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_pairs,
    )
    return dataset, loader


def build_test_loader_cache(rows, durations, root, cli_args):
    cache = {}
    _ = rows  # keep signature stable; cache is intentionally CLI-driven for consistency.
    sr = int(cli_args.sr)
    test_csv = cli_args.new_test_csv
    test_audio_dir = cli_args.new_test_audio_dir
    batch_size = int(cli_args.batch_size)
    num_workers = int(cli_args.num_workers)
    for duration in durations:
        cache_key = (str(test_csv), str(test_audio_dir), sr, batch_size, num_workers, float(duration))
        if cache_key not in cache:
            cache[cache_key] = build_test_loader(
                root,
                test_csv,
                test_audio_dir,
                sr,
                duration,
                batch_size,
                num_workers,
            )
    return cache


def evaluate_one_checkpoint(row, checkpoint_kind, duration, root, device, cli_args, test_loader_cache):
    checkpoint_field = resolve_checkpoint_field(checkpoint_kind)
    checkpoint_rel = row.get(checkpoint_field)
    if not checkpoint_rel:
        raise ValueError(f"Missing {checkpoint_field} for model {row.get('model')}")

    checkpoint_path = root / checkpoint_rel
    state_dict, checkpoint_args, checkpoint_model_name = load_checkpoint_payload(checkpoint_path, map_location=device)

    record_field = resolve_record_field(checkpoint_kind)
    sidecar_args = load_sidecar_args(root / row[record_field]) if record_field and row.get(record_field) else {}
    saved_args = sidecar_args or checkpoint_args
    model_name = checkpoint_model_name or row["model"]

    if model_name not in MODEL_MODULES:
        raise ValueError(f"Unsupported model name in checkpoint: {model_name}")

    model_args = build_namespace(
        saved_args,
        {
            "sr": cli_args.sr,
            "n_mels": cli_args.n_mels,
            "emb_dim": cli_args.emb_dim,
            "student_channels": cli_args.student_channels,
            "ecapa_channels": cli_args.ecapa_channels,
            "redimnet_channels": cli_args.redimnet_channels,
            "sinc_channels": cli_args.sinc_channels,
            "final_model_weight_path": cli_args.final_model_weight_path,
            "dropout": cli_args.dropout,
            "time_mask_max": cli_args.time_mask_max,
            "freq_mask_max": cli_args.freq_mask_max,
        },
    )

    sr = int(getattr(model_args, "sr", cli_args.sr))
    test_csv = cli_args.new_test_csv
    test_audio_dir = cli_args.new_test_audio_dir
    batch_size = int(cli_args.batch_size)
    num_workers = int(cli_args.num_workers)
    eval_tta_swap = bool(getattr(model_args, "eval_tta_swap", cli_args.eval_tta_swap))

    cache_key = (str(test_csv), str(test_audio_dir), sr, batch_size, num_workers, float(duration))
    _, test_loader = test_loader_cache[cache_key]

    model = import_builder(model_name)(model_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ce_loss_fn = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=eval_tta_swap)
    test_time_seconds = time.perf_counter() - t0
    return {
        "model": row["model"],
        "run": row.get("run", 1),
        "checkpoint_kind": checkpoint_kind,
        "duration_seconds": duration,
        "half_duration_seconds": duration,
        "test_time_seconds": test_time_seconds,
        "test_acc": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_loss": metrics["loss"],
        "sample_count": metrics["sample_count"],
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate compare checkpoints on fixed per-side test durations around the midpoint split.")
    parser.add_argument("--summary-path", default="dl_model/compare/output/summary_random_duration.json")
    parser.add_argument("--checkpoint-kinds", nargs="+", default=["best_acc"], choices=["best_acc", "best_f1", "final"])
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-path", default="dl_model/compare/output/eval_fixed_duration_new_csv.json")
    parser.add_argument("--new-test-csv", default="dl_model/csv2/baseline_train_test_segments_switchlingua_seame.csv")
    parser.add_argument("--new-test-audio-dir", default="datasets/train_test2/test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-tta-swap", action="store_true", default=True)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
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
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    summary_path = root / args.summary_path
    rows = load_summary_rows(summary_path)
    if args.models:
        allowed = set(args.models)
        rows = [row for row in rows if row.get("model") in allowed]

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    set_seed(args.seed)
    results = []
    print(f"Using device: {device}")
    print(f"Summary: {summary_path}")
    print("Duration semantics: per-side duration on the 4s extracted clip, split at the midpoint.")
    test_loader_cache = build_test_loader_cache(rows, args.durations, root, args)
    print(f"Prepared {len(test_loader_cache)} fixed-duration test loader(s).")

    for row in rows:
        for checkpoint_kind in args.checkpoint_kinds:
            for duration in args.durations:
                result = evaluate_one_checkpoint(
                    row,
                    checkpoint_kind,
                    duration,
                    root,
                    device,
                    args,
                    test_loader_cache,
                )
                results.append(result)
                print(
                    f"{result['model']} [{checkpoint_kind}] duration={duration:.2f}s | "
                    f"test_acc={result['test_acc']:.4f} test_f1={result['test_f1']:.4f} "
                    f"test_time={result['test_time_seconds']:.3f}s"
                )

    output_path = root / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    write_csv(output_path.with_suffix(".csv"), results)
    print(f"\nSaved evaluation to: {output_path}")


if __name__ == "__main__":
    main()
