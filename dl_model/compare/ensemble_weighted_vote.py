import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.common import compute_metrics
from dl_model.compare.shared import set_seed
from dl_model.dataloader import (
    DistillationPairDataset,
    build_samples_from_new_extracted,
    collate_audio_pairs,
    preload_audio_pairs,
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


def load_summary_runs(summary_path):
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
    p = Path(record_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("args", {})


def load_checkpoint_state(checkpoint_path, map_location):
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("args", {}), payload.get("model_name")
    if isinstance(payload, dict):
        return payload, {}, None
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_namespace(saved_args, fallback):
    merged = dict(fallback)
    merged.update(saved_args)
    return argparse.Namespace(**merged)


def checkpoint_field(kind):
    if kind == "final":
        return "final_checkpoint"
    return f"{kind}_checkpoint"


def record_field(kind):
    if kind == "final":
        return None
    return f"{kind}_record"


def weight_from_row(row, field):
    value = row.get(field)
    if value is None:
        return None
    return float(value)


def select_best_run_per_model(runs, metric_field):
    best_by_model = {}
    for row in runs:
        model = row["model"]
        metric = row.get(metric_field)
        if metric is None:
            continue
        if model not in best_by_model or float(metric) > float(best_by_model[model].get(metric_field, -1.0)):
            best_by_model[model] = row
    return list(best_by_model.values())


def build_test_loader(root, args, per_side_duration):
    samples = build_samples_from_new_extracted(
        root / args.new_test_csv,
        root / args.new_test_audio_dir,
        target_sr=args.sr,
        split="test",
    )
    for sample in samples:
        sample["half_duration"] = float(per_side_duration)
        sample["teacher_prob"] = 0.5
    if args.preload_audio:
        preload_audio_pairs(samples, limit=None)
    dataset = DistillationPairDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )
    return dataset, loader


def load_voters(root, args, runs, device):
    rows = select_best_run_per_model(runs, args.run_select_metric)
    if args.models:
        allow = set(args.models)
        rows = [row for row in rows if row["model"] in allow]

    exclude = {name.strip() for name in args.exclude_models}
    rows = [row for row in rows if row["model"] not in exclude]

    voters = []
    for row in rows:
        model_name = row["model"]
        if model_name not in MODEL_MODULES:
            continue

        ckpt_rel = row.get(checkpoint_field(args.checkpoint_kind))
        if not ckpt_rel:
            continue
        ckpt_path = root / ckpt_rel
        state_dict, checkpoint_args, checkpoint_model_name = load_checkpoint_state(ckpt_path, map_location=device)

        rec_field = record_field(args.checkpoint_kind)
        sidecar = load_sidecar_args(root / row[rec_field]) if rec_field and row.get(rec_field) else {}
        saved_args = sidecar or checkpoint_args
        effective_model_name = checkpoint_model_name or model_name
        if effective_model_name not in MODEL_MODULES:
            continue

        weight = weight_from_row(row, args.weight_field)
        if weight is None:
            continue

        model_args = build_namespace(
            saved_args,
            {
                "sr": args.sr,
                "n_mels": args.n_mels,
                "emb_dim": args.emb_dim,
                "student_channels": args.student_channels,
                "ecapa_channels": args.ecapa_channels,
                "redimnet_channels": args.redimnet_channels,
                "sinc_channels": args.sinc_channels,
                "final_model_weight_path": args.final_model_weight_path,
                "dropout": args.dropout,
                "time_mask_max": args.time_mask_max,
                "freq_mask_max": args.freq_mask_max,
            },
        )

        model = import_builder(effective_model_name)(model_args).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        voters.append(
            {
                "name": model_name,
                "model": model,
                "weight": float(weight),
                "eval_tta_swap": bool(getattr(model_args, "eval_tta_swap", args.eval_tta_swap)),
            }
        )
    return voters


def evaluate_weighted_voting(voters, loader, device):
    labels_all = []
    ensemble_preds = []
    per_model_preds = {v["name"]: [] for v in voters}

    with torch.no_grad():
        for batch in loader:
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"].to(device)

            vote_scores = torch.zeros((labels.size(0), 2), dtype=torch.float32, device=device)
            for voter in voters:
                logits = voter["model"](left, right)
                if voter["eval_tta_swap"]:
                    logits = 0.5 * (logits + voter["model"](right, left))
                preds = torch.argmax(logits, dim=1)
                per_model_preds[voter["name"]].extend(preds.detach().cpu().tolist())
                vote_scores[torch.arange(labels.size(0), device=device), preds] += voter["weight"]

            ens_pred = torch.argmax(vote_scores, dim=1)
            ensemble_preds.extend(ens_pred.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    model_metrics = {name: compute_metrics(labels_all, preds) for name, preds in per_model_preds.items()}
    ensemble_metrics = compute_metrics(labels_all, ensemble_preds)
    return labels_all, model_metrics, ensemble_metrics


def main():
    parser = argparse.ArgumentParser(description="Weighted voting ensemble experiment for compare models.")
    parser.add_argument("--summary-path", default="dl_model/compare/output/summary_random_duration.json")
    parser.add_argument("--checkpoint-kind", default="best_acc", choices=["best_acc", "best_f1", "final"])
    parser.add_argument("--run-select-metric", default="best_acc", choices=["best_acc", "best_f1", "best_f1_acc", "best_acc_f1"])
    parser.add_argument("--weight-field", default="best_acc", choices=["best_acc", "best_f1", "best_f1_acc", "best_acc_f1"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--exclude-models", nargs="+", default=["sincnet_tdnn"])
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--new-test-csv", default="dl_model/csv2/baseline_train_test_segments_switchlingua_seame.csv")
    parser.add_argument("--new-test-audio-dir", default="datasets/train_test2/test")
    parser.add_argument("--student-device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-audio", action="store_true", default=True)
    parser.add_argument("--no-preload-audio", dest="preload_audio", action="store_false")
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
    parser.add_argument("--output-path", default="dl_model/compare/output/ensemble_weighted_vote_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    summary_path = root / args.summary_path
    runs = load_summary_runs(summary_path)

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    set_seed(args.seed)
    voters = load_voters(root, args, runs, device)
    if not voters:
        raise RuntimeError("No eligible voters loaded. Check summary path / models / exclude-models.")

    print(f"Using device: {device}")
    print(f"Summary: {summary_path}")
    print(
        "Voters: "
        + ", ".join(f"{v['name']}(w={v['weight']:.4f})" for v in voters)
    )
    print("Voting rule: weighted hard vote by predicted class.")

    results = []
    for duration in args.durations:
        _, loader = build_test_loader(root, args, duration)
        _, model_metrics, ensemble_metrics = evaluate_weighted_voting(voters, loader, device)

        print(
            f"\nDuration={duration:.2f}s/side | "
            f"Ensemble acc={ensemble_metrics['accuracy']:.4f} f1={ensemble_metrics['f1']:.4f}"
        )
        for v in voters:
            m = model_metrics[v["name"]]
            print(
                f"  {v['name']}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} weight={v['weight']:.4f}"
            )

        results.append(
            {
                "duration_seconds_per_side": duration,
                "ensemble_metrics": ensemble_metrics,
                "model_metrics": {name: model_metrics[name] for name in model_metrics},
                "weights": {v["name"]: v["weight"] for v in voters},
                "exclude_models": args.exclude_models,
                "checkpoint_kind": args.checkpoint_kind,
                "weight_field": args.weight_field,
            }
        )

    output_path = root / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved ensemble report to: {output_path}")


if __name__ == "__main__":
    main()
