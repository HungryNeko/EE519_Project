import argparse
import csv
import importlib
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.dataloader import DistillationPairDataset, collate_audio_pairs
from dl_model.old.speechbrain_ablation.shared import SincNetPairStudent
from datasets.train_test2.dataloader import build_train_val_test_samples


MODEL_MODULES = {
    "tdnn": "dl_model.compare.model_tdnn",
    "ecapatdnn": "dl_model.compare.model_escapetdnn",
    "escapetdnn": "dl_model.compare.model_escapetdnn",
    "redimnet": "dl_model.compare.model_redimnet",
    "sincnet": "dl_model.compare.model_sincnet",
}


def normalize_model_name(name):
    text = str(name).strip().lower()
    if text == "escapetdnn":
        return "ecapatdnn"
    return text


def import_builder(model_name):
    if model_name == "final_model":
        return lambda args: SincNetPairStudent(
            sample_rate=args.sr,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            sinc_channels=args.sinc_channels,
        )
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def resolve_repo_path(root: Path, maybe_rel: str):
    p = Path(str(maybe_rel))
    if not p.is_absolute():
        p = root / p
    return p


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows, existing_fieldnames):
    if not rows:
        return
    output_fields = list(existing_fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in output_fields:
                output_fields.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_state(ckpt_path: Path, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("model_name")
    if isinstance(payload, dict):
        return payload, None
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


def prepare_samples(samples, half_duration):
    for sample in samples:
        sample["half_duration"] = float(half_duration)
        sample["teacher_prob"] = 0.5


def build_loader(samples, batch_size, num_workers):
    return DataLoader(
        DistillationPairDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_pairs,
    )


def collect_scores(model, loader, device, use_tta_swap=False, score_type="logit_diff"):
    model.eval()
    labels_all = []
    scores_all = []
    with torch.no_grad():
        for batch in loader:
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"]

            logits = model(left, right)
            if use_tta_swap:
                logits = 0.5 * (logits + model(right, left))

            if score_type == "prob":
                scores = torch.softmax(logits, dim=1)[:, 1]
            elif score_type == "logit":
                scores = logits[:, 1]
            else:
                scores = logits[:, 1] - logits[:, 0]

            labels_all.extend(labels.cpu().tolist())
            scores_all.extend(scores.detach().cpu().tolist())
    return labels_all, scores_all


def compute_eer(labels, scores):
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
    for i in range(len(points) - 1):
        far1, frr1, _ = points[i]
        far2, frr2, _ = points[i + 1]
        diff1 = far1 - frr1
        diff2 = far2 - frr2
        if diff1 == 0.0:
            eer_interp = far1
            break
        if diff2 == 0.0:
            eer_interp = far2
            break
        if diff1 * diff2 < 0:
            alpha = diff1 / (diff1 - diff2)
            eer_interp = far1 + alpha * (far2 - far1)
            break

    if eer_interp is None:
        return float(eer_nearest), float(best_thr)
    return float(eer_interp), float(best_thr)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Temporary utility: compute speaker-verification EER for each run row and "
            "write into run-level + aggregate CSV."
        )
    )
    parser.add_argument("--runs-csv", default="dl_model/compare/output/summary_manifest_runs.csv")
    parser.add_argument("--aggregate-csv", default="dl_model/compare/output/summary_manifest_aggregate.csv")
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--half-duration", type=float, default=2.0)
    parser.add_argument("--checkpoint-field", default="best_checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-tta-swap", action="store_true", default=True)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--score-type", default="logit_diff", choices=["logit_diff", "logit", "prob"])
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
    runs_csv = resolve_repo_path(root, args.runs_csv)
    aggregate_csv = resolve_repo_path(root, args.aggregate_csv)

    run_rows, run_fields = load_csv(runs_csv)
    if not run_rows:
        raise RuntimeError(f"No rows in run csv: {runs_csv}")

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    splits = build_train_val_test_samples(
        manifest_csv=resolve_repo_path(root, args.manifest_csv),
        dataset_root=resolve_repo_path(root, args.dataset_root),
        target_sr=args.sr,
        include_time_windows=False,
    )
    split_samples = splits[args.split]
    if not split_samples:
        raise RuntimeError(f"No samples loaded for split={args.split}")
    prepare_samples(split_samples, args.half_duration)
    loader = build_loader(split_samples, args.batch_size, args.num_workers)

    eer_by_model = {}
    for row in run_rows:
        model_name = normalize_model_name(row.get("model", ""))
        ckpt_raw = str(row.get(args.checkpoint_field, "")).strip()
        if not ckpt_raw:
            print(f"[WARN] Skip row without checkpoint: model={row.get('model')} run={row.get('run')}")
            continue
        ckpt_path = resolve_repo_path(root, ckpt_raw)
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            continue

        state_dict, ckpt_model_name = load_checkpoint_state(ckpt_path, device)
        if ckpt_model_name:
            model_name = normalize_model_name(ckpt_model_name)
        if model_name not in MODEL_MODULES and model_name != "final_model":
            print(f"[WARN] Unsupported model={model_name}, skip")
            continue

        model = import_builder(model_name)(args).to(device)
        model.load_state_dict(state_dict, strict=True)

        labels, scores = collect_scores(
            model=model,
            loader=loader,
            device=device,
            use_tta_swap=args.eval_tta_swap,
            score_type=args.score_type,
        )
        eer, threshold = compute_eer(labels, scores)
        row["test_eer"] = eer
        row["test_eer_threshold"] = threshold

        model_key = normalize_model_name(row.get("model", model_name))
        eer_by_model.setdefault(model_key, []).append(eer)
        print(
            f"model={row.get('model', model_name)} run={row.get('run', 1)} "
            f"split={args.split} half={args.half_duration:.1f} eer={eer:.6f}"
        )

    write_csv(runs_csv, run_rows, run_fields)
    print(f"[DONE] Updated runs csv: {runs_csv}")

    agg_rows, agg_fields = load_csv(aggregate_csv)
    if agg_rows:
        for row in agg_rows:
            model_key = normalize_model_name(row.get("model", ""))
            values = eer_by_model.get(model_key, [])
            if not values:
                row["test_eer_mean"] = ""
                row["test_eer_std"] = ""
                continue
            mean, std = mean_std(values)
            row["test_eer_mean"] = mean
            row["test_eer_std"] = std
        write_csv(aggregate_csv, agg_rows, agg_fields)
        print(f"[DONE] Updated aggregate csv: {aggregate_csv}")
    else:
        print(f"[WARN] Aggregate csv has no rows: {aggregate_csv}")


if __name__ == "__main__":
    main()

