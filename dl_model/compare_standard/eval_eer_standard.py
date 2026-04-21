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

from dl_model.compare.shared import set_seed
from dl_model.dataloader import DistillationPairDataset, collate_audio_pairs
from datasets.train_test2.dataloader import build_train_val_test_samples


MODEL_MODULES = {
    "tdnn": "dl_model.compare_standard.model_official_tdnn",
    "ecapatdnn": "dl_model.compare_standard.model_official_ecapatdnn",
    "resnet": "dl_model.compare_standard.model_official_resnet",
}


MD_STANDER_EXPERIMENTS = {
    "s1_standard_baseline": {
        "runs_csv": "dl_model/compare_standard/output_official/results_runs.csv",
        "half_durations": [2.0],
    },
    "s2_standard_1to2s": {
        "runs_csv": "dl_model/compare_standard/output_official/results_runs_1to2s.csv",
        "half_durations": [1.0, 1.5, 2.0],
    },
}


def import_builder(model_name):
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_model_name(name):
    return str(name).strip().lower()


def resolve_checkpoint(root: Path, ckpt_field: str):
    ckpt_path = Path(ckpt_field)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    return ckpt_path


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
        sample.pop("left_audio", None)
        sample.pop("right_audio", None)


def build_loader(samples, batch_size, num_workers):
    dataset = DistillationPairDataset(samples)
    return DataLoader(
        dataset,
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


def resolve_experiments(args):
    if args.mode == "single":
        return [
            {
                "name": "single",
                "runs_csv": args.runs_csv,
                "half_durations": [float(args.half_duration)],
            }
        ]

    names = args.experiments or list(MD_STANDER_EXPERIMENTS.keys())
    out = []
    for name in names:
        if name not in MD_STANDER_EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {name}")
        config = dict(MD_STANDER_EXPERIMENTS[name])
        config["name"] = name
        out.append(config)
    return out


def evaluate_experiment(experiment, rows, split_samples, root, args, device):
    results = []
    for row in rows:
        model_name = normalize_model_name(row.get("model", ""))
        ckpt_raw = row.get(args.checkpoint_field, "")
        if not ckpt_raw:
            continue

        ckpt_path = resolve_checkpoint(root, ckpt_raw)
        if not ckpt_path.exists():
            continue

        state_dict, ckpt_model_name = load_checkpoint_state(ckpt_path, device)
        if ckpt_model_name:
            model_name = normalize_model_name(ckpt_model_name)
        if model_name not in MODEL_MODULES:
            continue

        model = import_builder(model_name)(args).to(device)
        model.load_state_dict(state_dict, strict=True)

        for half_duration in experiment["half_durations"]:
            prepare_samples(split_samples, half_duration)
            loader = build_loader(split_samples, args.batch_size, args.num_workers)
            labels, scores = collect_scores(
                model=model,
                loader=loader,
                device=device,
                use_tta_swap=args.eval_tta_swap,
                score_type=args.score_type,
            )
            eer, _ = compute_eer(labels, scores)
            run_id = int(row.get("run", 1)) if str(row.get("run", "")).strip() else 1
            out_row = {
                "experiment": experiment["name"],
                "half_duration": float(half_duration),
                "model": row.get("model", model_name),
                "run": run_id,
                "eer": eer,
            }
            results.append(out_row)
            print(
                f"{out_row['experiment']},{out_row['model']},run={out_row['run']},"
                f"half={out_row['half_duration']:.1f},eer={out_row['eer']:.6f}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate standard checkpoints and output EER only. "
            "Default mode runs md summary standard experiments (S1/S2)."
        )
    )
    parser.add_argument("--mode", default="md_all", choices=["md_all", "single"])
    parser.add_argument("--experiments", nargs="+", default=None, choices=list(MD_STANDER_EXPERIMENTS.keys()))
    parser.add_argument("--runs-csv", default="dl_model/compare_standard/output_official/results_runs.csv")
    parser.add_argument("--half-duration", type=float, default=2.0)
    parser.add_argument("--manifest-csv", default="datasets/train_test2/compare_train_val_test_manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/train_test2")
    parser.add_argument("--checkpoint-field", default="best_checkpoint")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-tta-swap", action="store_true", default=False)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--score-type", default="logit_diff", choices=["logit_diff", "logit", "prob"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--student-channels", type=int, nargs="+", default=[128, 192, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--output-csv", default="dl_model/compare_standard/output_official/eer_standard_md_all.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    experiments = resolve_experiments(args)

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    splits = build_train_val_test_samples(
        manifest_csv=root / args.manifest_csv,
        dataset_root=root / args.dataset_root,
        target_sr=args.sr,
        include_time_windows=False,
    )
    if not splits["test"]:
        raise RuntimeError("Empty test split.")

    set_seed(args.seed)
    all_results = []
    for exp in experiments:
        runs_csv = root / exp["runs_csv"]
        rows = load_rows(runs_csv)
        if args.models:
            allowed = {normalize_model_name(m) for m in args.models}
            rows = [r for r in rows if normalize_model_name(r.get("model", "")) in allowed]
        if not rows:
            continue

        split_samples = splits["test"]
        exp_results = evaluate_experiment(
            experiment=exp,
            rows=rows,
            split_samples=split_samples,
            root=root,
            args=args,
            device=device,
        )
        all_results.extend(exp_results)

    if not all_results:
        raise RuntimeError("No model was evaluated. Check experiments/rows/checkpoints/model filters.")

    output_csv = root / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, all_results)


if __name__ == "__main__":
    main()
