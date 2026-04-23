from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.common import compute_metrics


EPS = 1e-6


def _as_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _clip_probs(scores: np.ndarray) -> np.ndarray:
    return np.clip(scores, EPS, 1.0 - EPS)


def _score_to_base(scores: np.ndarray) -> np.ndarray:
    scores = _as_numpy(scores)
    if np.nanmin(scores) >= -0.05 and np.nanmax(scores) <= 1.05:
        p = _clip_probs(scores)
        return np.log(p / (1.0 - p))
    return scores


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = _as_numpy(probs)
    labels = _as_numpy(labels)
    return float(np.mean((probs - labels) ** 2))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = _as_numpy(probs)
    labels = _as_numpy(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    if total == 0:
        return 0.0

    val = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (probs >= lo) & (probs <= hi) if i == len(bins) - 2 else (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        val += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(val)


def print_metrics(name: str, metrics: dict):
    print(
        f"  {name:<36} acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
        f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
        f"bal_acc={metrics['balanced_accuracy']:.4f}"
    )


def load_summary(summary_path: Path) -> dict[str, dict]:
    models: dict[str, dict] = {}
    with open(summary_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["model"]
            models[name] = {k: float(v) if v else None for k, v in row.items() if k != "model"}
            models[name]["model"] = name
    return models


def load_predictions(pred_path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["test_row_index"])
            rows[idx] = {
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "same_speaker_score": float(row["same_speaker_score"]),
            }
    return rows


def fit_temperature(scores: np.ndarray, labels: np.ndarray, max_steps: int = 800, lr: float = 0.05) -> float:
    x = torch.tensor(_score_to_base(scores), dtype=torch.float32)
    y = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    if torch.unique(y).numel() < 2:
        return 1.0

    log_t = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([log_t], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(max_steps):
        opt.zero_grad()
        t = torch.exp(log_t).clamp_min(1e-3)
        loss = criterion(x / t, y)
        loss.backward()
        opt.step()

    return float(torch.exp(log_t).clamp_min(1e-3).item())


def apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    base = _score_to_base(scores)
    return _sigmoid(base / max(T, 1e-3))


def fit_platt(scores: np.ndarray, labels: np.ndarray, max_steps: int = 800, lr: float = 0.05) -> tuple[float, float]:
    x = torch.tensor(_score_to_base(scores), dtype=torch.float32)
    y = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    if torch.unique(y).numel() < 2:
        return 1.0, 0.0

    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([a, b], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(max_steps):
        opt.zero_grad()
        logits = a * x + b
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    return float(a.item()), float(b.item())


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    base = _score_to_base(scores)
    return _sigmoid(a * base + b)


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, method: str) -> dict:
    if method == "temperature":
        T = fit_temperature(scores, labels)
        return {"method": "temperature", "T": T}
    if method == "platt":
        a, b = fit_platt(scores, labels)
        return {"method": "platt", "a": a, "b": b}
    return {"method": "none"}


def apply_calibrator(scores: np.ndarray, params: dict) -> np.ndarray:
    method = params.get("method", "none")
    if method == "temperature":
        return apply_temperature(scores, float(params["T"]))
    if method == "platt":
        return apply_platt(scores, float(params["a"]), float(params["b"]))
    return _as_numpy(scores).copy()


def calibrate_crossval(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str,
    n_folds: int,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    scores = _as_numpy(scores)
    labels = _as_numpy(labels)

    n = len(scores)
    calibrated = np.zeros(n, dtype=np.float64)
    params_list: list[dict] = []

    if method == "none":
        return scores.copy(), [{"method": "none"}]

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    pos_idx = indices[labels[indices] == 1]
    neg_idx = indices[labels[indices] == 0]

    effective_folds = min(n_folds, len(pos_idx), len(neg_idx))
    if effective_folds < 2:
        params = fit_calibrator(scores, labels, method)
        return apply_calibrator(scores, params), [params]

    pos_folds = np.array_split(pos_idx, effective_folds)
    neg_folds = np.array_split(neg_idx, effective_folds)

    for k in range(effective_folds):
        val_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)

        if len(train_idx) == 0 or len(val_idx) == 0:
            calibrated[val_idx] = scores[val_idx]
            params_list.append({"method": "none"})
            continue

        params = fit_calibrator(scores[train_idx], labels[train_idx], method)
        calibrated[val_idx] = apply_calibrator(scores[val_idx], params)
        params_list.append(params)

    return calibrated, params_list


def calibrate_same(scores: np.ndarray, labels: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    if method == "none":
        return _as_numpy(scores).copy(), {"method": "none"}

    params = fit_calibrator(scores, labels, method)
    return apply_calibrator(scores, params), params


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def build_features_for_fold(
    voters: list[dict],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    feature_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_cols: list[np.ndarray] = []
    X_eval_cols: list[np.ndarray] = []
    y_train = np.array([voters[0]["preds"][i]["label"] for i in train_indices], dtype=np.float64)

    for v in voters:
        raw_train = np.array([v["preds"][i]["same_speaker_score"] for i in train_indices], dtype=np.float64)
        raw_eval = np.array([v["preds"][i]["same_speaker_score"] for i in eval_indices], dtype=np.float64)

        if feature_type == "score":
            X_train_cols.append(raw_train)
            X_eval_cols.append(raw_eval)

        elif feature_type in ("platt", "temperature"):
            params = fit_calibrator(raw_train, y_train, feature_type)
            X_train_cols.append(apply_calibrator(raw_train, params))
            X_eval_cols.append(apply_calibrator(raw_eval, params))

        elif feature_type == "both":
            params = fit_calibrator(raw_train, y_train, "platt")
            X_train_cols.append(raw_train)
            X_train_cols.append(apply_calibrator(raw_train, params))
            X_eval_cols.append(raw_eval)
            X_eval_cols.append(apply_calibrator(raw_eval, params))

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    X_train = np.stack(X_train_cols, axis=1).astype(np.float32)
    X_eval = np.stack(X_eval_cols, axis=1).astype(np.float32)
    return X_train, X_eval


class LogRegMeta(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPMeta(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        if self.training and self.dropout > 0:
            h = h * (torch.rand_like(h) > self.dropout) / (1 - self.dropout)
        h = torch.relu(self.fc2(h))
        if self.training and self.dropout > 0:
            h = h * (torch.rand_like(h) > self.dropout) / (1 - self.dropout)
        return self.fc3(h).squeeze(-1)


def train_meta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    lr: float,
    epochs: int,
    mlp_hidden: int,
    weight_decay: float,
    patience: int,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]
    model = LogRegMeta(n_features) if model_type == "logreg" else MLPMeta(n_features, mlp_hidden)

    mean, std = fit_standardizer(X_train)
    X_train = apply_standardizer(X_train, mean, std).astype(np.float32)
    X_val_std = apply_standardizer(X_val, mean, std).astype(np.float32) if X_val is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train.astype(np.float32))

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    model.train()
    for _ in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if X_val_std is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(X_val_std))
                val_loss = criterion(val_logits, torch.from_numpy(y_val.astype(np.float32))).item()
            model.train()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, mean, std


def predict_meta(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        return torch.sigmoid(logits).numpy()


def stratified_kfold_indices(labels: np.ndarray, n_folds: int, seed: int = 42):
    labels = _as_numpy(labels).astype(int)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError("Need at least 2 positive and 2 negative samples for stratified CV.")

    effective_folds = min(n_folds, len(pos_idx), len(neg_idx))
    if effective_folds < 2:
        raise ValueError("Not enough data for at least 2-fold stacking CV.")

    pos_shuffled = rng.permutation(pos_idx)
    neg_shuffled = rng.permutation(neg_idx)

    pos_folds = np.array_split(pos_shuffled, effective_folds)
    neg_folds = np.array_split(neg_shuffled, effective_folds)

    folds = []
    for k in range(effective_folds):
        val = np.concatenate([pos_folds[k], neg_folds[k]])
        train = np.concatenate(
            [pos_folds[j] for j in range(effective_folds) if j != k]
            + [neg_folds[j] for j in range(effective_folds) if j != k]
        )
        folds.append((train, val))
    return folds


def stacking_crossval(
    voters: list[dict],
    common_indices: list[int],
    labels_arr: np.ndarray,
    feature_type: str,
    meta_learner: str,
    n_folds: int,
    lr: float,
    epochs: int,
    mlp_hidden: int,
    weight_decay: float,
    patience: int,
    seed: int,
) -> tuple[list[int], list[int], list[float], list[dict]]:
    y_full = labels_arr.astype(np.float32)
    folds = stratified_kfold_indices(y_full, n_folds, seed=seed)

    oof_probs = np.zeros(len(common_indices), dtype=np.float64)
    fold_records: list[dict] = []

    for k, (train_idx, val_idx) in enumerate(folds):
        train_indices = np.array([common_indices[i] for i in train_idx], dtype=int)
        val_indices = np.array([common_indices[i] for i in val_idx], dtype=int)

        X_tr, X_val = build_features_for_fold(
            voters=voters,
            train_indices=train_indices,
            eval_indices=val_indices,
            feature_type=feature_type,
        )
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        model, sc_mean, sc_std = train_meta(
            X_train=X_tr,
            y_train=y_tr,
            model_type=meta_learner,
            lr=lr,
            epochs=epochs,
            mlp_hidden=mlp_hidden,
            weight_decay=weight_decay,
            patience=patience,
            X_val=X_val,
            y_val=y_val,
        )

        X_val_std = apply_standardizer(X_val, sc_mean, sc_std).astype(np.float32)
        oof_probs[val_idx] = predict_meta(model, X_val_std)

        if meta_learner == "logreg":
            w = model.linear.weight.detach().cpu().numpy().flatten().tolist()
            b = float(model.linear.bias.detach().cpu().item())
            fold_records.append({"fold": k, "weights": w, "bias": b})

    labels = [int(l) for l in y_full]
    binary_preds = [int(p >= 0.5) for p in oof_probs]
    return labels, binary_preds, oof_probs.tolist(), fold_records


def load_vote_files(results_dir: Path, summary_path: Path, weight_field: str, models, exclude_models):
    summary = load_summary(summary_path)
    exclude = set(exclude_models or [])

    pred_files = sorted(results_dir.glob("*_predictions.csv"))
    voters = []
    for pred_file in pred_files:
        model_name = pred_file.stem.replace("_predictions", "")
        if models and model_name not in models:
            continue
        if model_name in exclude:
            continue
        if model_name not in summary:
            print(f"  [skip] {model_name}: not in summary.csv")
            continue
        weight = summary[model_name].get(weight_field)
        if weight is None:
            print(f"  [skip] {model_name}: no {weight_field} in summary")
            continue
        preds = load_predictions(pred_file)
        voters.append({"name": model_name, "weight": weight, "preds": preds})

    if not voters:
        raise RuntimeError("No eligible voters found.")
    return voters


def _hard_vote(voters, common_indices):
    labels, preds = [], []
    for idx in common_indices:
        labels.append(voters[0]["preds"][idx]["label"])
        scores = [0.0, 0.0]
        for v in voters:
            scores[v["preds"][idx]["prediction"]] += v["weight"]
        preds.append(0 if scores[0] >= scores[1] else 1)
    return labels, preds


def _calibrated_vote(voters, common_indices, labels_arr, calib_method, calib_folds, seed):
    total_w = sum(v["weight"] for v in voters)
    cal_cols = []
    for v in voters:
        raw = np.array([v["preds"][i]["same_speaker_score"] for i in common_indices], dtype=np.float64)
        cal, _ = calibrate_crossval(raw, labels_arr, calib_method, calib_folds, seed=seed)
        cal_cols.append(cal * v["weight"])
    weighted_sum = sum(cal_cols) / total_w
    labels = [voters[0]["preds"][i]["label"] for i in common_indices]
    preds = [1 if s >= 0.5 else 0 for s in weighted_sum]
    return labels, preds


def run_all_methods(
    voters,
    common_indices,
    labels_arr,
    calib_folds: int,
    stack_folds: int,
    meta_epochs: int,
    meta_lr: float,
    meta_wd: float,
    meta_patience: int,
    mlp_hidden: int,
    seed: int,
):
    results = []

    for v in voters:
        t0 = time.perf_counter()
        y_true = [v["preds"][i]["label"] for i in common_indices]
        y_pred = [v["preds"][i]["prediction"] for i in common_indices]
        elapsed = time.perf_counter() - t0
        m = compute_metrics(y_true, y_pred)
        results.append({
            "method": f"{v['name']}",
            "acc": m["accuracy"],
            "f1": m["f1"],
            "prec": m["precision"],
            "rec": m["recall"],
            "bal_acc": m["balanced_accuracy"],
            "time_s": elapsed,
            "kind": "single",
        })

    t0 = time.perf_counter()
    labels, preds = _hard_vote(voters, common_indices)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(labels, preds)
    results.append({
        "method": "ensemble_hard_vote",
        "acc": m["accuracy"],
        "f1": m["f1"],
        "prec": m["precision"],
        "rec": m["recall"],
        "bal_acc": m["balanced_accuracy"],
        "time_s": elapsed,
        "kind": "ensemble",
    })

    t0 = time.perf_counter()
    labels, preds = _calibrated_vote(voters, common_indices, labels_arr, "platt", calib_folds, seed)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(labels, preds)
    results.append({
        "method": "ensemble_platt_calibrated",
        "acc": m["accuracy"],
        "f1": m["f1"],
        "prec": m["precision"],
        "rec": m["recall"],
        "bal_acc": m["balanced_accuracy"],
        "time_s": elapsed,
        "kind": "ensemble",
    })

    # stack_configs = [
    #     ("stacking_logreg_score", "logreg", "score"),
    #     ("stacking_logreg_both", "logreg", "both"),
    #     ("stacking_mlp_score", "mlp", "score"),
    # ]
    # for name, meta, feats in stack_configs:
    #     t0 = time.perf_counter()
    #     labels, preds, _, _ = stacking_crossval(
    #         voters=voters,
    #         common_indices=common_indices,
    #         labels_arr=labels_arr,
    #         feature_type=feats,
    #         meta_learner=meta,
    #         n_folds=stack_folds,
    #         lr=meta_lr,
    #         epochs=meta_epochs,
    #         mlp_hidden=mlp_hidden,
    #         weight_decay=meta_wd,
    #         patience=meta_patience,
    #         seed=seed,
    #     )
    #     elapsed = time.perf_counter() - t0
    #     m = compute_metrics(labels, preds)
    #     results.append({
    #         "method": name,
    #         "acc": m["accuracy"],
    #         "f1": m["f1"],
    #         "prec": m["precision"],
    #         "rec": m["recall"],
    #         "bal_acc": m["balanced_accuracy"],
    #         "time_s": elapsed,
    #         "kind": "ensemble",
    #     })

    return results


def print_comparison_table(results: list[dict]):
    print("\n" + "-" * 94)
    print(f"  {'Method':<34} {'acc':>7} {'f1':>7} {'prec':>7} {'rec':>7} {'time_s':>9}")
    print("-" * 94)
    for r in results:
        star = "*" if r["acc"] == max(x["acc"] for x in results) else " "
        print(
            f"{star} {r['method']:<34} "
            f"{r['acc']:>7.4f} {r['f1']:>7.4f} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['time_s']:>9.3f}"
        )
    print("-" * 94)


def main():
    parser = argparse.ArgumentParser(description="Calibration + stacking ensemble over saved baseline predictions.")
    parser.add_argument("--results-dir", default="baseline/results_switchlingua_seame")
    parser.add_argument("--weight-field", default="accuracy", choices=["accuracy", "f1", "balanced_accuracy"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--exclude-models", nargs="+", default=["speechbrain_xvector", "wespeaker_english"])
    parser.add_argument("--compare-all", action="store_true", default=False)
    parser.add_argument("--stack-folds", type=int, default=5)
    parser.add_argument("--meta-epochs", type=int, default=500)
    parser.add_argument("--meta-lr", type=float, default=0.05)
    parser.add_argument("--meta-weight-decay", type=float, default=1e-3)
    parser.add_argument("--meta-patience", type=int, default=30)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    total_t0 = time.perf_counter()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / args.results_dir
    summary_path = results_dir / "summary.csv"

    voters = load_vote_files(results_dir, summary_path, args.weight_field, args.models, args.exclude_models)
    common_indices = sorted(set.intersection(*[set(v["preds"].keys()) for v in voters]))
    labels_arr = np.array([voters[0]["preds"][i]["label"] for i in common_indices], dtype=np.float64)

    print(f"\nResults dir  : {results_dir}")
    print(f"Models       : {[v['name'] for v in voters]}")
    print(f"Samples      : {len(common_indices)}")
    print(f"\nRunning all methods...")

    all_results = run_all_methods(
        voters=voters,
        common_indices=common_indices,
        labels_arr=labels_arr,
        calib_folds=5,
        stack_folds=args.stack_folds,
        meta_epochs=args.meta_epochs,
        meta_lr=args.meta_lr,
        meta_wd=args.meta_weight_decay,
        meta_patience=args.meta_patience,
        mlp_hidden=args.mlp_hidden,
        seed=args.seed,
    )

    print_comparison_table(all_results)

    total_t1 = time.perf_counter()
    print(f"\n[Timing] total elapsed: {total_t1 - total_t0:.2f} s")

    if args.output:
        out_path = repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = out_path.with_suffix(".csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "kind", "acc", "f1", "prec", "rec", "bal_acc", "time_s"],
            )
            writer.writeheader()
            writer.writerows(all_results)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)

        print(f"Saved CSV  to: {csv_path}")
        print(f"Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()