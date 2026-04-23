"""Compare all models (individual + ensemble) on acc / f1 / prec / rec / time_s.

Usage
-----
python -m baseline.compare_all_models                          # defaults
python -m baseline.compare_all_models --results-dir baseline/results_switchlingua_seame
python -m baseline.compare_all_models --output baseline/results_switchlingua_seame/all_models_comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.common import compute_metrics
from baseline.ensemble_calibrated_stacking import (
    load_predictions,
    load_summary,
    calibrate_crossval,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _metrics_row(name: str, kind: str, labels, preds, time_s: float) -> dict:
    m = compute_metrics(list(labels), list(preds))
    return {
        "method": name,
        "kind": kind,
        "acc": m["accuracy"],
        "f1": m["f1"],
        "prec": m["precision"],
        "rec": m["recall"],
        "time_s": time_s,
    }


def _hard_vote(voters: list[dict], common_indices: list[int]) -> tuple[list, list, float]:
    t0 = time.perf_counter()
    labels, preds = [], []
    for idx in common_indices:
        labels.append(voters[0]["preds"][idx]["label"])
        votes = [0.0, 0.0]
        for v in voters:
            votes[v["preds"][idx]["prediction"]] += 1.0
        preds.append(0 if votes[0] >= votes[1] else 1)
    return labels, preds, time.perf_counter() - t0


def _platt_ensemble(
    voters: list[dict],
    common_indices: list[int],
    labels_arr: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[list, list, float]:
    t0 = time.perf_counter()
    cal_cols = []
    for v in voters:
        raw = np.array(
            [v["preds"][i]["same_speaker_score"] for i in common_indices], dtype=np.float64
        )
        cal, _ = calibrate_crossval(raw, labels_arr, "platt", n_folds, seed=seed)
        cal_cols.append(cal)
    avg = np.mean(cal_cols, axis=0)
    labels = [voters[0]["preds"][i]["label"] for i in common_indices]
    preds = [1 if s >= 0.5 else 0 for s in avg]
    return labels, preds, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# main comparison
# ---------------------------------------------------------------------------

def compare_all(
    results_dir: Path,
    exclude: set[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    exclude = exclude or set()
    summary_path = results_dir / "summary.csv"
    summary = load_summary(summary_path)

    pred_files = sorted(results_dir.glob("*_predictions.csv"))
    voters: list[dict] = []
    for pred_file in pred_files:
        name = pred_file.stem.replace("_predictions", "")
        if name in exclude:
            continue
        if name not in summary:
            print(f"  [skip] {name}: not in summary.csv")
            continue
        preds = load_predictions(pred_file)
        voters.append({"name": name, "preds": preds, "summary": summary[name]})

    if not voters:
        raise RuntimeError("No eligible voter files found.")

    common_indices = sorted(
        set.intersection(*[set(v["preds"].keys()) for v in voters])
    )
    labels_arr = np.array(
        [voters[0]["preds"][i]["label"] for i in common_indices], dtype=np.float64
    )

    print(f"Models       : {[v['name'] for v in voters]}")
    print(f"Common rows  : {len(common_indices)}")

    rows: list[dict] = []

    # ---- individual models ----
    for v in voters:
        inf_time = v["summary"].get("inference_time_s") or 0.0
        labels = [v["preds"][i]["label"] for i in common_indices]
        preds  = [v["preds"][i]["prediction"] for i in common_indices]
        rows.append(_metrics_row(v["name"], "single", labels, preds, inf_time))

    # ---- ensemble: hard vote (unweighted majority) ----
    labels, preds, t = _hard_vote(voters, common_indices)
    rows.append(_metrics_row("ensemble_hard_vote", "ensemble", labels, preds, t))

    # ---- ensemble: Platt-calibrated average ----
    labels, preds, t = _platt_ensemble(voters, common_indices, labels_arr, n_folds, seed)
    rows.append(_metrics_row("ensemble_platt_calibrated", "ensemble", labels, preds, t))

    return rows


def print_table(rows: list[dict]):
    cols = ("method", "acc", "f1", "prec", "rec", "time_s")
    w = 36
    header = f"  {'Method':<{w}} {'acc':>7} {'f1':>7} {'prec':>7} {'rec':>7} {'time_s':>9}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    best_acc = max(r["acc"] for r in rows)
    for r in rows:
        star = "*" if r["acc"] == best_acc else " "
        print(
            f"{star} {r['method']:<{w}} "
            f"{r['acc']:>7.4f} {r['f1']:>7.4f} {r['prec']:>7.4f} {r['rec']:>7.4f} {r['time_s']:>9.3f}"
        )
    print(sep)


def save_results(rows: list[dict], out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_stem.with_suffix(".csv")
    json_path = out_stem.with_suffix(".json")
    fieldnames = ["method", "kind", "acc", "f1", "prec", "rec", "time_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2, ensure_ascii=False)
    print(f"Saved CSV  → {csv_path}")
    print(f"Saved JSON → {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare all models: acc / f1 / prec / rec / time_s")
    parser.add_argument("--results-dir", default="baseline/results_switchlingua_seame")
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=["speechbrain_xvector", "wespeaker_english"],
        help="Model names to exclude from comparison",
    )
    parser.add_argument("--calib-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output file stem (without extension). Both .csv and .json will be written.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / args.results_dir

    t0 = time.perf_counter()
    rows = compare_all(
        results_dir=results_dir,
        exclude=set(args.exclude_models or []),
        n_folds=args.calib_folds,
        seed=args.seed,
    )
    print_table(rows)
    print(f"\nTotal elapsed: {time.perf_counter() - t0:.2f} s")

    if args.output:
        save_results(rows, repo_root / args.output)


if __name__ == "__main__":
    main()
