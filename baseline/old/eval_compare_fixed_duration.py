import argparse
import csv
import gc
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dl_model.compare.shared import evaluate_student, set_seed
from dl_model.dataloader import DistillationPairDataset, build_samples_from_new_extracted, collate_audio_pairs
from baseline.common import compute_metrics


COMPARE_MODEL_MODULES = {
    "tdnn": "dl_model.compare.model_tdnn",
    "final_model": "dl_model.compare.model_final_model",
    "escapetdnn": "dl_model.compare.model_escapetdnn",
    "ecapatdnn": "dl_model.compare.model_escapetdnn",
    "redimnet": "dl_model.compare.model_redimnet",
    "sincnet": "dl_model.compare.model_sincnet",
    "sincnet_tdnn": "dl_model.compare.model_sincnet_tdnn",
}

BASELINE_MODEL_SPECS = {
    "pure_sincnet": ("baseline.pure_sincnet", "PureSincNetBaseline"),
    "distilled_mel_tdnn": ("baseline.distilled_mel_tdnn", "DistilledMelTDNNBaseline"),
    "project_mlp_whisper": ("baseline.project_mlp_whisper", "ProjectMLPWhisperBaseline"),
    "resemblyzer_ge2e": ("baseline.resemblyzer_ge2e", "ResemblyzerGE2EBaseline"),
    "speechbrain_ecapa": ("baseline.speechbrain_ecapa", "SpeechBrainECAPABaseline"),
    "speechbrain_xvector": ("baseline.speechbrain_xvector", "SpeechBrainXVectorBaseline"),
    "wespeaker_english": ("baseline.wespeaker_english", "WeSpeakerEnglishBaseline"),
    "microsoft_wavlm_base_plus_sv": (
        "baseline.microsoft_wavlm_base_plus_sv",
        "MicrosoftWavLMBasePlusSVBaseline",
    ),
    "pyannote_wespeaker_voxceleb_resnet34_lm": (
        "baseline.pyannote_wespeaker_voxceleb_resnet34_lm",
        "PyannoteWeSpeakerVoxCelebResnet34LMBaseline",
    ),
}

DEFAULT_COMPARE_MODELS = [
    "tdnn",
    "final_model",
    "escapetdnn",
    "ecapatdnn",
    "redimnet",
    "sincnet",
]
DEFAULT_BASELINE_MODELS_SPEECHBRAIN = [
    "pure_sincnet",
    "distilled_mel_tdnn",
    "speechbrain_ecapa",
    "speechbrain_xvector",
    "microsoft_wavlm_base_plus_sv",
]
OPTIONAL_BASELINE_MODELS = [
    "project_mlp_whisper",
    "resemblyzer_ge2e",
    "wespeaker_english",
    "pyannote_wespeaker_voxceleb_resnet34_lm",
]
ALL_MODEL_NAMES = list(COMPARE_MODEL_MODULES.keys()) + list(BASELINE_MODEL_SPECS.keys())


def import_builder(model_name):
    module = importlib.import_module(COMPARE_MODEL_MODULES[model_name])
    return module.build_model


def import_baseline_class(model_name):
    module_name, class_name = BASELINE_MODEL_SPECS[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_summary_rows(rows):
    summary = []
    for row in rows:
        if row.get("test_acc") is None:
            continue
        summary.append(
            {
                "duration": row.get("duration_seconds_per_side"),
                "model": row.get("model"),
                "acc": row.get("test_acc"),
                "f1": row.get("test_f1"),
                "prec": row.get("test_precision"),
                "rec": row.get("test_recall"),
            }
        )
    return summary


def load_summary_rows(summary_path):
    with open(summary_path, "r", encoding="utf-8-sig") as f:
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
    with open(path, "r", encoding="utf-8-sig") as f:
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
    return "final_checkpoint" if kind == "final" else f"{kind}_checkpoint"


def resolve_record_field(kind):
    return None if kind == "final" else f"{kind}_record"


def resolve_path(root, path_text):
    path_obj = Path(path_text)
    return path_obj if path_obj.is_absolute() else (root / path_obj)


def smart_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if all(str(key).startswith("module.") for key in state_dict.keys()):
        stripped = {str(key)[7:]: value for key, value in state_dict.items()}
        model.load_state_dict(stripped)
        return

    model.load_state_dict(state_dict)


def _alias_compare_model_names(model_name):
    if model_name == "ecapatdnn":
        return ["ecapatdnn", "escapetdnn"]
    if model_name == "escapetdnn":
        return ["escapetdnn", "ecapatdnn"]
    return [model_name]


def _infer_model_name_from_checkpoint_path(checkpoint_path):
    stem = Path(checkpoint_path).stem
    for suffix in ("_best_acc", "_best_f1", "_final"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _candidate_compare_models(row_model, checkpoint_model_name, checkpoint_path):
    raw_candidates = []
    if checkpoint_model_name:
        raw_candidates.append(str(checkpoint_model_name))
    if row_model:
        raw_candidates.append(str(row_model))
    hint_name = _infer_model_name_from_checkpoint_path(checkpoint_path)
    if hint_name:
        raw_candidates.append(hint_name)

    expanded = []
    for name in raw_candidates:
        for candidate in _alias_compare_model_names(name):
            if candidate in COMPARE_MODEL_MODULES and candidate not in expanded:
                expanded.append(candidate)
    return expanded


def ensure_test_loader(cache, root, csv_rel_path, audio_rel_path, sr, half_duration, batch_size, num_workers):
    cache_key = (
        str(csv_rel_path),
        str(audio_rel_path),
        int(sr),
        int(batch_size),
        int(num_workers),
        float(half_duration),
    )
    if cache_key in cache:
        return cache[cache_key]

    test_samples = build_samples_from_new_extracted(
        resolve_path(root, csv_rel_path),
        resolve_path(root, audio_rel_path),
        target_sr=int(sr),
        split="test",
    )
    for sample in test_samples:
        sample["half_duration"] = float(half_duration)
        sample["teacher_prob"] = 0.5

    if not test_samples:
        raise RuntimeError(
            "No test samples found. Please check --new-test-csv and --new-test-audio-dir paths."
        )

    dataset = DistillationPairDataset(test_samples)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_audio_pairs,
    )
    cache[cache_key] = (dataset, loader)
    return cache[cache_key]


def infer_final_model_weight_path(root, args):
    if args.final_model_weight_path:
        return str(resolve_path(root, args.final_model_weight_path))

    summary_path = resolve_path(root, args.summary_path)
    default_from_compare = summary_path.parent / "checkpoints_random_duration" / "sincnet_best_acc.pth"
    if default_from_compare.exists():
        return str(default_from_compare)

    return "dl_model/compare/output/checkpoints_random_duration/sincnet_best_acc.pth"


def evaluate_one_compare_checkpoint(row, checkpoint_kind, duration, root, device, cli_args, test_loader_cache):
    checkpoint_field = resolve_checkpoint_field(checkpoint_kind)
    checkpoint_rel = row.get(checkpoint_field)
    if not checkpoint_rel:
        raise ValueError(f"Missing {checkpoint_field} for model {row.get('model')}")

    checkpoint_path = resolve_path(root, checkpoint_rel)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict, checkpoint_args, checkpoint_model_name = load_checkpoint_payload(checkpoint_path, map_location=device)

    record_field = resolve_record_field(checkpoint_kind)
    sidecar_args = load_sidecar_args(resolve_path(root, row[record_field])) if record_field and row.get(record_field) else {}
    saved_args = sidecar_args or checkpoint_args
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
            "final_model_weight_path": cli_args.effective_final_model_weight_path,
            "dropout": cli_args.dropout,
            "time_mask_max": cli_args.time_mask_max,
            "freq_mask_max": cli_args.freq_mask_max,
        },
    )

    sr = int(getattr(model_args, "sr", cli_args.sr))
    eval_tta_swap = bool(getattr(model_args, "eval_tta_swap", cli_args.eval_tta_swap))
    _, test_loader = ensure_test_loader(
        cache=test_loader_cache,
        root=root,
        csv_rel_path=cli_args.new_test_csv,
        audio_rel_path=cli_args.new_test_audio_dir,
        sr=sr,
        half_duration=duration,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )

    candidates = _candidate_compare_models(
        row_model=row.get("model"),
        checkpoint_model_name=checkpoint_model_name,
        checkpoint_path=checkpoint_path,
    )
    if not candidates:
        raise ValueError(
            "Unable to infer compare model type from row/checkpoint metadata. "
            f"row_model={row.get('model')}, checkpoint_model_name={checkpoint_model_name}, checkpoint={checkpoint_path}"
        )

    load_errors = {}
    model = None
    selected_model_name = None
    for candidate in candidates:
        try:
            trial_model = import_builder(candidate)(model_args).to(device)
            smart_load_state_dict(trial_model, state_dict)
            trial_model.eval()
            model = trial_model
            selected_model_name = candidate
            break
        except Exception as exc:
            load_errors[candidate] = f"{type(exc).__name__}: {exc}"
            continue

    if model is None:
        raise RuntimeError(
            "Failed to load compare checkpoint with inferred model candidates: "
            f"{candidates}. First error: {load_errors.get(candidates[0], 'unknown')}"
        )

    ce_loss_fn = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=eval_tta_swap)
    test_time_seconds = time.perf_counter() - t0
    return {
        "model_family": "compare",
        "model": row["model"],
        "checkpoint_model_name": selected_model_name or "",
        "status": "ok",
        "run": row.get("run", 1),
        "checkpoint_kind": checkpoint_kind,
        "duration_seconds_per_side": float(duration),
        "test_time_seconds": test_time_seconds,
        "test_acc": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_loss": metrics["loss"],
        "sample_count": metrics["sample_count"],
        "checkpoint_path": str(checkpoint_path),
    }


def ensure_baseline_pairs(cache, root, csv_rel_path, audio_rel_path, sr, half_duration):
    cache_key = (str(csv_rel_path), str(audio_rel_path), int(sr), float(half_duration))
    if cache_key in cache:
        return cache[cache_key]

    samples = build_samples_from_new_extracted(
        resolve_path(root, csv_rel_path),
        resolve_path(root, audio_rel_path),
        target_sr=int(sr),
        split="test",
    )
    for sample in samples:
        sample["half_duration"] = float(half_duration)
        sample["teacher_prob"] = 0.5

    dataset = DistillationPairDataset(samples)
    pairs = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        pairs.append(
            {
                "left_audio": item["left_audio"],
                "right_audio": item["right_audio"],
                "sample_rate": int(item.get("target_sr", sr)),
                "label": int(item["label"]),
            }
        )

    if not pairs:
        raise RuntimeError(
            "No test samples found. Please check --new-test-csv and --new-test-audio-dir paths."
        )

    cache[cache_key] = pairs
    return pairs


def evaluate_one_baseline_model(model_name, duration, root, device, cli_args, baseline_pairs_cache, baseline_cache_dir):
    baseline_class = import_baseline_class(model_name)
    init_kwargs = {
        "device": str(device),
        "cache_dir": baseline_cache_dir,
    }
    if model_name == "pyannote_wespeaker_voxceleb_resnet34_lm":
        init_kwargs["hf_token"] = cli_args.hf_token

    init_t0 = time.perf_counter()
    model = baseline_class(**init_kwargs)
    init_time_s = time.perf_counter() - init_t0

    pairs = ensure_baseline_pairs(
        cache=baseline_pairs_cache,
        root=root,
        csv_rel_path=cli_args.new_test_csv,
        audio_rel_path=cli_args.new_test_audio_dir,
        sr=cli_args.sr,
        half_duration=duration,
    )

    labels = []
    predictions = []
    t0 = time.perf_counter()
    for pair in pairs:
        pred = model.predict(pair["left_audio"], pair["right_audio"], pair["sample_rate"])
        labels.append(pair["label"])
        predictions.append(int(pred.prediction))
    test_time_seconds = time.perf_counter() - t0

    metrics = compute_metrics(labels, predictions)
    return {
        "model_family": "baseline",
        "model": model_name,
        "checkpoint_model_name": model_name,
        "status": "ok",
        "run": 1,
        "checkpoint_kind": "baseline",
        "duration_seconds_per_side": float(duration),
        "test_time_seconds": test_time_seconds,
        "init_time_seconds": init_time_s,
        "test_acc": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_loss": None,
        "sample_count": metrics["sample_count"],
        "checkpoint_path": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate compare models on fixed per-side durations. Evaluation only, no training."
    )
    parser.add_argument("--summary-path", default="dl_model/compare/output/summary_random_duration.json")
    parser.add_argument("--checkpoint-kinds", nargs="+", default=["best_acc"], choices=["best_acc", "best_f1", "final"])
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_COMPARE_MODELS + DEFAULT_BASELINE_MODELS_SPEECHBRAIN,
        choices=ALL_MODEL_NAMES,
    )
    parser.add_argument(
        "--include-optional-baselines",
        action="store_true",
        help="Also include optional baseline models (project_mlp_whisper/resemblyzer/wespeaker/pyannote).",
    )
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-cpu", action="store_true", default=True)
    parser.add_argument("--no-force-cpu", dest="force_cpu", action="store_false")
    parser.add_argument("--output-path", default="baseline/results_compare/eval_compare_fixed_duration.csv")
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
    parser.add_argument("--final-model-weight-path", default=None)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--time-mask-max", type=int, default=12)
    parser.add_argument("--freq-mask-max", type=int, default=8)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--baseline-cache-dir", default="baseline/results_compare/model_cache")
    parser.add_argument("--strict-missing-models", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.include_optional_baselines:
        seen_models = set(args.models)
        for model_name in OPTIONAL_BASELINE_MODELS:
            if model_name not in seen_models:
                args.models.append(model_name)
                seen_models.add(model_name)

    root = Path(__file__).resolve().parents[1]
    summary_path = resolve_path(root, args.summary_path)
    all_rows = load_summary_rows(summary_path)

    requested_compare_models = [name for name in args.models if name in COMPARE_MODEL_MODULES]
    requested_baseline_models = [name for name in args.models if name in BASELINE_MODEL_SPECS]

    rows = [row for row in all_rows if row.get("model") in requested_compare_models]
    found_compare_models = {row.get("model") for row in rows}
    missing_compare_models = sorted(set(requested_compare_models) - found_compare_models)
    if missing_compare_models:
        msg = f"No summary rows for compare models: {', '.join(missing_compare_models)}"
        if args.strict_missing_models:
            raise RuntimeError(msg)
        print(f"[warn] {msg}. They will be skipped.")

    if args.force_cpu:
        device = torch.device("cpu")
    elif args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)

    args.effective_final_model_weight_path = infer_final_model_weight_path(root, args)
    baseline_cache_dir = resolve_path(root, args.baseline_cache_dir)
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    results = []
    test_loader_cache = {}
    baseline_pairs_cache = {}
    print(f"Using device: {device}")
    print(f"Summary path: {summary_path}")
    print(f"Models requested: {', '.join(args.models)}")
    print(f"Compare models: {', '.join(requested_compare_models) if requested_compare_models else '(none)'}")
    print(f"Baseline models: {', '.join(requested_baseline_models) if requested_baseline_models else '(none)'}")
    print(f"Test CSV: {args.new_test_csv}")
    print(f"Test audio dir: {args.new_test_audio_dir}")
    print("Duration semantics: per-side duration around midpoint split.")

    for row in rows:
        for checkpoint_kind in args.checkpoint_kinds:
            for duration in args.durations:
                try:
                    result = evaluate_one_compare_checkpoint(
                        row=row,
                        checkpoint_kind=checkpoint_kind,
                        duration=duration,
                        root=root,
                        device=device,
                        cli_args=args,
                        test_loader_cache=test_loader_cache,
                    )
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    error_message = f"{type(exc).__name__}: {exc}"
                    result = {
                        "model_family": "compare",
                        "model": row.get("model"),
                        "checkpoint_model_name": "",
                        "status": "failed",
                        "run": row.get("run", 1),
                        "checkpoint_kind": checkpoint_kind,
                        "duration_seconds_per_side": float(duration),
                        "test_time_seconds": 0.0,
                        "test_acc": None,
                        "test_f1": None,
                        "test_precision": None,
                        "test_recall": None,
                        "test_loss": None,
                        "sample_count": 0,
                        "checkpoint_path": row.get(resolve_checkpoint_field(checkpoint_kind), ""),
                    }
                    print(
                        f"[failed] {row.get('model')} [{checkpoint_kind}] duration={duration:.2f}s | "
                        f"{error_message}"
                    )
                    results.append(result)
                    continue

                results.append(result)
                print(
                    f"{result['model']} [{checkpoint_kind}] duration={duration:.2f}s | "
                    f"acc={result['test_acc']:.4f} f1={result['test_f1']:.4f} "
                    f"time={result['test_time_seconds']:.3f}s"
                )

    for model_name in requested_baseline_models:
        for duration in args.durations:
            try:
                result = evaluate_one_baseline_model(
                    model_name=model_name,
                    duration=duration,
                    root=root,
                    device=device,
                    cli_args=args,
                    baseline_pairs_cache=baseline_pairs_cache,
                    baseline_cache_dir=baseline_cache_dir,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                error_message = f"{type(exc).__name__}: {exc}"
                result = {
                    "model_family": "baseline",
                    "model": model_name,
                    "checkpoint_model_name": model_name,
                    "status": "failed",
                    "run": 1,
                    "checkpoint_kind": "baseline",
                    "duration_seconds_per_side": float(duration),
                    "test_time_seconds": 0.0,
                    "init_time_seconds": 0.0,
                    "test_acc": None,
                    "test_f1": None,
                    "test_precision": None,
                    "test_recall": None,
                    "test_loss": None,
                    "sample_count": 0,
                    "checkpoint_path": "",
                }
                print(f"[failed] {model_name} [baseline] duration={duration:.2f}s | {error_message}")
                results.append(result)
                continue

            results.append(result)
            print(
                f"{result['model']} [baseline] duration={duration:.2f}s | "
                f"acc={result['test_acc']:.4f} f1={result['test_f1']:.4f} "
                f"time={result['test_time_seconds']:.3f}s"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path = resolve_path(root, args.output_path)
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, to_summary_rows(results))
    print(f"\nSaved CSV evaluation to: {output_path}")


if __name__ == "__main__":
    main()
