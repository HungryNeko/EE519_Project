"""
使用 mlp_train 数据集格式测试所有 baseline 模型
CSV 格式：audio_path,is_switch,split,left_start,switch_time,right_end
音频文件：datasets/mlp_train/{train,test}/{row_index}.wav
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from baseline.common import compute_metrics, load_eval_samples, preload_segment_pairs
from baseline.project_mlp_whisper import ProjectMLPWhisperBaseline
from baseline.resemblyzer_ge2e import ResemblyzerGE2EBaseline
from baseline.speechbrain_ecapa import SpeechBrainECAPABaseline
from baseline.speechbrain_xvector import SpeechBrainXVectorBaseline
from baseline.wespeaker_english import WeSpeakerEnglishBaseline


MODEL_REGISTRY = {
    "speechbrain_ecapa": SpeechBrainECAPABaseline,
    "speechbrain_xvector": SpeechBrainXVectorBaseline,
    # "resemblyzer_ge2e": ResemblyzerGE2EBaseline,  # 需要额外安装
    "wespeaker_english": WeSpeakerEnglishBaseline,
    # "project_mlp_whisper": ProjectMLPWhisperBaseline,  # 跳过
}


def load_mlp_train_samples(csv_path: Path, audio_dir: Path, split: str = "test"):
    """
    从 mlp_train 格式的 CSV 和音频目录加载测试样本
    CSV 格式：audio_path,is_switch,split,left_start,switch_time,right_end
    音频文件：{audio_dir}/{row_index}.wav (2 秒，从 left_start 到 right_end)
    """
    samples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row["split"].lower() != split:
                continue

            # 音频文件名：{row_index}.wav (2 秒音频)
            audio_file = audio_dir / f"{idx + 1}.wav"
            if not audio_file.exists():
                print(f"Warning: Audio not found: {audio_file}")
                continue

            sample = {
                "test_row_index": idx + 1,
                "audio_path": row["audio_path"],
                "audio_abs_path": str(audio_file.resolve()),  # 2 秒音频路径
                "label": 1 if row["is_switch"].lower() == "true" else 0,
            }
            samples.append(sample)

    return samples


def write_predictions_csv(output_path: Path, prediction_rows):
    fieldnames = [
        "test_row_index",
        "audio_path",
        "label",
        "prediction",
        "correct",
        "same_speaker_score",
        "raw_score",
        "runtime_ms",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)


def normalize_summary_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_summary_csv(output_path: Path, summary_rows):
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: normalize_summary_value(value) for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on mlp_train dataset"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which split to evaluate"
    )
    parser.add_argument(
        "--output",
        default="baseline/results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "dl_model" / "baseline_train_test_segments.csv"
    audio_dir = repo_root / "datasets" / "mlp_train" / args.split
    output_dir = repo_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"device: {device}")
    print(f"split: {args.split}")

    # 加载样本
    print("\nLoading samples...")
    samples = load_mlp_train_samples(csv_path, audio_dir, split=args.split)
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        raise RuntimeError("No samples found!")
    
    # 统计类别分布
    positives = sum(s["label"] for s in samples)
    negatives = len(samples) - positives
    print(f"positives (is_switch=True): {positives}")
    print(f"negatives (is_switch=False): {negatives}")

    # 运行所有模型
    summary_rows = []
    
    for model_name in MODEL_REGISTRY.keys():
        model_class = MODEL_REGISTRY[model_name]
        print(f"\n=== Running {model_name} ===")

        init_start = time.perf_counter()
        model = model_class(device=device, cache_dir=cache_dir)
        init_time_s = time.perf_counter() - init_start

        labels = []
        predictions = []
        prediction_rows = []
        inference_time_s = 0.0

        num_batches = len(samples)
        
        for i, sample in enumerate(samples):
            # audio_abs_path 指向的是 2 秒音频文件
            audio_path = Path(sample["audio_abs_path"])
            from baseline.common import load_audio
            
            wav, sr = load_audio(audio_path, sr=16000)
            
            # 所有模型都从 2 秒音频中间切开：左段 [0s, 1s], 右段 [1s, 2s]
            mid = len(wav) // 2
            left_audio = wav[:mid]
            right_audio = wav[mid:]

            step_start = time.perf_counter()
            result = model.predict(left_audio, right_audio, sr)
            step_time_s = time.perf_counter() - step_start
            inference_time_s += step_time_s

            label = sample["label"]
            prediction = int(result.prediction)
            labels.append(label)
            predictions.append(prediction)
            
            prediction_rows.append(
                {
                    "test_row_index": sample["test_row_index"],
                    "audio_path": sample["audio_path"],
                    "label": label,
                    "prediction": prediction,
                    "correct": int(label == prediction),
                    "same_speaker_score": f"{result.same_speaker_score:.6f}",
                    "raw_score": f"{result.raw_score:.6f}",
                    "runtime_ms": f"{step_time_s * 1000.0:.4f}",
                }
            )
            
            # 实时显示当前准确率
            if (i + 1) % 100 == 0 or (i + 1) == len(samples):
                curr_acc = sum(1 for l, p in zip(labels, predictions) if l == p) / len(labels)
                print(f"  [{i+1}/{len(samples)}] curr_acc={curr_acc:.4f}")

        metrics = compute_metrics(labels, predictions)
        total_model_time_s = init_time_s + inference_time_s
        avg_inference_ms = inference_time_s * 1000.0 / len(samples)

        summary = {
            "model": model_name,
            "sample_count": metrics["sample_count"],
            "positives": metrics["positives"],
            "negatives": metrics["negatives"],
            "accuracy": metrics["accuracy"],
            "positive_accuracy": metrics["positive_accuracy"],
            "negative_accuracy": metrics["negative_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "init_time_s": init_time_s,
            "inference_time_s": inference_time_s,
            "total_model_time_s": total_model_time_s,
            "avg_inference_ms": avg_inference_ms,
        }
        summary_rows.append(summary)

        acc_str = f"{summary['accuracy']:.4f}" if summary['accuracy'] else "NA"
        print(f"accuracy={acc_str} f1={summary['f1']:.4f} total_time={total_model_time_s:.2f}s avg_inference={avg_inference_ms:.2f}ms")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存结果
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for row in summary_rows:
        acc = row['accuracy'] if row['accuracy'] else 0
        f1 = row['f1'] if row['f1'] else 0
        time_s = row['total_model_time_s'] if row['total_model_time_s'] else 0
        avg_ms = row['avg_inference_ms'] if row['avg_inference_ms'] else 0
        print(f"{row['model']:25s} acc={acc:.4f} f1={f1:.4f} total_time={time_s:.2f}s avg_inference={avg_ms:.2f}ms")
    print(f"\nResults saved to: {output_dir}/summary.csv")


if __name__ == "__main__":
    main()
