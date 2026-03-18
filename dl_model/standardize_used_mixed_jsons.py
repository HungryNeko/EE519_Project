import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_used_json_list(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def normalize_dataset_path(raw_path: str, dataset_prefix: str) -> str:
    path = raw_path.replace("\\", "/")
    path_lower = path.lower()
    datasets_prefix = "datasets/"

    if datasets_prefix in path_lower:
        start = path_lower.index(datasets_prefix)
        rel_path = path[start:]
        parts = rel_path.split("/")
        if len(parts) >= 2:
            suffix = "/".join(parts[2:])
            return f"{dataset_prefix}/{suffix}" if suffix else dataset_prefix

    return f"{dataset_prefix}/{Path(path).name}"


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def standardize_span(span: dict) -> dict:
    return {
        "language": span.get("language"),
        "start": to_float(span.get("start", 0.0)),
        "end": to_float(span.get("end", 0.0)),
        "text": span.get("text", ""),
        "score": to_float(span.get("score")),
    }


def standardize_segment(segment: dict) -> dict:
    spans = [standardize_span(span) for span in segment.get("language_spans", [])]
    return {
        "segment_id": segment.get("segment_id"),
        "start": to_float(segment.get("start", 0.0)),
        "end": to_float(segment.get("end", 0.0)),
        "text": segment.get("text", ""),
        "scores": segment.get("scores", {}),
        "language_spans": spans,
    }


def standardize_item(item: dict, dataset_prefix: str) -> dict:
    path = normalize_dataset_path(item.get("path", ""), dataset_prefix)
    return {
        "path": path,
        "audio_name": Path(path).name,
        "whisper_language": item.get("whisper_language", "mixed"),
        "segments": [standardize_segment(segment) for segment in item.get("segments", [])],
    }


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_old{path.suffix}")


def save_backup(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def main():
    root = project_root()
    used_json_path = root / "dl_model" / "used_json.txt"
    json_paths = load_used_json_list(used_json_path)

    for rel_path in json_paths:
        json_path = root / rel_path
        dataset_prefix = str(rel_path.parent).replace("\\", "/")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        standardized = [standardize_item(item, dataset_prefix) for item in data]

        old_path = backup_path(json_path)
        save_backup(json_path, old_path)
        json_path.write_text(
            json.dumps(standardized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"standardized: {json_path.as_posix()}")
        print(f"backup_saved: {old_path.as_posix()}")
        print(f"records: {len(standardized)}")
        if standardized:
            print(f"sample_path: {standardized[0]['path']}")
            print(f"sample_audio_name: {standardized[0]['audio_name']}")


if __name__ == "__main__":
    main()
