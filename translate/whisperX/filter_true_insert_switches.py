import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LANG_EQUIV_GROUPS = [
    {"hi", "ur"},
]


def normalize_dataset_rel(path_str: str) -> str:
    p = str(path_str or "").replace("\\", "/").lower().strip()
    marker = "datasets/"
    idx = p.find(marker)
    if idx >= 0:
        return p[idx:]
    return p.lstrip("./")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON must be a list: {path}")
    return data


def load_ground_truth_insertions(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    gt: Dict[str, Dict[str, Any]] = {}
    for manifest_path in dataset_root.rglob("mixed_manifest.json"):
        rows = load_json_list(manifest_path)
        for row in rows:
            rel = normalize_dataset_rel(row.get("path", ""))
            if not rel:
                continue

            insert_at = row.get("insert_at_sec")
            insert_dur = row.get("insert_duration_sec")
            if not isinstance(insert_at, (int, float)) or not isinstance(insert_dur, (int, float)):
                continue

            target_lang = str(row.get("target_whisper_language") or "").lower()
            source_lang = str(row.get("insert_source_whisper_language") or "").lower()

            gt[rel] = {
                "insert_start_sec": float(insert_at),
                "insert_end_sec": float(insert_at) + float(insert_dur),
                "insert_duration_sec": float(insert_dur),
                "target_lang": target_lang,
                "insert_lang": source_lang,
            }
    return gt


def switch_lang_tuple(sw: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(sw.get("from_language") or "").lower(),
        str(sw.get("to_language") or "").lower(),
    )


def langs_equivalent(a: str, b: str) -> bool:
    a = (a or "").lower()
    b = (b or "").lower()
    if a == b:
        return True
    for g in LANG_EQUIV_GROUPS:
        if a in g and b in g:
            return True
    return False


def lang_pair_matches(actual: Tuple[str, str], expected: Tuple[str, str]) -> bool:
    return langs_equivalent(actual[0], expected[0]) and langs_equivalent(actual[1], expected[1])


def choose_best_switch(
    switches: List[Dict[str, Any]],
    expected_time: float,
    tolerance: float,
    expected_lang: Optional[Tuple[str, str]],
    allow_time_only_fallback: bool,
) -> Optional[Tuple[int, Dict[str, Any], float]]:
    candidates: List[Tuple[int, Dict[str, Any], float]] = []
    for i, sw in enumerate(switches):
        t = sw.get("switch_time")
        if not isinstance(t, (int, float)):
            continue
        dt = abs(float(t) - expected_time)
        if dt <= tolerance:
            candidates.append((i, sw, dt))

    if not candidates:
        return None

    if expected_lang and expected_lang[0] and expected_lang[1]:
        lang_match = [c for c in candidates if lang_pair_matches(switch_lang_tuple(c[1]), expected_lang)]
        if lang_match:
            return min(lang_match, key=lambda x: x[2])
        if not allow_time_only_fallback:
            return None

    return min(candidates, key=lambda x: x[2])


def filter_true_insert_switches(
    whisper_switch_rows: List[Dict[str, Any]],
    gt_map: Dict[str, Dict[str, Any]],
    tolerance_sec: float,
    allow_time_only_fallback: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for row in whisper_switch_rows:
        path_raw = row.get("path", "")
        rel = normalize_dataset_rel(path_raw)
        gt = gt_map.get(rel)
        if gt is None:
            continue

        switches = row.get("switch_positions", [])
        if not isinstance(switches, list):
            switches = []

        start_expected = gt["insert_start_sec"]
        end_expected = gt["insert_end_sec"]

        enter_lang = None
        exit_lang = None
        if gt["target_lang"] and gt["insert_lang"]:
            enter_lang = (gt["target_lang"], gt["insert_lang"])
            exit_lang = (gt["insert_lang"], gt["target_lang"])

        start_pick = choose_best_switch(
            switches=switches,
            expected_time=start_expected,
            tolerance=tolerance_sec,
            expected_lang=enter_lang,
            allow_time_only_fallback=allow_time_only_fallback,
        )

        used_idx = set()
        if start_pick is not None:
            used_idx.add(start_pick[0])

        end_candidates = [sw for i, sw in enumerate(switches) if i not in used_idx]
        end_pick_local = choose_best_switch(
            switches=end_candidates,
            expected_time=end_expected,
            tolerance=tolerance_sec,
            expected_lang=exit_lang,
            allow_time_only_fallback=allow_time_only_fallback,
        )

        end_pick = None
        if end_pick_local is not None:
            local_idx, sw, dt = end_pick_local
            # Map local index back to original index.
            kept_indices = [i for i in range(len(switches)) if i not in used_idx]
            if 0 <= local_idx < len(kept_indices):
                end_pick = (kept_indices[local_idx], sw, dt)
                used_idx.add(end_pick[0])

        true_switches: List[Dict[str, Any]] = []
        if start_pick is not None:
            true_switches.append(
                {
                    "type": "insert_start",
                    "expected_time": round(start_expected, 6),
                    "matched_time": round(float(start_pick[1]["switch_time"]), 6),
                    "time_error_sec": round(float(start_pick[2]), 6),
                    "from_language": start_pick[1].get("from_language"),
                    "to_language": start_pick[1].get("to_language"),
                    "from_segment_id": start_pick[1].get("from_segment_id"),
                    "to_segment_id": start_pick[1].get("to_segment_id"),
                }
            )

        if end_pick is not None:
            true_switches.append(
                {
                    "type": "insert_end",
                    "expected_time": round(end_expected, 6),
                    "matched_time": round(float(end_pick[1]["switch_time"]), 6),
                    "time_error_sec": round(float(end_pick[2]), 6),
                    "from_language": end_pick[1].get("from_language"),
                    "to_language": end_pick[1].get("to_language"),
                    "from_segment_id": end_pick[1].get("from_segment_id"),
                    "to_segment_id": end_pick[1].get("to_segment_id"),
                }
            )

        self_switches = [sw for i, sw in enumerate(switches) if i not in used_idx]

        out.append(
            {
                "path": path_raw,
                "insert_start_sec_gt": round(start_expected, 6),
                "insert_end_sec_gt": round(end_expected, 6),
                "insert_duration_sec_gt": round(gt["insert_duration_sec"], 6),
                "target_lang_gt": gt["target_lang"],
                "insert_lang_gt": gt["insert_lang"],
                "true_insert_switch_count": len(true_switches),
                "true_insert_switches": true_switches,
                "self_switch_count": len(self_switches),
                "self_switches": self_switches,
            }
        )

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Filter Whisper language switches by ground-truth insertion timeline from mixed_manifest.json, "
            "keeping only true insertion boundary switches and separating self code-switches."
        )
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/crossfade_insertions"),
        help="Root folder containing mixed_manifest.json and whisper switch json.",
    )
    p.add_argument(
        "--switch-json",
        type=Path,
        default=Path("datasets/crossfade_insertions/whisper_language_switch_crossfade_insertions.json"),
        help="Whisper switch summary json.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/crossfade_insertions/whisper_language_switch_crossfade_insertions_true_only.json"),
        help="Output json path.",
    )
    p.add_argument(
        "--tolerance-sec",
        type=float,
        default=0.8,
        help="Time tolerance for matching switch_time to insertion boundaries.",
    )
    p.add_argument(
        "--allow-time-only-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If True, when language-direction matching fails, fallback to time-only nearest switch. "
            "Default False avoids misclassifying self code-switches as insertion boundaries."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.tolerance_sec < 0:
        raise ValueError("--tolerance-sec must be >= 0")

    gt_map = load_ground_truth_insertions(args.dataset_root)
    whisper_rows = load_json_list(args.switch_json)

    filtered = filter_true_insert_switches(
        whisper_switch_rows=whisper_rows,
        gt_map=gt_map,
        tolerance_sec=float(args.tolerance_sec),
        allow_time_only_fallback=bool(args.allow_time_only_fallback),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    total = len(filtered)
    has_start = sum(
        1
        for r in filtered
        if any(s.get("type") == "insert_start" for s in r.get("true_insert_switches", []))
    )
    has_end = sum(
        1
        for r in filtered
        if any(s.get("type") == "insert_end" for s in r.get("true_insert_switches", []))
    )
    both = sum(1 for r in filtered if r.get("true_insert_switch_count", 0) >= 2)

    print(f"[Done] Wrote: {args.output_json.as_posix()}")
    print(f"[Stats] total={total} start_matched={has_start} end_matched={has_end} both_matched={both}")


if __name__ == "__main__":
    main()
