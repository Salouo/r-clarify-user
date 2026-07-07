from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "merged_result"
OUTPUT_PATH = SCRIPT_DIR / "human_time_comparison.pdf"
CONDITION_ORDER = ["R-Clarify", "w/o TUR"]


SECONDS_FIELDS = [
    "sample_duration_seconds",
    "total_duration_seconds",
    "duration_seconds",
    "elapsed_seconds",
    "interaction_time_seconds",
    "time_seconds",
]
MINUTES_FIELDS = [
    "duration_minutes",
    "total_duration_minutes",
    "interaction_time_minutes",
    "time_minutes",
]
TIMESTAMP_PAIRS = [
    ("sample_started_at", "sample_finished_at"),
    ("run_started_at", "run_finished_at"),
    ("started_at", "finished_at"),
    ("start_time", "end_time"),
    ("start", "end"),
]


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is JSONL but the line is not an object.")
            records.append(value)
    return records


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def union_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for record in records:
        keys.update(record.keys())
    return sorted(keys)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def condition_from_value(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "R-Clarify" if value else "w/o TUR"

    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "R-Clarify"
    if text in {"false", "no", "0"}:
        return "w/o TUR"
    if text in {"r-clarify", "reflection", "r_clarify", "r clarify"}:
        return "R-Clarify"
    if text in {"clarify", "without_reflection", "without-reflection", "without reflection", "w/o tur", "wo tur"}:
        return "w/o TUR"
    return None


def infer_file_condition(path: Path, payload: Any, records: list[dict[str, Any]]) -> str | None:
    candidates: set[str] = set()

    # File-level merged JSON metadata in this project:
    # r-clarify.json has mode="r-clarify", use_reflection=true, result_name="reflection.json".
    # clarify.json has mode="clarify", use_reflection=false, result_name="without_reflection.json".
    if isinstance(payload, dict):
        for field in ("mode", "condition", "variant", "result_name"):
            condition = condition_from_value(payload.get(field))
            if condition:
                candidates.add(condition)
        if "use_reflection" in payload:
            candidates.add(condition_from_value(payload.get("use_reflection")) or "")

    # Fall back to exact file names only. Avoid substring matching because "r-clarify"
    # also contains "clarify".
    stem = path.stem.lower()
    filename_condition = condition_from_value(stem)
    if filename_condition:
        candidates.add(filename_condition)

    # Flat CSV/JSONL files may carry condition/mode/use_reflection per row.
    for field in ("condition", "mode", "variant", "use_reflection"):
        values = {condition_from_value(record.get(field)) for record in records}
        values.discard(None)
        candidates.update(values)

    candidates.discard("")
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous condition in {path}. Candidates={sorted(candidates)}. "
            f"Available columns: {union_keys(records)}"
        )
    return next(iter(candidates), None)


def duration_from_timestamps(record: dict[str, Any]) -> tuple[float | None, str | None]:
    for start_field, end_field in TIMESTAMP_PAIRS:
        start = parse_time(record.get(start_field))
        end = parse_time(record.get(end_field))
        if start and end:
            return (end - start).total_seconds(), f"{start_field}/{end_field}"
    return None, None


def duration_field_reliable(
    records: list[dict[str, Any]],
    duration_field: str,
    *,
    duration_scale: float,
) -> bool:
    values = [parse_float(record.get(duration_field)) for record in records]
    values = [value for value in values if value is not None]
    if not values or any(value < 0 for value in values):
        return False

    # If timestamps are present, verify that explicit durations are consistent.
    # The stored human-study JSON rounds timestamps to seconds, so a <=2s mismatch
    # is accepted. If timestamps are absent, a positive numeric duration field is
    # considered reliable.
    checked = 0
    for record in records:
        explicit = parse_float(record.get(duration_field))
        timestamp_duration, _ = duration_from_timestamps(record)
        if explicit is None or timestamp_duration is None:
            continue
        checked += 1
        if abs(explicit * duration_scale - timestamp_duration) > 2.0:
            return False
    return True if checked else True


def choose_duration_source(records: list[dict[str, Any]], path: Path) -> tuple[str, float]:
    keys = set(union_keys(records))
    second_candidates = [field for field in SECONDS_FIELDS if field in keys]
    minute_candidates = [field for field in MINUTES_FIELDS if field in keys]

    # Prefer explicit duration fields over computing from timestamps. Within merged
    # human-study JSON, sample_duration_seconds is preferred over run totals because
    # it aggregates instance-level interaction time and avoids double-counting
    # overlapping resume runs.
    for field in second_candidates:
        if duration_field_reliable(records, field, duration_scale=1.0):
            return field, 1.0
    for field in minute_candidates:
        if duration_field_reliable(records, field, duration_scale=60.0):
            return field, 60.0

    has_timestamp_pair = any(start in keys and end in keys for start, end in TIMESTAMP_PAIRS)
    if has_timestamp_pair:
        return "timestamps", 1.0

    raise ValueError(
        f"No reliable time column found in {path}. Available columns: {union_keys(records)}"
    )


def duration_seconds(record: dict[str, Any], duration_source: str, scale: float) -> float:
    if duration_source == "timestamps":
        seconds, fields = duration_from_timestamps(record)
        if seconds is None:
            raise ValueError(f"Could not compute duration from timestamp fields in record: {record}")
        return seconds

    value = parse_float(record.get(duration_source))
    if value is None:
        raise ValueError(f"Missing numeric duration field '{duration_source}' in record: {record}")
    return value * scale


def has_any_time_field(records: list[dict[str, Any]]) -> bool:
    keys = set(union_keys(records))
    return bool(
        keys.intersection(SECONDS_FIELDS)
        or keys.intersection(MINUTES_FIELDS)
        or any(start in keys and end in keys for start, end in TIMESTAMP_PAIRS)
    )


def records_from_payload(path: Path, payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} contains a JSON list, but not every item is an object.")
        return payload

    if not isinstance(payload, dict):
        raise ValueError(f"{path} contains unsupported JSON structure: {type(payload).__name__}")

    results = payload.get("results")
    runs = payload.get("runs")

    # Prefer instance-level results when sample_duration_seconds is present.
    # This is the intended path for the current merged human-study files.
    if isinstance(results, list) and results and all(isinstance(item, dict) for item in results):
        result_keys = set(union_keys(results))
        if "sample_duration_seconds" in result_keys:
            return results

    if isinstance(runs, list) and runs and all(isinstance(item, dict) for item in runs):
        return runs

    if isinstance(results, list) and all(isinstance(item, dict) for item in results):
        return results

    raise ValueError(
        f"{path} does not contain usable records. Top-level keys: {sorted(payload.keys())}"
    )


def load_records(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    if path.suffix.lower() == ".csv":
        records = load_csv(path)
        return records, records
    if path.suffix.lower() == ".jsonl":
        records = load_jsonl(path)
        return records, records
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        return payload, records_from_payload(path, payload)
    raise ValueError(f"Unsupported file type: {path}")


def compute_totals() -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"records": 0, "seconds": 0.0})

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue

        payload, records = load_records(path)
        if not records:
            continue

        condition = infer_file_condition(path, payload, records)
        if condition is None:
            if has_any_time_field(records):
                raise ValueError(
                    f"Could not infer condition for {path}. Available columns: {union_keys(records)}"
                )
            print(f"Skipping unrelated file without time fields: {path}")
            continue

        duration_source, scale = choose_duration_source(records, path)
        for record in records:
            totals[condition]["records"] += 1
            totals[condition]["seconds"] += duration_seconds(record, duration_source, scale)

        print(f"Loaded {path.name}: condition={condition}, time_source={duration_source}")

    missing = [condition for condition in CONDITION_ORDER if condition not in totals]
    if missing:
        raise ValueError(f"Missing required condition(s): {missing}")
    return totals


def print_summary(totals: dict[str, dict[str, float]]) -> None:
    print()
    print(
        f"{'condition':<12} {'records':>8} {'total_seconds':>15} "
        f"{'total_minutes':>15} {'avg_seconds':>13} {'avg_minutes':>13}"
    )
    print("-" * 84)
    for condition in CONDITION_ORDER:
        records = int(totals[condition]["records"])
        seconds = totals[condition]["seconds"]
        minutes = seconds / 60
        avg_seconds = seconds / records
        avg_minutes = avg_seconds / 60
        print(
            f"{condition:<12} {records:>8d} {seconds:>15.3f} "
            f"{minutes:>15.3f} {avg_seconds:>13.3f} {avg_minutes:>13.3f}"
        )


def plot_totals(totals: dict[str, dict[str, float]]) -> None:
    labels = CONDITION_ORDER
    avg_minutes = [
        totals[label]["seconds"] / totals[label]["records"] / 60 for label in labels
    ]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars = ax.bar(
        labels,
        avg_minutes,
        width=0.52,
        color=["#4E79A7", "#F28E2B"],
        edgecolor="#2F3A45",
        linewidth=0.8,
    )

    ax.set_ylabel("Average Time per Instance (min)")
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

    y_max = max(avg_minutes) * 1.18
    ax.set_ylim(0, y_max)
    for bar, value in zip(bars, avg_minutes):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    totals = compute_totals()
    print_summary(totals)
    plot_totals(totals)
    print(f"\nSaved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
