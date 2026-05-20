from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("outputs/human_runs")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "merged_results"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def common_value(payloads: list[dict[str, Any]], key: str) -> Any:
    values = [payload.get(key) for payload in payloads if key in payload]
    if not values:
        return None

    first = values[0]
    if all(value == first for value in values):
        return first

    unique_values: dict[str, Any] = {}
    for value in values:
        unique_values.setdefault(
            json.dumps(value, ensure_ascii=False, sort_keys=True),
            value,
        )
    return [unique_values[key] for key in sorted(unique_values)]


def discover_result_names(input_dir: Path) -> list[str]:
    return sorted({path.name for path in input_dir.rglob("results/*.json")})


def discover_result_files(input_dir: Path, result_name: str) -> list[Path]:
    return sorted(path for path in input_dir.rglob(f"results/{result_name}") if path.is_file())


def run_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = path.parent.parent
    results = payload.get("results", [])
    sample_ids = payload.get("sample_ids")
    if sample_ids is None and isinstance(results, list):
        sample_ids = [
            record.get("sample_index")
            for record in results
            if isinstance(record, dict)
        ]

    return {
        "participant_id": payload.get("participant_id") or run_dir.name,
        "run_id": payload.get("run_id") or run_dir.name,
        "source_file": str(path),
        "n_processed": len(results) if isinstance(results, list) else 0,
        "n_samples": payload.get("n_samples"),
        "seed": payload.get("seed"),
        "sample_ids": sample_ids,
        "mode": payload.get("mode"),
        "use_reflection": payload.get("use_reflection"),
        "run_started_at": payload.get("run_started_at"),
        "run_finished_at": payload.get("run_finished_at"),
        "total_duration_seconds": payload.get("total_duration_seconds"),
    }


def duplicate_sample_index_report(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        sample_index = record.get("sample_index")
        if sample_index is None:
            continue
        by_index[sample_index].append(
            {
                "participant_id": record.get("participant_id"),
                "run_id": record.get("run_id"),
                "sample_position": record.get("sample_position"),
            }
        )

    report: list[dict[str, Any]] = []
    for sample_index, occurrences in by_index.items():
        if len(occurrences) <= 1:
            continue
        report.append(
            {
                "sample_index": sample_index,
                "count": len(occurrences),
                "occurrences": occurrences,
            }
        )
    return sorted(report, key=lambda item: item["sample_index"])


def merge_result_files(
    files: list[Path],
    *,
    input_dir: Path,
    result_name: str,
) -> dict[str, Any]:
    payloads = [load_json(path) for path in files]
    summaries = [run_summary(path, payload) for path, payload in zip(files, payloads)]
    results: list[dict[str, Any]] = []

    for path, payload, summary in zip(files, payloads, summaries):
        records = payload.get("results", [])
        if not isinstance(records, list):
            raise ValueError(f"Expected a list field named 'results': {path}")

        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Every result record must be a JSON object: {path}")
            merged_record = deepcopy(record)
            merged_record.setdefault("participant_id", summary["participant_id"])
            merged_record.setdefault("run_id", summary["run_id"])
            merged_record.setdefault("mode", summary["mode"])
            merged_record.setdefault("use_reflection", summary["use_reflection"])
            results.append(merged_record)

    records_with_sample_index = [
        record for record in results if record.get("sample_index") is not None
    ]
    unique_sample_indices = {
        record.get("sample_index") for record in records_with_sample_index
    }

    return {
        "n_processed": len(results),
        "result_name": result_name,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(input_dir),
        "n_source_files": len(files),
        "source_files": [str(path) for path in files],
        "participants": sorted(
            {
                summary["participant_id"]
                for summary in summaries
                if summary.get("participant_id") is not None
            }
        ),
        "runs": summaries,
        "mode": common_value(payloads, "mode"),
        "use_reflection": common_value(payloads, "use_reflection"),
        "user_source": common_value(payloads, "user_source") or "human",
        "clarify_quota": common_value(payloads, "clarify_quota"),
        "memo_window": common_value(payloads, "memo_window"),
        "dataset_path": common_value(payloads, "dataset_path"),
        "model": common_value(payloads, "model"),
        "sampling_scope": common_value(payloads, "sampling_scope"),
        "within_run_replacement": common_value(payloads, "within_run_replacement"),
        "across_run_replacement": common_value(payloads, "across_run_replacement"),
        "unique_sample_indices": len(unique_sample_indices),
        "duplicate_sample_index_records": len(records_with_sample_index)
        - len(unique_sample_indices),
        "duplicate_sample_indices": duplicate_sample_index_report(results),
        "duplicate_sample_index_note": (
            "Duplicate sample_index values are expected for human runs and are "
            "preserved. Records are not deduplicated or overwritten."
        ),
        "results": results,
    }


def merge_all_human_results(input_dir: Path, output_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    result_names = discover_result_names(input_dir)
    if not result_names:
        raise SystemExit(f"No result JSON files found under {input_dir}")

    output_paths: list[Path] = []
    for result_name in result_names:
        files = discover_result_files(input_dir, result_name)
        payload = merge_result_files(files, input_dir=input_dir, result_name=result_name)
        output_path = output_dir / result_name
        write_json(output_path, payload)
        output_paths.append(output_path)
        print(
            f"Merged {len(files)} files into {output_path} "
            f"({payload['n_processed']} records, "
            f"{payload['unique_sample_indices']} unique sample indices)."
        )

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-participant result JSON files under outputs/human_runs. "
            "Files are grouped by result filename, such as reflection.json and "
            "without_reflection.json."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Human runs root directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for merged result files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_all_human_results(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
