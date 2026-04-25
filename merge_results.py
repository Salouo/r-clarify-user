import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_results(files: list[Path]) -> dict:
    results_by_index: dict[int, dict] = {}
    results_no_index: list[dict] = []
    clarify_quota = None
    memo_window = None
    dup_indices: set[int] = set()

    for path in files:
        data = load_json(path)
        if clarify_quota is None:
            clarify_quota = data.get("clarify_quota")
        if memo_window is None:
            memo_window = data.get("memo_window")

        for rec in data.get("results", []):
            idx = rec.get("sample_index")
            if idx is None:
                results_no_index.append(rec)
                continue
            if idx in results_by_index:
                dup_indices.add(idx)
            results_by_index[idx] = rec

    results = list(results_by_index.values()) + results_no_index
    results.sort(key=lambda r: (r.get("sample_index") is None, r.get("sample_index", -1)))

    payload = {
        "n_processed": len(results),
        "clarify_quota": clarify_quota,
        "memo_window": memo_window,
        "results": results,
    }
    if dup_indices:
        payload["_merge_warnings"] = {
            "duplicate_sample_indices": sorted(dup_indices),
            "note": "Duplicates were overwritten by later files.",
        }
    return payload


def main() -> None:
    input_dir = Path("outputs/Qwen3-30B-A3B-Instruct-2507/results")
    glob_pattern = "r-clarify_non_cot_*.json"
    output_path = Path("outputs//results/r-clarify_non_cot.json")

    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        raise SystemExit(f"No files matched {glob_pattern} in {input_dir}")

    payload = merge_results(files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(files)} files into {output_path} ({payload['n_processed']} records).")


if __name__ == "__main__":
    main()
