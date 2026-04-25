#!/usr/bin/env python3
"""Compute clarify usage rate among successful trials in a results JSON."""

import json
from pathlib import Path


def has_clarify_in_success_trial(res: dict, max_trial: int | None) -> bool:
    """Return True if the successful trial contains a clarify step."""
    num_trials = res.get("num_trials", 0)
    if not res.get("correct") or num_trials <= 0:
        return False
    if max_trial is not None and num_trials > max_trial:
        return False

    trial_idx = num_trials - 1

    steps_detail = res.get("steps_detail_per_trial") or []
    if trial_idx < len(steps_detail):
        trial_steps = steps_detail[trial_idx] or []
        if any((step or {}).get("action") == "clarify" for step in trial_steps):
            return True

    # Fallback: some outputs may include per-trial clarify counts.
    clarify_turns = res.get("clarify_turns_per_trial") or []
    if trial_idx < len(clarify_turns):
        try:
            return int(clarify_turns[trial_idx]) > 0
        except (TypeError, ValueError):
            return False

    return False


def has_any_clarify(res: dict) -> bool:
    """Return True if any trial contains a clarify step."""
    steps_detail = res.get("steps_detail_per_trial") or []
    for trial_steps in steps_detail:
        if any((step or {}).get("action") == "clarify" for step in (trial_steps or [])):
            return True

    clarify_turns = res.get("clarify_turns_per_trial") or []
    for turns in clarify_turns:
        try:
            if int(turns) > 0:
                return True
        except (TypeError, ValueError):
            continue

    return False


def main() -> int:
    # Set these values directly.
    results_path = "outputs/Qwen3-Next-80B-A3B-Instruct/results/clarify.json"
    max_trial = 1  # e.g., 3; use None to disable

    path = Path(results_path)
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    success_total = 0
    success_with_clarify = 0
    failed_without_clarify = 0

    for res in results:
        if not res.get("correct"):
            if not has_any_clarify(res):
                failed_without_clarify += 1
            continue

        if res.get("correct") and res.get("num_trials", 0) > 0:
            if max_trial is not None and res.get("num_trials", 0) > max_trial:
                continue
            success_total += 1
            if has_clarify_in_success_trial(res, max_trial):
                success_with_clarify += 1

    ratio = (success_with_clarify / success_total) if success_total else 0.0

    print(f"success_total: {success_total}")
    print(f"success_with_clarify: {success_with_clarify}")
    print(f"clarify_success_ratio: {ratio:.6f}")
    print(f"failed_without_clarify: {failed_without_clarify}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
