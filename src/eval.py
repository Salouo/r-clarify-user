import argparse
import json
from pathlib import Path
from typing import Any


def calculate_accuracy_by_trial(data: dict, trial: int) -> float:
    results = data["results"]
    total = len(results)
    if total == 0:
        return 0.0

    num_correct = 0
    for res in results:
        if res.get("correct") and res.get("num_trials", 0) <= trial:
            num_correct += 1

    return num_correct / total


def get_first_successful_trial(instance_result: dict, k: int) -> dict[str, Any] | None:
    """
    Return the first successful trial within the first k trials.

    The runner stops as soon as a sample succeeds, so the existing eval logic
    represents the first successful trial as `num_trials` when `correct` is true.
    """
    if not instance_result.get("correct"):
        return None

    num_trials = instance_result.get("num_trials", 0)
    if num_trials <= 0 or num_trials > k:
        return None

    trial_idx = num_trials - 1
    trial: dict[str, Any] = {
        "trial_index": num_trials,
        "steps": [],
    }

    steps_detail = instance_result.get("steps_detail_per_trial", [])
    if trial_idx < len(steps_detail):
        trial["steps"] = steps_detail[trial_idx] or []

    clarify_turns = instance_result.get("clarify_turns_per_trial", [])
    if trial_idx < len(clarify_turns):
        trial["clarifications"] = clarify_turns[trial_idx] or []

    return trial


def count_clarifications_in_trial(trial: dict[str, Any] | list[dict]) -> int:
    """
    Count clarification questions in one trial only.

    Prefer the precomputed `clarify_turns_per_trial` entries when available;
    otherwise infer from the trial step log by counting clarify actions/events.
    """
    if isinstance(trial, dict):
        if "clarifications" in trial and trial["clarifications"] is not None:
            return len(trial["clarifications"])
        steps = trial.get("steps", [])
    else:
        steps = trial

    if not isinstance(steps, list):
        return 0

    clarify_count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        if (
            step.get("action") == "clarify"
            or step.get("event") == "clarify"
            or step.get("type") == "clarify"
            or step.get("next_decision") == "clarify"
        ):
            clarify_count += 1
    return clarify_count


def compute_cspass(results: list[dict], k: int) -> float:
    """Clarification-supported Pass@k."""
    total = len(results)
    if total == 0:
        return 0.0

    supported_success = 0
    for res in results:
        successful_trial = get_first_successful_trial(res, k)
        if successful_trial is None:
            continue
        if count_clarifications_in_trial(successful_trial) > 0:
            supported_success += 1

    return supported_success / total


def compute_avg_clarify_in_success(results: list[dict], k: int) -> float:
    """Average number of clarifications in the first successful trial."""
    total_clarifications = 0
    total_success = 0

    for res in results:
        successful_trial = get_first_successful_trial(res, k)
        if successful_trial is None:
            continue
        total_clarifications += count_clarifications_in_trial(successful_trial)
        total_success += 1

    if total_success == 0:
        return 0.0
    return total_clarifications / total_success


def count_successes_by_trial(results: list[dict], k: int) -> int:
    return sum(1 for res in results if get_first_successful_trial(res, k) is not None)


def count_clarification_supported_successes(results: list[dict], k: int) -> int:
    total = 0
    for res in results:
        successful_trial = get_first_successful_trial(res, k)
        if successful_trial is None:
            continue
        if count_clarifications_in_trial(successful_trial) > 0:
            total += 1
    return total


def calculate_average_steps_of_successful_trial(
    data: dict, trial: int
) -> float:
    """
    If only up to `trial` trials are allowed, compute the average steps
    of the successful trial for successful samples only.
    """
    results = data["results"]

    total_steps = 0
    total_success = 0

    for res in results:
        if not res.get("correct"):
            continue
        num_trials = res.get("num_trials", 0)
        if num_trials <= 0 or num_trials > trial:
            continue
        steps_per_trial = res.get("steps_per_trial", [])
        if len(steps_per_trial) < num_trials:
            continue
        total_steps += steps_per_trial[num_trials - 1]
        total_success += 1

    if total_success == 0:
        return 0.0
    return total_steps / total_success


def calculate_average_turns_by_trial_budget(
    data: dict, max_trial: int
) -> dict[int, float]:
    """
    Average number of turns (agent decisions) in the successful trial,
    under different allowed trial budgets.
    """
    averages: dict[int, float] = {}
    for t in range(1, max_trial + 1):
        averages[t] = calculate_average_steps_of_successful_trial(data=data, trial=t)
    return averages


def calculate_average_turns_per_trial(
    data: dict, max_trial: int
) -> dict[int, float]:
    """
    Average number of turns for each trial index (includes both success and failure trials).
    Only samples that reached the given trial are counted.
    """
    results = data["results"]

    averages: dict[int, float] = {}
    for t in range(1, max_trial + 1):
        total_steps = 0
        total_trials = 0
        for res in results:
            steps_per_trial = res.get("steps_per_trial", [])
            if len(steps_per_trial) < t:
                continue
            total_steps += steps_per_trial[t - 1]
            total_trials += 1
        averages[t] = (total_steps / total_trials) if total_trials else 0.0
    return averages


def calculate_dpass_by_trial(
    data: dict, trial: int, alpha: float
) -> float:
    """
    DPass for the given trial budget:
    - Successful samples are discounted by steps in the successful trial.
    - Failed samples score 0.
    - Normalize by total samples.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    results = data["results"]
    total = len(results)
    if total == 0:
        return 0.0

    total_score = 0.0

    for res in results:
        if not res.get("correct"):
            continue
        num_trials = res.get("num_trials", 0)
        if num_trials <= 0 or num_trials > trial:
            continue
        steps_per_trial = res.get("steps_per_trial", [])
        if len(steps_per_trial) < num_trials:
            continue
        steps = steps_per_trial[num_trials - 1]
        total_score += alpha ** (steps - 1)

    return total_score / total


def calculate_reflection_token_usage_before_success(
    data: dict, trial: int
) -> dict:
    """
    Average reflection token usage before success (prompt/completion/total).
    Only samples that succeed within `trial` are counted.
    """
    results = data["results"]

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    num_success = 0

    for res in results:
        if not res.get("correct"):
            continue
        num_trials = res.get("num_trials", 0)
        if num_trials <= 0 or num_trials > trial:
            continue

        steps_detail = res.get("steps_detail_per_trial", [])
        for t in range(num_trials - 1):
            if t >= len(steps_detail):
                break
            trial_steps = steps_detail[t] or []
            usage = None
            for step in trial_steps:
                usage = step.get("token_usage_reflect")
                if usage:
                    break
            if not usage:
                continue
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", prompt + completion)
            totals["prompt_tokens"] += prompt
            totals["completion_tokens"] += completion
            totals["total_tokens"] += total

        num_success += 1

    if num_success == 0:
        return {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "num_success": 0,
            "num_samples": len(results),
        }

    return {
        "prompt_tokens": totals["prompt_tokens"] / num_success,
        "completion_tokens": totals["completion_tokens"] / num_success,
        "total_tokens": totals["total_tokens"] / num_success,
        "num_success": num_success,
        "num_samples": len(results),
    }


def calculate_total_token_usage_before_success(
    data: dict, trial: int
) -> dict:
    """
    Average total token usage up to and including the success trial.
    Includes reflect + execute + clarify with a breakdown.
    Only samples that succeed within `trial` are counted.
    """
    results = data["results"]

    def _new_bucket() -> dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _add_usage(bucket: dict, usage: dict) -> None:
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)
        bucket["prompt_tokens"] += prompt
        bucket["completion_tokens"] += completion
        bucket["total_tokens"] += total

    totals = {
        "overall": _new_bucket(),
        "reflect": _new_bucket(),
        "execute": _new_bucket(),
        "clarify": _new_bucket(),
    }
    num_success = 0

    for res in results:
        if not res.get("correct"):
            continue
        num_trials = res.get("num_trials", 0)
        if num_trials <= 0 or num_trials > trial:
            continue

        steps_detail = res.get("steps_detail_per_trial", [])
        # Include the success trial: count 0..num_trials-1.
        for t in range(num_trials):
            if t >= len(steps_detail):
                break
            trial_steps = steps_detail[t] or []
            for step in trial_steps:
                action = step.get("action")
                usage_agent = step.get("token_usage_agent")
                if usage_agent:
                    _add_usage(totals["overall"], usage_agent)
                    if action == "clarify":
                        _add_usage(totals["clarify"], usage_agent)
                    else:
                        _add_usage(totals["execute"], usage_agent)

                usage_reflect = step.get("token_usage_reflect")
                if usage_reflect:
                    _add_usage(totals["overall"], usage_reflect)
                    _add_usage(totals["reflect"], usage_reflect)

        num_success += 1

    if num_success == 0:
        return {
            "overall": {**_new_bucket()},
            "reflect": {**_new_bucket()},
            "execute": {**_new_bucket()},
            "clarify": {**_new_bucket()},
            "num_success": 0,
            "num_samples": len(results),
        }

    def _avg(bucket: dict) -> dict:
        return {
            "prompt_tokens": bucket["prompt_tokens"] / num_success,
            "completion_tokens": bucket["completion_tokens"] / num_success,
            "total_tokens": bucket["total_tokens"] / num_success,
        }

    return {
        "overall": _avg(totals["overall"]),
        "reflect": _avg(totals["reflect"]),
        "execute": _avg(totals["execute"]),
        "clarify": _avg(totals["clarify"]),
        "num_success": num_success,
        "num_samples": len(results),
    }


def load_metrics_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def new_token_bucket() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def add_token_usage(bucket: dict[str, int], usage: dict | None) -> None:
    if not isinstance(usage, dict):
        return
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    bucket["prompt_tokens"] += prompt
    bucket["completion_tokens"] += completion
    bucket["total_tokens"] += total


def average_token_bucket(bucket: dict[str, int], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    return {
        "prompt_tokens": bucket["prompt_tokens"] / denominator,
        "completion_tokens": bucket["completion_tokens"] / denominator,
        "total_tokens": bucket["total_tokens"] / denominator,
    }


def calculate_token_usage(data: dict) -> dict[str, Any]:
    """
    Sum token usage stored in result steps.

    This intentionally counts only the main agent (`token_usage_agent`) and
    reflection (`token_usage_reflect`) fields. Simulated-user LLM calls are not
    serialized into those fields, so they are excluded from the total here.
    """
    results = data.get("results", [])
    totals: dict[str, dict[str, int]] = {
        "overall": new_token_bucket(),
        "agent": new_token_bucket(),
        "clarify": new_token_bucket(),
        "execute": new_token_bucket(),
        "reflect": new_token_bucket(),
    }
    counts = {
        "agent_calls": 0,
        "clarify_agent_calls": 0,
        "execute_agent_calls": 0,
        "reflection_calls": 0,
        "steps": 0,
        "trials": 0,
    }

    for res in results:
        for trial_steps in res.get("steps_detail_per_trial", []):
            if not isinstance(trial_steps, list):
                continue
            counts["trials"] += 1
            for step in trial_steps:
                if not isinstance(step, dict):
                    continue
                counts["steps"] += 1
                action = step.get("action")

                agent_usage = step.get("token_usage_agent")
                if isinstance(agent_usage, dict):
                    add_token_usage(totals["overall"], agent_usage)
                    add_token_usage(totals["agent"], agent_usage)
                    counts["agent_calls"] += 1
                    if action == "clarify":
                        add_token_usage(totals["clarify"], agent_usage)
                        counts["clarify_agent_calls"] += 1
                    else:
                        add_token_usage(totals["execute"], agent_usage)
                        counts["execute_agent_calls"] += 1

                reflect_usage = step.get("token_usage_reflect")
                if isinstance(reflect_usage, dict):
                    add_token_usage(totals["overall"], reflect_usage)
                    add_token_usage(totals["reflect"], reflect_usage)
                    counts["reflection_calls"] += 1

    num_samples = len(results)
    return {
        "totals": totals,
        "avg_per_sample": {
            name: average_token_bucket(bucket, num_samples)
            for name, bucket in totals.items()
        },
        "counts": counts,
        "num_samples": num_samples,
    }


def compute_summary_metrics(data: dict, alpha: float) -> dict[str, Any]:
    results = data.get("results", [])
    metrics: dict[str, Any] = {
        "total_samples": len(results),
        "token_usage": calculate_token_usage(data),
    }
    for k in (3, 5):
        metrics[f"pass_{k}"] = calculate_accuracy_by_trial(data=data, trial=k)
        metrics[f"dpass_{k}"] = calculate_dpass_by_trial(
            data=data,
            trial=k,
            alpha=alpha,
        )
        metrics[f"cspass_{k}"] = compute_cspass(results=results, k=k)
        metrics[f"avg_clarify_success_{k}"] = compute_avg_clarify_in_success(
            results=results,
            k=k,
        )
        metrics[f"success_count_{k}"] = count_successes_by_trial(results, k)
        metrics[f"cs_success_count_{k}"] = count_clarification_supported_successes(
            results,
            k,
        )
    return metrics


def _percent(value: float) -> float:
    return value * 100


def _format_tokens(usage: dict[str, int] | dict[str, float]) -> str:
    return (
        f"prompt={usage.get('prompt_tokens', 0):,.0f}, "
        f"completion={usage.get('completion_tokens', 0):,.0f}, "
        f"total={usage.get('total_tokens', 0):,.0f}"
    )


def print_metric_summary(label: str, metrics: dict[str, Any], alpha: float) -> None:
    print(f"{label}:")
    print(
        f"  Pass@3: {_percent(metrics['pass_3']):.2f} | "
        f"Pass@5: {_percent(metrics['pass_5']):.2f}"
    )
    print(
        f"  DPass@3 (alpha={alpha}): {_percent(metrics['dpass_3']):.2f} | "
        f"DPass@5 (alpha={alpha}): {_percent(metrics['dpass_5']):.2f}"
    )
    print(
        f"  CSPass@3: {_percent(metrics['cspass_3']):.2f} | "
        f"CSPass@5: {_percent(metrics['cspass_5']):.2f}"
    )
    print(
        "  Avg. # Clarify in Success @3: "
        f"{metrics['avg_clarify_success_3']:.2f} | "
        "Avg. # Clarify in Success @5: "
        f"{metrics['avg_clarify_success_5']:.2f}"
    )
    token_usage = metrics["token_usage"]
    token_totals = token_usage["totals"]
    avg_per_sample = token_usage["avg_per_sample"]
    counts = token_usage["counts"]
    print("  Token usage (excludes simulated user):")
    print(f"    Total: {_format_tokens(token_totals['overall'])}")
    print(f"    Reflection: {_format_tokens(token_totals['reflect'])}")
    print(f"    Agent total: {_format_tokens(token_totals['agent'])}")
    print(f"    Agent clarify: {_format_tokens(token_totals['clarify'])}")
    print(f"    Agent execute: {_format_tokens(token_totals['execute'])}")
    print(f"    Avg/sample total: {_format_tokens(avg_per_sample['overall'])}")
    print(
        "    Calls: "
        f"agent={counts['agent_calls']}, "
        f"reflection={counts['reflection_calls']}, "
        f"trials={counts['trials']}, "
        f"steps={counts['steps']}"
    )


def print_sanity_check_summary(label: str, metrics: dict[str, Any]) -> None:
    print(f"{label}:")
    print(f"  Total samples: {metrics['total_samples']}")
    print(
        f"  Successful samples: k=3 {metrics['success_count_3']}, "
        f"k=5 {metrics['success_count_5']}"
    )
    print(
        "  Clarification-supported success samples: "
        f"k=3 {metrics['cs_success_count_3']}, "
        f"k=5 {metrics['cs_success_count_5']}"
    )
    print(
        f"  CSPass@3 / CSPass@5: "
        f"{_percent(metrics['cspass_3']):.2f} / "
        f"{_percent(metrics['cspass_5']):.2f}"
    )
    print(
        "  Avg. # Clarify in Success @3 / @5: "
        f"{metrics['avg_clarify_success_3']:.2f} / "
        f"{metrics['avg_clarify_success_5']:.2f}"
    )


def infer_result_label(path: Path, data: dict) -> str:
    result_name = data.get("result_name")
    mode = data.get("mode")
    if result_name and mode:
        return f"{result_name} ({mode})"
    if result_name:
        return str(result_name)
    if mode:
        return str(mode)
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Pass, DPass, CSPass, and clarification counts."
    )
    parser.add_argument(
        "result_path",
        nargs="?",
        type=Path,
        help="Path to one result JSON file.",
    )
    parser.add_argument(
        "--result-path",
        "-r",
        type=Path,
        dest="result_path_flag",
        help="Path to one result JSON file. Equivalent to the positional path.",
    )
    parser.add_argument(
        "--label",
        "-l",
        dest="label",
        help="Optional display label for the result path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Discount factor for DPass. Default: 0.8.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    alpha = args.alpha
    metrics_path = args.result_path_flag or args.result_path

    if metrics_path is None:
        raise SystemExit("Please provide one result JSON path.")
    if args.result_path_flag and args.result_path:
        raise SystemExit("Please provide the result path only once.")

    data = load_metrics_data(metrics_path)
    label = args.label or infer_result_label(metrics_path, data)
    metrics = compute_summary_metrics(data=data, alpha=alpha)

    print("Metric summary")
    print(f"\nResult path: {metrics_path}")
    print_metric_summary(label, metrics, alpha)

    print("\nSanity check summary")
    print_sanity_check_summary(label, metrics)


if __name__ == "__main__":
    main()
