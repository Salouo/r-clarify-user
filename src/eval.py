import json
from pathlib import Path


def calculate_accuracy_by_trial(data: dict, trial: int, k: int | None = None) -> float:
    results = data["results"]
    if k is not None:
        results = results[:k]
    total = len(results)
    if total == 0:
        return 0.0

    num_correct = 0
    for res in results:
        if res.get("correct") and res.get("num_trials", 0) <= trial:
            num_correct += 1

    return num_correct / total



def calculate_average_steps_of_successful_trial(
    data: dict, trial: int, k: int | None = None
) -> float:
    """
    If only up to `trial` trials are allowed, compute the average steps
    of the successful trial for successful samples only.
    """
    results = data["results"]
    if k is not None:
        results = results[:k]

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


def calculate_average_turns_by_pass_k(
    data: dict, max_trial: int, k: int | None = None
) -> dict[int, float]:
    """
    Average number of turns (agent decisions) in the successful trial,
    under different Pass@k settings.
    """
    averages: dict[int, float] = {}
    for t in range(1, max_trial + 1):
        averages[t] = calculate_average_steps_of_successful_trial(data=data, trial=t, k=k)
    return averages


def calculate_average_turns_per_trial(
    data: dict, max_trial: int, k: int | None = None
) -> dict[int, float]:
    """
    Average number of turns for each trial index (includes both success and failure trials).
    Only samples that reached the given trial are counted.
    """
    results = data["results"]
    if k is not None:
        results = results[:k]

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


def calculate_dpass_at_k(
    data: dict, trial: int, alpha: float, k: int | None = None
) -> float:
    """
    DPass@k(alpha):
    - Successful samples are discounted by steps in the successful trial.
    - Failed samples score 0.
    - Normalize by total samples.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    results = data["results"]
    if k is not None:
        results = results[:k]
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
    data: dict, trial: int, k: int | None = None
) -> dict:
    """
    Average reflection token usage before success (prompt/completion/total).
    Only samples that succeed within `trial` are counted.
    """
    results = data["results"]
    if k is not None:
        results = results[:k]

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
    data: dict, trial: int, k: int | None = None
) -> dict:
    """
    Average total token usage up to and including the success trial.
    Includes reflect + execute + clarify with a breakdown.
    Only samples that succeed within `trial` are counted.
    """
    results = data["results"]
    if k is not None:
        results = results[:k]

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


def main():
    metrics_path = Path("outputs/Qwen3-Next-80B-A3B-Instruct/results/r-clarify.json")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trial = 5   # up to n_trials
    k = None     # eval up to n_samples
    alpha = 0.8
    acc = calculate_accuracy_by_trial(data=data, trial=trial, k=k)
    avg_steps = calculate_average_steps_of_successful_trial(data=data, trial=trial, k=k)
    dpass = calculate_dpass_at_k(data=data, trial=trial, alpha=alpha, k=k)
    total_usage = calculate_total_token_usage_before_success(
        data=data, trial=trial, k=k
    )
    avg_turns_by_pass_k = calculate_average_turns_by_pass_k(
        data=data, max_trial=trial, k=k
    )
    avg_turns_per_trial = calculate_average_turns_per_trial(
        data=data, max_trial=trial, k=k
    )

    if k:  
        print(f"Pass@{trial} (k={k}): {acc:.4f}")
        print(f"DPass@{trial} (k={k}, alpha={alpha}): {dpass:.4f}")
    else:
        print(f"Pass@{trial} (k=full): {acc:.4f}")
        print(f"DPass@{trial} (k=full, alpha={alpha}): {dpass:.4f}")

    print(
        "Total token usage up to success (reflect + execute + clarify)\n"
        f"(k={'full' if k is None else k}, trial={trial}):\n"
        f"overall_prompt_tokens: {total_usage['overall']['prompt_tokens']:.2f},\n"
        f"overall_completion_tokens: {total_usage['overall']['completion_tokens']:.2f},\n"
        f"overall_total_tokens: {total_usage['overall']['total_tokens']:.2f},\n"
        f"reflect_total_tokens: {total_usage['reflect']['total_tokens']:.2f},\n"
        f"execute_total_tokens: {total_usage['execute']['total_tokens']:.2f},\n"
        f"clarify_total_tokens: {total_usage['clarify']['total_tokens']:.2f},\n"
        f"success={total_usage['num_success']}\n"
    )

    print(f"Steps per successful trial: {avg_steps:.4f}")
    print("Average turns per successful trial under Pass@k:")
    for pass_k in range(1, trial + 1):
        avg_turns = avg_turns_by_pass_k.get(pass_k, 0.0)
        print(f"  Pass@{pass_k}: {avg_turns:.3f}")
    print("Average turns per trial (success + failure):")
    for trial_idx in range(1, trial + 1):
        avg_turns = avg_turns_per_trial.get(trial_idx, 0.0)
        print(f"  Trial {trial_idx}: {avg_turns:.3f}")


if __name__ == "__main__":
    main()
