import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

from .reflection import run_trials_for_one_sample
from .state import action_to_id, id_to_action
from .utils import extract_trial_steps
from .llm_factory import get_primary_model_dirname, get_primary_model_name


def _run_for_one_sample(
    sample: dict,
    use_reflection: bool,
    mode: str,
    run_id: str | None,
    clarify_quota: int | None,
    max_trials: int,
    memo_window: int,
) -> List[dict]:
    """
    Run the specified mode on one sample (clarify / r-clarify / reflect).
    Returns a list of final_state entries for each trial of that sample.
    """
    return run_trials_for_one_sample(
        sample,
        use_reflection=use_reflection,
        mode=mode,
        run_id=run_id,
        clarify_quota=clarify_quota,
        max_trials=max_trials,
        memo_window=memo_window,
    )


def main(
    mode: str,
    clarify_quota: int | None = None,
    max_trials: int | None = None,
    memo_window: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> None:
    """
    Run the dataset in a given mode.
    - mode = "clarify"    -> ReAct, no reflection
    - mode = "r-clarify"  -> ReAct + reflection
    - mode = "reflect"    -> ReAct + reflection, execute-only (no clarify)
    - mode = "non_thinking_clarify" -> ReAct, no reflection, no thought output
    - mode = "r-clarify_non_cot" -> ReAct + reflection, no thought output
    - mode = "cot"        -> pure CoT, single-step (no clarify)
    - mode = "cot_reflect" -> CoT + reflection (no clarify)
    - mode = "direct"     -> fully direct: almost no thinking, pick action immediately
    - mode = "r-clarify_reflexion"  -> ReAct + reflection, multi-trial with last k memos
    - mode = "reflect_action_only" -> ReAct + reflection (wrong actions only)
    """
    assert mode in {
        "clarify",
        "r-clarify",
        "reflect",
        "non_thinking_clarify",
        "r-clarify_non_cot",
        "cot",
        "cot_reflect",
        "direct",
        "r-clarify_reflexion",
        "reflect_action_only",
    }, "mode 必须是 'clarify' / 'r-clarify' / 'reflect' / 'non_thinking_clarify' / 'r-clarify_non_cot' / 'cot' / 'cot_reflect' / 'direct' / 'r-clarify_reflexion' / 'reflect_action_only'"

    # Enable post-clarification reflection based on mode.
    if mode in {"cot", "direct"}:
        use_reflection = False
    elif mode in {"clarify", "non_thinking_clarify"}:
        use_reflection = False
    elif mode in {"r-clarify", "r-clarify_non_cot", "r-clarify_reflexion", "cot_reflect", "reflect_action_only", "reflect"}:
        use_reflection = True

    # max_trials must be provided externally (applies to all modes).
    if max_trials is None:
        raise ValueError("max_trials must be provided via --max-trials.")

    # Reflection memo window (last k memos passed to next trial).
    if memo_window is None:
        memo_window = 1 if mode in {"r-clarify", "r-clarify_non_cot", "cot_reflect", "reflect_action_only", "reflect"} else (3 if mode == "r-clarify_reflexion" else 0)

    # Load samples.
    data_path = Path("data/processed_data_expanded.json")
    with data_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    if start is not None or end is not None:
        start_idx = 0 if start is None else start
        end_idx = len(samples) if end is None else end
        samples = [s for s in samples if start_idx <= s.get("index", -1) < end_idx]

    # Results file (split by model + mode for resume/eval).
    model_dir = get_primary_model_dirname()
    results_dir = Path("outputs") / model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_suffix = ""
    if start is not None or end is not None:
        metrics_suffix = f"_{start_idx}_{end_idx}"
    metrics_path = results_dir / f"{mode}{metrics_suffix}.json"

    # Load completed samples if available.
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            records: List[dict] = data.get("results", [])
    else:
        records = []

    done_indices = {
        rec.get("sample_index")
        for rec in records
        if rec.get("sample_index") is not None
    }

    model_name = get_primary_model_name()
    quota_text = "infinite" if clarify_quota is None else f"{clarify_quota}"
    trial_text = f"{max_trials}"
    memo_text = f"{memo_window}" if memo_window else "0"
    shard_text = ""
    if start is not None or end is not None:
        shard_text = f", range: [{start_idx}, {end_idx})"
    print(
        f"[Progress] Mode: {mode}, Model: {model_name}, Sample(done): {len(done_indices)}, "
        f"Clarification quote: {quota_text}, max_trials: {trial_text}, memo_window: {memo_text}"
        f"{shard_text}"
    )

    # Filter completed samples; run remaining and fill any missing ones.
    remaining_samples = [s for s in samples if s.get("index") not in done_indices]

    if not remaining_samples:
        print(f"All samples had been processed under {mode} mode.")
        return

    print(
        f"[Progress] Number of sample to be processed: {len(remaining_samples)} "
    )

    # Do not use timestamp/run ID; histories go to histories/<mode>.
    run_id = None

    # Use a thread pool; currently max_workers=1 (no parallelism).
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_sample = {
            executor.submit(
                _run_for_one_sample,
                sample,
                use_reflection,
                mode,
                run_id,
                clarify_quota,
                max_trials,
                memo_window,
            ): sample
            for sample in remaining_samples
        }

        processed = 0

        for future in tqdm(
            as_completed(future_to_sample),
            total=len(remaining_samples),
            desc="Samples",
            unit="sample",
        ):
            raw_states = future.result() or []
            states: List[dict] = []
            for st in raw_states:
                if isinstance(st, dict):
                    converted = st
                elif hasattr(st, "model_dump"):
                    converted = st.model_dump()
                elif hasattr(st, "dict"):
                    converted = st.dict()
                else:
                    try:
                        converted = dict(st)
                    except Exception:
                        converted = {}
                states.append(converted)
            sample = future_to_sample[future]
            sample_index = sample.get("index")
            label_action_id = sample.get("label_action_id")
            if label_action_id is None:
                reflective_action = sample.get("reflective_action")
                if reflective_action:
                    label_action_id = action_to_id.get(reflective_action)
            gold_action_desc = sample.get("reflective_action")

            if not states:
                num_trials = 0
                correct = False
                steps_total = 0
                steps_per_trial: List[int] = []
                steps_detail_per_trial: List[List[dict]] = []
                clarifications_per_trial: List[List[dict]] = []
            else:
                num_trials = len(states)
                steps_per_trial = [st.get("step", 0) for st in states]
                steps_total = sum(steps_per_trial)
                steps_detail_per_trial = [extract_trial_steps(st) for st in states]
                clarifications_per_trial = []
                for detail in steps_detail_per_trial:
                    clarifications = [
                        {
                            "t": step.get("t"),
                            "clarification_question": step.get("clarification_question"),
                            "clarification_question_words": step.get("clarification_question_words"),
                            "user_reply": step.get("user_reply"),
                            "user_reply_words": step.get("user_reply_words"),
                            "user_source": step.get("user_source"),
                            "response_latency_seconds": step.get("response_latency_seconds"),
                        }
                        for step in detail
                        if step.get("action") == "clarify"
                    ]
                    clarifications_per_trial.append(clarifications)

                last_state = states[-1]
                pred = last_state.get("action_id")
                label_action_id = last_state.get("label_action_id") or label_action_id
                correct = (pred == label_action_id)

            if not gold_action_desc and label_action_id is not None:
                gold_action_desc = id_to_action.get(label_action_id)
            if not states:
                steps_detail_per_trial = []
                clarifications_per_trial = []

            records.append(
                {
                    "sample_index": sample_index,
                    "max_trials": max_trials,
                    "num_trials": num_trials,
                    "correct": correct,
                    "steps_total": steps_total,
                    "steps_per_trial": steps_per_trial,
                    "steps_detail_per_trial": steps_detail_per_trial,
                    "clarify_turns_per_trial": clarifications_per_trial,
                    "gold_action": gold_action_desc,
                    "user_source": "simulated",
                }
            )

            processed += 1

            # Periodically sort and write back to avoid frequent disk writes.
            if processed % 2 == 0:
                records.sort(
                    key=lambda r: (
                        r.get("sample_index") is None,
                        r.get("sample_index", -1),
                    )
                )
                payload = {
                    "n_processed": len(records),
                    "clarify_quota": clarify_quota,
                    "memo_window": memo_window,
                    "results": records,
                }
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

    # Final sort and write after the loop to save remaining samples.
    records.sort(
        key=lambda r: (r.get("sample_index") is None, r.get("sample_index", -1))
    )
    payload = {
        "n_processed": len(records),
        "clarify_quota": clarify_quota,
        "memo_window": memo_window,
        "results": records,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset in specified mode.")
    parser.add_argument(
        "--mode",
        choices=[
            "clarify",
            "r-clarify",
            "reflect",
            "non_thinking_clarify",
            "r-clarify_non_cot",
            "cot",
            "cot_reflect",
            "direct",
            "r-clarify_reflexion",
            "reflect_action_only",
        ],
        help="Select running mode.",
    )
    parser.add_argument(
        "--clarify-quota",
        type=int,
        default=None,
        help="Optional cap on how many clarify questions the agent may ask per sample (None = unlimited).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        required=True,
        help="Max trials to run per sample (must be provided explicitly).",
    )
    parser.add_argument(
        "--memo-window",
        type=int,
        default=None,
        help="How many latest reflection memos to pass to the next trial (r-clarify/reflect/cot_reflect/reflect_action_only default 1, r-clarify_reflexion default 3).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start sample index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End sample index (exclusive).",
    )

    args = parser.parse_args()
    main(
        mode=args.mode,
        clarify_quota=args.clarify_quota,
        max_trials=args.max_trials,
        memo_window=args.memo_window,
        start=args.start,
        end=args.end,
    )
