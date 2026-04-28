from __future__ import annotations

import json
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.llm_factory import get_primary_model_name
from src.reflection import generate_trial_reflection_memo
from src.state import AgentState, AgentStep, action_to_id, id_to_action
from src.token_usage import TokenUsageCollector
from src.utils import (
    extract_trial_steps,
    format_single_trial_episode,
    make_system_prompt_from_one_sample,
)


SAMPLING_SCOPE = "full_dataset_each_run"


@dataclass
class HumanExperimentConfig:
    dataset_path: Path
    output_dir: Path = Path("outputs/human_runs")
    run_id: str = ""
    participant_id: str | None = None
    n_samples: int = 30
    seed: int | None = None
    sample_ids: list[int] | None = None
    subset_output_path: Path | None = None
    mode: str = "r-clarify"
    use_reflection: bool = True
    max_trials: int = 5
    clarify_quota: int | None = 2
    memo_window: int | None = None
    show_gold_to_user: bool = True
    timestamp: str = ""


@dataclass
class HumanEpisodeState:
    run_id: str
    participant_id: str | None
    sample: dict[str, Any]
    sample_id: int
    sample_position: int
    total_samples: int
    subset_sample_ids: list[int]
    mode: str
    use_reflection: bool
    max_trials: int
    clarify_quota: int | None
    memo_window: int
    show_gold_to_user: bool
    current_trial_index: int = 0
    current_agent_state: AgentState | None = None
    current_question: str | None = None
    current_question_started_at: float | None = None
    trial_states: list[dict[str, Any]] = field(default_factory=list)
    reflection_memos: list[str] = field(default_factory=list)
    trial_memos: list[dict[str, Any]] = field(default_factory=list)
    trial_contexts: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[int | None] = field(default_factory=list)
    response_latencies: list[dict[str, Any]] = field(default_factory=list)
    human_visible_contexts: list[dict[str, Any]] = field(default_factory=list)
    reflection_usage: TokenUsageCollector = field(default_factory=TokenUsageCollector)
    finished: bool = False
    success: bool = False
    last_action_id: int | None = None
    last_action_desc: str = ""
    error_message: str | None = None
    start_time: float = field(default_factory=time.time)
    finished_at: float | None = None


@dataclass
class HumanExperimentState:
    config: HumanExperimentConfig
    samples: list[dict[str, Any]]
    sample_ids: list[int]
    subset_metadata: dict[str, Any]
    run_dir: Path
    subset_path: Path | None
    current_pos: int = 0
    current_episode: HumanEpisodeState | None = None
    finished_episodes: dict[int, HumanEpisodeState] = field(default_factory=dict)
    records_by_sample_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    run_start_time: float = field(default_factory=time.time)
    last_results_path: Path | None = None
    last_export_message: str = ""


def parse_sample_ids(sample_ids: str | list[int] | None) -> list[int] | None:
    if sample_ids is None:
        return None
    if isinstance(sample_ids, list):
        return [int(x) for x in sample_ids]
    text = sample_ids.strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def generate_run_id(participant_id: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)
    participant = _safe_slug(participant_id or "anon")
    return f"human_{participant}_{stamp}_{rand}"


def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Dataset JSON must contain a list field named 'samples'.")
    return samples


def load_human_subset(
    *,
    dataset_path: str | Path = "data/processed_data_expanded.json",
    n_samples: int = 30,
    seed: int | None = None,
    sample_ids: str | list[int] | None = None,
    participant_id: str | None = None,
    subset_output_path: str | Path | None = None,
    output_dir: str | Path = "outputs/human_runs",
    run_id: str | None = None,
    mode: str = "r-clarify",
    use_reflection: bool = True,
    max_trials: int = 5,
    clarify_quota: int | None = 2,
    memo_window: int | None = None,
    show_gold_to_user: bool = True,
) -> HumanExperimentState:
    all_samples = load_dataset(dataset_path)
    sample_by_id = {int(s["index"]): s for s in all_samples if "index" in s}
    if len(sample_by_id) != len(all_samples):
        raise ValueError("Every sample must have a unique integer 'index'.")

    explicit_sample_ids = parse_sample_ids(sample_ids)
    actual_seed = seed if seed is not None else secrets.randbelow(2**32)

    if explicit_sample_ids is not None:
        if len(explicit_sample_ids) != len(set(explicit_sample_ids)):
            raise ValueError("--sample_ids must not contain duplicates within one run.")
        missing = [sid for sid in explicit_sample_ids if sid not in sample_by_id]
        if missing:
            raise ValueError(f"Unknown sample_ids: {missing}")
        selected_ids = explicit_sample_ids
    else:
        if n_samples <= 0:
            raise ValueError("--n_samples must be greater than 0.")
        if n_samples > len(sample_by_id):
            raise ValueError(
                f"--n_samples={n_samples} exceeds dataset size {len(sample_by_id)}."
            )
        rng = random.Random(actual_seed)
        selected_ids = rng.sample(list(sample_by_id.keys()), n_samples)

    selected_samples = [sample_by_id[sid] for sid in selected_ids]
    actual_run_id = run_id or generate_run_id(participant_id)
    run_dir = Path(output_dir) / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    subset_path = Path(subset_output_path) if subset_output_path else None

    timestamp = datetime.now().isoformat(timespec="seconds")
    actual_memo_window = _resolve_memo_window(mode=mode, memo_window=memo_window)
    config = HumanExperimentConfig(
        dataset_path=Path(dataset_path),
        output_dir=Path(output_dir),
        run_id=actual_run_id,
        participant_id=participant_id,
        n_samples=len(selected_ids),
        seed=actual_seed,
        sample_ids=selected_ids,
        subset_output_path=subset_path,
        mode=mode,
        use_reflection=use_reflection,
        max_trials=max_trials,
        clarify_quota=clarify_quota,
        memo_window=actual_memo_window,
        show_gold_to_user=show_gold_to_user,
        timestamp=timestamp,
    )
    subset_metadata = {
        "run_id": actual_run_id,
        "participant_id": participant_id,
        "timestamp": timestamp,
        "dataset_path": str(Path(dataset_path)),
        "n_samples": len(selected_ids),
        "seed": actual_seed,
        "sample_ids": selected_ids,
        "sampling_scope": SAMPLING_SCOPE,
        "within_run_replacement": False,
        "across_run_replacement": True,
    }
    if subset_path is not None:
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        with subset_path.open("w", encoding="utf-8") as f:
            json.dump(subset_metadata, f, ensure_ascii=False, indent=2)

    return HumanExperimentState(
        config=config,
        samples=selected_samples,
        sample_ids=selected_ids,
        subset_metadata=subset_metadata,
        run_dir=run_dir,
        subset_path=subset_path,
    )


def start_episode(
    sample: dict[str, Any],
    config: HumanExperimentConfig,
    *,
    sample_position: int = 0,
    total_samples: int | None = None,
    subset_sample_ids: list[int] | None = None,
) -> HumanEpisodeState:
    sample_id = int(sample["index"])
    episode = HumanEpisodeState(
        run_id=config.run_id,
        participant_id=config.participant_id,
        sample=sample,
        sample_id=sample_id,
        sample_position=sample_position,
        total_samples=total_samples or config.n_samples,
        subset_sample_ids=list(subset_sample_ids or config.sample_ids or [sample_id]),
        mode=config.mode,
        use_reflection=config.use_reflection,
        max_trials=config.max_trials,
        clarify_quota=config.clarify_quota,
        memo_window=_resolve_memo_window(mode=config.mode, memo_window=config.memo_window),
        show_gold_to_user=config.show_gold_to_user,
    )
    return _continue_episode(episode)


def start_current_episode(
    experiment_state: HumanExperimentState,
    *,
    restart: bool = True,
) -> HumanExperimentState:
    if not experiment_state.samples:
        raise ValueError("No samples are available for this run.")
    if experiment_state.current_pos >= len(experiment_state.samples):
        experiment_state.current_pos = len(experiment_state.samples) - 1
    if experiment_state.current_episode and not restart:
        return experiment_state

    sample = experiment_state.samples[experiment_state.current_pos]
    experiment_state.current_episode = start_episode(
        sample,
        experiment_state.config,
        sample_position=experiment_state.current_pos + 1,
        total_samples=len(experiment_state.samples),
        subset_sample_ids=experiment_state.sample_ids,
    )
    if experiment_state.current_episode.finished:
        _record_finished_episode(experiment_state, experiment_state.current_episode)
    return experiment_state


def submit_human_answer(
    state: HumanEpisodeState,
    human_answer: str,
) -> HumanEpisodeState:
    answer = (human_answer or "").strip()
    if state.finished:
        state.error_message = "このサンプルはすでに完了しています。次のサンプルへ進んでください。"
        return state
    if not state.current_agent_state or not state.current_question:
        state.error_message = "現在、回答が必要な明確化質問はありません。"
        return state
    if not answer:
        state.error_message = "回答を入力してください。"
        return state

    latency = None
    if state.current_question_started_at is not None:
        latency = max(0.0, time.time() - state.current_question_started_at)
        latency = round(latency, 3)

    visible_context = build_human_visible_context(state)
    visible_context["human_answer"] = answer
    visible_context["response_latency_seconds"] = latency
    state.human_visible_contexts.append(visible_context)
    state.response_latencies.append(
        {
            "trial": state.current_trial_index,
            "question": state.current_question,
            "response_latency_seconds": latency,
        }
    )

    agent_state = state.current_agent_state
    new_trace = list(agent_state.trace)
    if new_trace:
        last = new_trace[-1]
        update = {
            "user_reply": answer,
            "user_source": "human",
            "response_latency_seconds": latency,
        }
        if isinstance(last, AgentStep):
            new_trace[-1] = _copy_agent_step(last, update)
        elif isinstance(last, dict):
            last = dict(last)
            last.update(update)
            new_trace[-1] = last

    state.current_agent_state = _apply_state_update(
        agent_state,
        {
            "messages": list(agent_state.messages) + [HumanMessage(content=answer)],
            "trace": new_trace,
            "last_error": None,
        },
    )
    state.current_question = None
    state.current_question_started_at = None
    state.error_message = None
    return _continue_episode(state)


def submit_answer_for_current(
    experiment_state: HumanExperimentState,
    human_answer: str,
) -> HumanExperimentState:
    if experiment_state.current_episode is None:
        raise ValueError("No active sample. Start the current sample first.")
    experiment_state.current_episode = submit_human_answer(
        experiment_state.current_episode,
        human_answer,
    )
    if experiment_state.current_episode.finished:
        _record_finished_episode(experiment_state, experiment_state.current_episode)
    return experiment_state


def next_sample(
    experiment_state: HumanExperimentState,
    *,
    auto_start: bool = False,
) -> HumanExperimentState:
    if experiment_state.current_episode and not experiment_state.current_episode.finished:
        experiment_state.current_episode.error_message = (
            "現在のサンプルはまだ完了していません。完了後に次へ進んでください。"
        )
        return experiment_state
    if experiment_state.current_pos + 1 >= len(experiment_state.samples):
        experiment_state.current_episode = None
        return experiment_state
    experiment_state.current_pos += 1
    experiment_state.current_episode = None
    if auto_start:
        start_current_episode(experiment_state, restart=True)
    return experiment_state


def export_logs(experiment_state: HumanExperimentState) -> tuple[Path, list[Path]]:
    run_dir = experiment_state.run_dir
    results_dir = run_dir / "results"
    episodes_dir = run_dir / "episodes"
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    episode_paths: list[Path] = []
    for sample_id, episode in sorted(
        experiment_state.finished_episodes.items(),
        key=lambda item: item[1].sample_position,
    ):
        path = episodes_dir / f"sample{sample_id}_human_{experiment_state.config.run_id}.txt"
        path.write_text(_format_human_episode(episode), encoding="utf-8")
        episode_paths.append(path)

    participant = _safe_slug(experiment_state.config.participant_id or "anon")
    mode = _safe_slug(experiment_state.config.mode)
    results_path = (
        results_dir
        / f"human_{mode}_n{len(experiment_state.sample_ids)}_{participant}_{experiment_state.config.run_id}.json"
    )
    records = sorted(
        experiment_state.records_by_sample_id.values(),
        key=lambda r: r.get("sample_position", 0),
    )
    payload = {
        "n_processed": len(records),
        "clarify_quota": experiment_state.config.clarify_quota,
        "memo_window": experiment_state.config.memo_window,
        "mode": experiment_state.config.mode,
        "use_reflection": experiment_state.config.use_reflection,
        "user_source": "human",
        "participant_id": experiment_state.config.participant_id,
        "run_id": experiment_state.config.run_id,
        "timestamp": experiment_state.config.timestamp,
        "dataset_path": str(experiment_state.config.dataset_path),
        "n_samples": experiment_state.config.n_samples,
        "seed": experiment_state.config.seed,
        "sample_ids": experiment_state.config.sample_ids,
        "run_started_at": experiment_state.config.timestamp,
        "run_finished_at": _run_finished_at(experiment_state),
        "total_duration_seconds": _run_duration_seconds(experiment_state),
        "model": get_primary_model_name(),
        "sampling_scope": SAMPLING_SCOPE,
        "within_run_replacement": False,
        "across_run_replacement": True,
        "results": records,
    }
    if experiment_state.subset_path is not None:
        payload["sample_subset_path"] = str(experiment_state.subset_path)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    experiment_state.last_results_path = results_path
    experiment_state.last_export_message = (
        f"ログを書き出しました: {results_path} / episodes={len(episode_paths)}"
    )
    return results_path, episode_paths


def build_human_visible_context(state: HumanEpisodeState) -> dict[str, Any]:
    annotations = state.sample.get("annotations") or {}
    qa_history = []
    if state.current_agent_state:
        for step in state.current_agent_state.trace:
            next_decision = _step_get(step, "next_decision")
            if next_decision != "clarify":
                continue
            question = _step_get(step, "clarification_question")
            reply = _step_get(step, "user_reply")
            if question and question != state.current_question:
                qa_history.append({"question": question, "answer": reply})

    hidden_need = state.sample.get("reflective_action") if state.show_gold_to_user else None
    return {
        "run_id": state.run_id,
        "participant_id": state.participant_id,
        "sample_id": state.sample_id,
        "sample_position": state.sample_position,
        "total_samples": state.total_samples,
        "trial": state.current_trial_index,
        "initial_utterance": state.sample.get("utterance", ""),
        "user_specific_information": {
            "position": annotations.get("position"),
            "has": annotations.get("has") or [],
            "near_items": annotations.get("near_items") or [],
        },
        "environmental_context": {
            "sofa_front_table_items": annotations.get("sofa_front_table_items") or [],
            "kitchen_front_table_items": annotations.get("kitchen_front_table_items") or [],
            "kitchen_items": annotations.get("kitchen_items") or [],
        },
        "hidden_need": hidden_need,
        "clarification_question": state.current_question,
        "qa_history": qa_history,
        "answer_rules": (
            "状況情報と目標ニーズに基づいて、実際のユーザとして自然に答えてください。"
            "質問で明確に求められていない限り、エージェントに具体的な行動を直接命令しないでください。"
        ),
    }


def build_output_record(
    episode: HumanEpisodeState,
    config: HumanExperimentConfig,
) -> dict[str, Any]:
    states = episode.trial_states
    label_action_id = action_to_id.get(episode.sample.get("reflective_action"))
    steps_per_trial = [int(st.get("step", 0) or 0) for st in states]
    steps_detail_per_trial = [extract_trial_steps(st) for st in states]
    clarifications_per_trial = []
    for detail in steps_detail_per_trial:
        clarifications = []
        for step in detail:
            if step.get("action") != "clarify":
                continue
            item = {
                "t": step.get("t"),
                "clarification_question": step.get("clarification_question"),
                "clarification_question_words": step.get("clarification_question_words"),
                "user_reply": step.get("user_reply"),
                "user_reply_words": step.get("user_reply_words"),
                "user_source": step.get("user_source"),
            }
            if step.get("response_latency_seconds") is not None:
                item["response_latency_seconds"] = step.get("response_latency_seconds")
            clarifications.append(item)
        clarifications_per_trial.append(clarifications)

    pred = states[-1].get("action_id") if states else None
    correct = bool(label_action_id is not None and pred == label_action_id)
    return {
        "sample_index": episode.sample_id,
        "sample_position": episode.sample_position,
        "max_trials": episode.max_trials,
        "num_trials": len(states),
        "correct": correct,
        "steps_total": sum(steps_per_trial),
        "steps_per_trial": steps_per_trial,
        "steps_detail_per_trial": steps_detail_per_trial,
        "clarify_turns_per_trial": clarifications_per_trial,
        "gold_action": episode.sample.get("reflective_action"),
        "user_source": "human",
        "participant_id": config.participant_id,
        "run_id": config.run_id,
        "mode": config.mode,
        "use_reflection": config.use_reflection,
        "sampling_scope": SAMPLING_SCOPE,
        "sample_started_at": _format_epoch(episode.start_time),
        "sample_finished_at": _format_epoch(episode.finished_at),
        "sample_duration_seconds": _duration_seconds(
            episode.start_time,
            episode.finished_at,
        ),
        "response_latencies": episode.response_latencies,
        "human_visible_context": episode.human_visible_contexts,
    }


def _continue_episode(state: HumanEpisodeState) -> HumanEpisodeState:
    from src.graph import agent, clarify_node, execute_node

    while not state.finished:
        if state.current_agent_state is None:
            if state.current_trial_index >= state.max_trials:
                state.finished = True
                state.success = False
                state.finished_at = time.time()
                return state
            _start_next_trial(state)

        assert state.current_agent_state is not None
        agent_update = agent(state.current_agent_state)
        state.current_agent_state = _apply_state_update(
            state.current_agent_state, agent_update
        )

        if state.current_agent_state.last_error:
            state.error_message = (
                "モデル出力の解析に失敗したため、この試行では失敗行動として処理されました。"
            )

        if state.current_agent_state.next_decision == "clarify":
            clarify_update = clarify_node(state.current_agent_state)
            state.current_agent_state = _apply_state_update(
                state.current_agent_state, clarify_update
            )
            question = _last_ai_message_content(state.current_agent_state)
            if not question:
                state.error_message = "明確化質問を取得できませんでした。"
                state.current_agent_state = _apply_state_update(
                    state.current_agent_state,
                    {"next_decision": "execute", "action_id": -1},
                )
                continue
            state.current_question = question
            state.current_question_started_at = time.time()
            return state

        if state.current_agent_state.next_decision == "execute":
            execute_update = execute_node(state.current_agent_state)
            state.current_agent_state = _apply_state_update(
                state.current_agent_state, execute_update
            )
            _finalize_current_trial(state)
            if state.success or state.current_trial_index >= state.max_trials:
                state.finished = True
                state.current_question = None
                state.finished_at = time.time()
                return state
            state.current_agent_state = None
            state.current_question = None
            state.current_question_started_at = None
            continue

        state.error_message = "エージェントの状態を解釈できませんでした。"
        state.finished = True
        state.success = False
        state.finished_at = time.time()
        return state

    return state


def _start_next_trial(state: HumanEpisodeState) -> None:
    state.current_trial_index += 1
    prior_memos: list[str] = []
    extra_messages: list[SystemMessage] = []
    if state.use_reflection and state.reflection_memos:
        prior_memos = state.reflection_memos[-state.memo_window :]
        for idx, memo in enumerate(prior_memos, start=1):
            if state.mode in {
                "r-clarify",
                "r-clarify_non_cot",
                "cot_reflect",
                "reflect",
                "reflect_action_only",
            }:
                extra_messages.append(SystemMessage(content=f"【Reflection memo】\n{memo}"))
            else:
                distance = len(prior_memos) - idx + 1
                label = f"{distance} trial前"
                extra_messages.append(
                    SystemMessage(content=f"【Reflection memo（{label}）】\n{memo}")
                )

    user_context = make_system_prompt_from_one_sample(state.sample)
    reflective_action = state.sample.get("reflective_action")
    label_action_id = action_to_id.get(reflective_action) if reflective_action else None
    if label_action_id is None:
        raise ValueError(f"Unknown reflective_action: {reflective_action}")

    state.trial_contexts.append({"prior_memos": prior_memos if state.use_reflection else []})
    state.current_agent_state = AgentState(
        messages=[SystemMessage(content=user_context)] + extra_messages,
        next_decision=None,
        action_id=None,
        env=state.sample["annotations"],
        user_feedback=None,
        label_action_id=label_action_id,
        original_utterance=state.sample["utterance"],
        mode=state.mode,
        enable_reflection=state.use_reflection,
        user_context=user_context,
        clarify_quota_total=state.clarify_quota,
        clarify_quota_left=state.clarify_quota,
    )


def _finalize_current_trial(state: HumanEpisodeState) -> None:
    assert state.current_agent_state is not None
    final_state = _agent_state_to_dict(state.current_agent_state)
    pred = final_state.get("action_id")
    state.last_action_id = pred
    state.last_action_desc = id_to_action.get(pred, f"Unknown action_id={pred}")
    state.executed_actions.append(pred)

    label_action_id = final_state.get("label_action_id")
    success = bool(label_action_id is not None and pred == label_action_id)
    state.success = success
    is_last_trial = state.current_trial_index >= state.max_trials

    if state.use_reflection and (not success) and (not is_last_trial):
        usage_before = state.reflection_usage.snapshot()
        memo = generate_trial_reflection_memo(
            trial_index=state.current_trial_index,
            final_state=final_state,
            success=success,
            recent_memos=state.reflection_memos[-state.memo_window :],
            mode=state.mode,
            usage_collector=state.reflection_usage,
        )
        usage_after = state.reflection_usage.snapshot()
        delta = TokenUsageCollector.diff(usage_after, usage_before)
        final_state["reflection_token_usage"] = delta
        if memo:
            state.reflection_memos.append(memo)
            state.trial_memos.append(
                {"trial": state.current_trial_index, "new_memo": memo}
            )
            if len(state.reflection_memos) > state.memo_window:
                state.reflection_memos = state.reflection_memos[-state.memo_window :]

    state.trial_states.append(final_state)


def _record_finished_episode(
    experiment_state: HumanExperimentState,
    episode: HumanEpisodeState,
) -> None:
    experiment_state.finished_episodes[episode.sample_id] = episode
    experiment_state.records_by_sample_id[episode.sample_id] = build_output_record(
        episode,
        experiment_state.config,
    )
    export_logs(experiment_state)


def _format_human_episode(episode: HumanEpisodeState) -> str:
    lines = [
        "================ Human Run Metadata ================",
        f"Run ID: {episode.run_id}",
        f"Participant ID: {episode.participant_id or ''}",
        "User source: human",
        f"Mode: {episode.mode}",
        f"Use reflection: {episode.use_reflection}",
        f"Sample position: {episode.sample_position}/{episode.total_samples}",
        f"Sample ID: {episode.sample_id}",
        "",
        "================ 📉 User Context ================",
        "",
        make_system_prompt_from_one_sample(sample=episode.sample),
        "",
    ]
    for idx, trial_state in enumerate(episode.trial_states, start=1):
        context = episode.trial_contexts[idx - 1] if idx - 1 < len(episode.trial_contexts) else {}
        prior_memos = context.get("prior_memos") or []
        memo_for_trial = None
        for rec in episode.trial_memos:
            if rec.get("trial") == idx:
                memo_for_trial = rec.get("new_memo")
                break
        lines.extend(
            [
                f"---------------- ⚙️ Episode (trial {idx}/{episode.max_trials}) ----------------",
                "",
                format_single_trial_episode(trial_state, prior_memos=prior_memos),
            ]
        )
        if memo_for_trial:
            lines.extend(["", f"【Reflection memo after trial {idx}】", memo_for_trial])
        lines.append("")
    lines.append(f"Gold action: {episode.sample.get('reflective_action')}")
    return "\n".join(lines)


def _agent_state_to_dict(state: AgentState) -> dict[str, Any]:
    fields = getattr(type(state), "model_fields", None) or getattr(state, "__fields__", {})
    return {name: getattr(state, name) for name in fields}


def _apply_state_update(state: AgentState, update: dict[str, Any]) -> AgentState:
    if hasattr(state, "model_copy"):
        return state.model_copy(update=update)
    return state.copy(update=update)


def _copy_agent_step(step: AgentStep, update: dict[str, Any]) -> AgentStep:
    if hasattr(step, "model_copy"):
        return step.model_copy(update=update)
    return step.copy(update=update)


def _last_ai_message_content(state: AgentState) -> str | None:
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            return str(msg.content or "").strip()
    return None


def _step_get(step: AgentStep | dict[str, Any], key: str) -> Any:
    if isinstance(step, dict):
        return step.get(key)
    return getattr(step, key, None)


def _resolve_memo_window(*, mode: str, memo_window: int | None) -> int:
    if memo_window is not None:
        return max(1, memo_window)
    return 3 if mode == "r-clarify_reflexion" else 1


def _format_epoch(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _duration_seconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(max(0.0, end - start), 3)


def _all_samples_done(experiment_state: HumanExperimentState) -> bool:
    return bool(experiment_state.samples) and len(
        experiment_state.records_by_sample_id
    ) >= len(experiment_state.samples)


def _run_finished_at(experiment_state: HumanExperimentState) -> str | None:
    if not _all_samples_done(experiment_state):
        return None
    finished_times = [
        episode.finished_at
        for episode in experiment_state.finished_episodes.values()
        if episode.finished_at is not None
    ]
    if not finished_times:
        return None
    return _format_epoch(max(finished_times))


def _run_duration_seconds(experiment_state: HumanExperimentState) -> float | None:
    if not _all_samples_done(experiment_state):
        return None
    finished_times = [
        episode.finished_at
        for episode in experiment_state.finished_episodes.values()
        if episode.finished_at is not None
    ]
    if not finished_times:
        return None
    return _duration_seconds(experiment_state.run_start_time, max(finished_times))


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return slug.strip("_") or "anon"
