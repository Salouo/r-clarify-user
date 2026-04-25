from pathlib import Path
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState, action_to_id, id_to_action
from .utils import (
    make_system_prompt_from_one_sample,
    format_single_trial_episode,
)
from .llm_factory import get_primary_llm, get_primary_model_dirname
from .prompts import trial_reflection_system_prompt
from .token_usage import TokenUsageCollector, set_usage_collector, clear_usage_collector


# LLM for reflection; agent and reflection share the same LLM.
reflection_llm = get_primary_llm()
model_dirname = get_primary_model_dirname()


def _format_episode_for_reflection(
    *,
    trial_index: int,
    final_state: dict,
) -> str:
    """
    Format the latest trial into the same episode format
    (user context + step logs).
    """
    user_ctx = final_state.get("user_context") or ""
    trial_body = format_single_trial_episode(final_state)
    return (
        "================ 📖 User Context ================\n\n"
        f"{user_ctx}\n\n"
        f"---------------- ⚙️ Episode (trial {trial_index}) ----------------\n\n"
        f"{trial_body}"
    )


def generate_trial_reflection_memo(
    *,
    trial_index: int,
    final_state: dict,
    success: bool,
    recent_memos: list[str],
    mode: str,
    usage_collector: TokenUsageCollector | None = None,
) -> Optional[str]:
    """
    Generate a reflection memo for the next trial without revealing the gold label,
    only indicating success or failure.
    """
    # 只将过往错误动作进行记录
    if mode == "reflect_action_only":
        pred_id = final_state.get("action_id")
        if pred_id in id_to_action:
            action_desc = id_to_action[pred_id]
        else:
            action_desc = f"Unknown action_id={pred_id}"
        actions: List[str] = []
        for memo in recent_memos:
            memo_text = memo.strip()
            if memo_text.startswith("過去の誤り行動:"):
                tail = memo_text.replace("過去の誤り行動:", "", 1).strip()
                if tail:
                    actions.extend([a.strip() for a in tail.split(" / ") if a.strip()])
            elif memo_text.startswith("前回の誤り行動:"):
                tail = memo_text.replace("前回の誤り行動:", "", 1).strip()
                if tail:
                    actions.append(tail)
        if action_desc not in actions:
            actions.append(action_desc)
        return f"過去の誤り行動: {' / '.join(actions)}"

    episode_text = _format_episode_for_reflection(trial_index=trial_index, final_state=final_state)
    prior_memos_text = "\n".join(f"- {m}" for m in recent_memos) if recent_memos else "なし"

    if mode == "r-clarify_reflexion":
        memo_instruction = (
            "次のトライアルのために、このトライアルの失敗や学びを3〜6行の箇条書きで新規メモとしてまとめてください。"
            "既存メモを上書きせず、今回の学びを簡潔に追加するイメージで。"
        )
        
    elif mode in {"r-clarify", "r-clarify_non_cot", "cot_reflect", "reflect"}:
        memo_instruction = (
            "次のトライアルのために、直近メモを踏まえて必要なら修正しつつ、3〜6行の箇条書きでまとめてください。"
        )

    human_prompt = (
        f"トライアル {trial_index} の完全ログ:\n"
        f"{episode_text}\n"
        "\n"
        f"- 成否: {'成功' if success else '失敗'}\n"
        f"- 直近の反省メモ（新しい順）:\n{prior_memos_text}\n"
        "\n"
        f"{memo_instruction}"
    )

    system_msg = SystemMessage(content=trial_reflection_system_prompt(mode))
    token = None

    if usage_collector is not None:
        token = set_usage_collector(usage_collector)    # Capture token usage here.
    try:
        resp = reflection_llm.invoke([system_msg, HumanMessage(content=human_prompt)])  # Call the LLM
    finally:
        if token is not None:
            clear_usage_collector(token)

    return resp.content if resp and resp.content else None


def run_one_trial(
    sample: dict,
    mode: str = "clarify",
    use_reflection: bool = True,
    clarify_quota: int | None = None,
    extra_system_messages: Optional[list] = None,
) -> dict:

    # Build the system prompt: user context.
    system_prompt = make_system_prompt_from_one_sample(sample)

    # initial_state: first message is SystemMessage.
    new_system_messages = [SystemMessage(content=system_prompt)] + (extra_system_messages or [])

    initial_state = AgentState(
        messages=new_system_messages,
        next_decision=None,
        action_id=None,
        env=sample["annotations"],
        user_feedback=None,
        label_action_id=action_to_id[sample["reflective_action"]],
        original_utterance=sample["utterance"],
        mode=mode,
        enable_reflection=use_reflection,
        user_context=system_prompt,
        clarify_quota_total=clarify_quota,
        clarify_quota_left=clarify_quota,
    )

    # Import `graph` to avoid circular dependencies.
    from .graph import graph

    # Run a single trial.
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 1000},   # Allow up to 1000 steps in the graph.
    )
    return final_state


# Important: entrance of the main process.
def run_trials_for_one_sample(
    sample: dict,
    use_reflection: bool = True,
    mode: str = "clarify",
    run_id: str | None = None,
    clarify_quota: int | None = None,
    max_trials: int = 1,
    memo_window: int | None = None,
) -> List[dict]:

    reflective_action = sample.get("reflective_action")
    label_action_id = action_to_id.get(reflective_action) if reflective_action else None

    all_states: List[dict] = []
    reflection_memos: List[str] = []
    trial_memos: List[dict] = []
    trial_contexts: List[dict] = []

    reflection_usage = TokenUsageCollector()

    # memo_window defaults to 3 only for r-clarify_reflexion.
    if memo_window is None:
        memo_window = 3 if mode == "r-clarify_reflexion" else 1

    # Iterate over all trials.
    for trial_idx in range(max_trials):
        extra_messages: List[SystemMessage] = []

        prior_memos: List[str] = []
        if use_reflection and reflection_memos:
            prior_memos = reflection_memos[-memo_window:]   # Take the last k reflection memos.
            for idx, memo in enumerate(prior_memos, start=1):
                # For r-clarify/reflect/cot_reflect/reflect_action_only, memo length is fixed at 1.
                if mode in {"r-clarify", "r-clarify_non_cot", "cot_reflect", "reflect", "reflect_action_only"}:
                    extra_messages.append(SystemMessage(content=f"【Reflection memo】\n{memo}"))
                # For r-clarify_reflexion, annotate each memo with its distance from the current trial.
                else:
                    distance = len(prior_memos) - idx + 1
                    label = f"{distance} trial前"
                    extra_messages.append(SystemMessage(content=f"【Reflection memo（{label}）】\n{memo}"))

        # Get the final_state for this trial.
        final_state = run_one_trial(
            sample,
            mode=mode,
            use_reflection=use_reflection,
            clarify_quota=clarify_quota,
            extra_system_messages=extra_messages,
        )
        # Save the current trial result.
        all_states.append(final_state)

        # Save reflection memos for r-clarify_reflexion prompts.
        trial_contexts.append(
            {
                "prior_memos": prior_memos if use_reflection else [],
            }
        )

        # Get the reflective action id from the last Execute.
        pred = final_state.get("action_id")
        # Determine whether the prediction is correct.
        success = bool(label_action_id is not None and pred == label_action_id)

        is_last_trial = (trial_idx + 1) >= max_trials

        if use_reflection and (not success) and (not is_last_trial):
            usage_before = reflection_usage.snapshot()

            # Generate a reflection memo from the current trial outcome.
            memo = generate_trial_reflection_memo(
                trial_index=trial_idx + 1,
                final_state=final_state,
                success=success,
                recent_memos=reflection_memos[-memo_window:],
                mode=mode,
                usage_collector=reflection_usage,
            )
            usage_after = reflection_usage.snapshot()
            delta = TokenUsageCollector.diff(usage_after, usage_before)
            if memo:
                reflection_memos.append(memo)
                trial_memos.append(
                    {
                        "trial": trial_idx + 1,
                        "new_memo": memo,
                    }
                )
                if len(reflection_memos) > memo_window:
                    reflection_memos = reflection_memos[-memo_window:]
                # Attach reflection token usage to the final state.
                try:
                    all_states[-1]["reflection_token_usage"] = delta
                except Exception:
                    pass

        # Exit early on success; otherwise continue until max_trials.
        if success:
            break

    # Record episodes.
    # episodes dir: <model>/histories/<mode> or <model>/histories/<mode>_<run_id>
    episodes_base = Path("outputs") / model_dirname
    if run_id:
        episodes_dir = episodes_base / "episodes" / f"{mode}_{run_id}"
    else:
        episodes_dir = episodes_base / "episodes" / mode
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Generate the episode file.
    episode_name = f"sample{sample['index']}.txt"
    episode_lines = [
        "================ 📖 User Context ================\n",
        make_system_prompt_from_one_sample(sample=sample),
        "",
    ]

    # Record all trial results.
    for idx, state in enumerate(all_states, start=1):
        context = trial_contexts[idx - 1] if idx - 1 < len(trial_contexts) else {}
        prior_memos = context.get("prior_memos") or []
        memo_for_trial = None
        for rec in trial_memos:
            if rec.get("trial") == idx:
                memo_for_trial = rec.get("new_memo")
                break
        episode_lines.extend(
            [
                f"---------------- ⚙️ Episode (trial {idx}/{max_trials}) ----------------\n",
                format_single_trial_episode(
                    state,
                    prior_memos=prior_memos,
                ),
            ]
        )
        if memo_for_trial:
            episode_lines.extend(
                [
                    f"\n【Reflection memo after trial {idx}】",
                    memo_for_trial,
                ]
            )
        episode_lines.append("")

    # Save the gold action only when finalizing the episode.
    episode_lines.append(f"Gold action: {sample.get('reflective_action')}")

    episode_str = "\n".join(episode_lines)
    (episodes_dir / episode_name).write_text(episode_str, encoding="utf-8")

    return all_states
