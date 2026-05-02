import json
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import ValidationError

from .state import AgentState, id_to_action, AgentOutput, AgentStep
from .prompts import agent_system_prompt, user_sim_system_prompt
from .llm_factory import get_primary_llm
from .gpt_llm import OpenAIChatModel

load_dotenv(".env", override=True)

# Main agent (decision/clarify/execute).
llm = get_primary_llm()

# User simulator (use OpenAI for stable answers).
user_llm = OpenAIChatModel(model="gpt-5.2")

_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```",
    re.MULTILINE,
)

_JSON_REPAIR_PROMPT = (
    "You repair malformed model output into valid JSON.\n"
    "Return exactly one JSON object only (no markdown, no extra text).\n"
    "Allowed keys: thought, next_decision, action_id, clarification_question.\n"
    "Rules:\n"
    '- next_decision must be "clarify" or "execute".\n'
    "- action_id must be an integer 1..40 when next_decision is execute, otherwise null.\n"
    "- clarification_question must be a single-sentence question when next_decision is clarify, otherwise null.\n"
)


def _trace_to_messages(trace: list[AgentStep]) -> list[SystemMessage]:
    """
    - Expose prior agent thinking/decisions to the next turn as system notes.
    - This may duplicate system messages, but is kept to preserve structure.
    """
    msgs: list[SystemMessage] = []
    for t in trace:
        thought = t.thought or "（thought なし）"
        if t.next_decision == "clarify":
            question_text = t.clarification_question or "（clarification_question なし）"
            reply_text = t.user_reply or "（user_reply なし）"
            decision_detail = (
                f"next_decision: clarify; action_id: None（確認を選択、まだ行動未決定）; "
                f"clarification_question: {question_text}; user_reply: {reply_text}"
            )
        else:
            action_desc = id_to_action.get(t.action_id, "不明な行動")
            decision_detail = (
                f"next_decision: {t.next_decision}; action_id: {t.action_id}（{action_desc}）"
            )
        msgs.append(
            SystemMessage(
                content=(
                    f"【Agent前回ログ step {t.step}】thought: {thought}; "
                    f"{decision_detail}"
                )
            )
        )
    return msgs


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Prefer a block that looks like JSON.
        for b in blocks:
            b2 = b.strip()
            if "{" in b2 and "}" in b2:
                return b2
        return blocks[0].strip()
    return text


def _sanitize_json_text(text: str) -> str:
    text = _strip_code_fence(text)
    # Remove control chars to avoid json.loads errors.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # The model sometimes inserts raw newlines; JSON does not allow them.
    text = text.replace("\r", " ").replace("\n", " ")
    # The model may append an extra single quote before a comma, breaking JSON.
    text = text.replace("\"',", "\",")
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """
    Find the first balanced {...} and ignore braces inside JSON strings.
    """
    start = None
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]

    return None


def _safe_agent_output(message: AIMessage) -> AgentOutput:
    raw_text = str(getattr(message, "content", "")).strip()

    def _parse_json(text: str) -> dict:
        cleaned = _sanitize_json_text(text)
        # Try direct parse first.
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                return json.loads(cleaned, strict=False)
            except Exception:
                # Extract the first balanced {...}.
                snippet = _extract_first_json_object(cleaned)
                if snippet:
                    return json.loads(snippet, strict=False)
                raise

    try:
        data = _parse_json(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Agent output must be a JSON object.")
        if "clarification_question" not in data and "question" in data:
            data["clarification_question"] = data.get("question")
        if isinstance(data.get("action_id"), str) and data["action_id"].strip().isdigit():
            data["action_id"] = int(data["action_id"].strip())
        if isinstance(data.get("clarification_question"), str):
            data["clarification_question"] = data["clarification_question"].strip()
        return AgentOutput(**data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(
            #"[Agent] Failed to parse model output; falling back to failure action."
            #f"Error: {exc}\nRaw output: {raw_text}"
        )
        # Key: relies on schema allowing -1.
        return AgentOutput(
            thought="解析失败。execute -1 にフォールバック。",
            next_decision="execute",
            action_id=-1,
            parse_error=f"{exc}",
        )


def _token_usage_from_message(message: AIMessage) -> dict[str, int]:
    try:
        usage = getattr(message, "additional_kwargs", {}).get("token_usage") or {}
    except Exception:
        return {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    if total <= 0 and prompt <= 0 and completion <= 0:
        return {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _merge_token_usage(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    if not first and not second:
        return {}
    return {
        "prompt_tokens": int(first.get("prompt_tokens", 0)) + int(second.get("prompt_tokens", 0)),
        "completion_tokens": int(first.get("completion_tokens", 0))
        + int(second.get("completion_tokens", 0)),
        "total_tokens": int(first.get("total_tokens", 0)) + int(second.get("total_tokens", 0)),
    }


def _repair_agent_output_once(raw_text: str) -> AIMessage:
    repair_messages = [
        SystemMessage(content=_JSON_REPAIR_PROMPT),
        HumanMessage(content=raw_text),
    ]
    return llm.invoke(repair_messages)


# ========== Agent Node ==========

def agent(state: AgentState) -> AgentState:

    # Simple step limit: if max_steps exceeded, force execute failure action -1
    if state.step >= state.max_steps:
        return {
            "next_decision": "execute",
            "action_id": -1,
            "step": state.max_steps,
            "clarification_question": None,
            "trace": state.trace + [AgentStep(
                step=state.step,
                thought="最大ステップ数を超過したため、失敗を判定します。",
                next_decision="execute",
                action_id=-1,
            )]
        }

    # Choose a prompt based on the mode.
    system_prompt = agent_system_prompt(
        state.mode,
        clarify_quota_total=state.clarify_quota_total,
        clarify_quota_left=state.clarify_quota_left,
    )

    # Let the agent see its own prior thoughts/decisions within the trial.
    trace_messages = _trace_to_messages(state.trace)

    base_messages = list(state.messages)

    # Check if the system prompt already exists for this trial.
    has_same_system: bool = (
        base_messages
        and isinstance(base_messages[0], SystemMessage)
        and base_messages[0].content == system_prompt
    )
    # If not the first entry, keep the existing system prompt and insert trace after it.
    if has_same_system:
        messages = [base_messages[0], *trace_messages, *base_messages[1:]]
    # If the first entry, add the system prompt.
    else:
        messages = [SystemMessage(content=system_prompt), *trace_messages, *base_messages]
    
    # Get the agent's raw output.
    raw_resp = llm.invoke(messages)

    # Record token usage.
    token_usage = _token_usage_from_message(raw_resp)
    
    # Parse agent's raw output and obtain clear agent's output.
    resp = _safe_agent_output(raw_resp)
    if resp.parse_error:
        repaired_raw_resp = _repair_agent_output_once(str(getattr(raw_resp, "content", "")))
        token_usage = _merge_token_usage(
            token_usage,
            _token_usage_from_message(repaired_raw_resp),
        )
        repaired = _safe_agent_output(repaired_raw_resp)
        if not repaired.parse_error:
            resp = repaired

    # Determine the agent's decision (Clarify / Execute).
    valid_decisions = {"clarify", "execute"}
    next_decision = (resp.next_decision or "").strip().lower()
    if next_decision not in valid_decisions:
        next_decision = "execute"

    # If action_id is invalid (null or out of 1-40), fall back to -1 to avoid errors.
    if isinstance(resp.action_id, int) and 1 <= resp.action_id <= 40:
        action_id = resp.action_id
    else:
        action_id = -1

    # If the agent chose Clarify, record the question.
    clarification_question = (
        (resp.clarification_question or "").strip() if next_decision == "clarify" else None
    )

    # Decide whether clarify is allowed based on mode:
    # - clarify / r-clarify / r-clarify_reflexion / reflect_action_only / non_thinking_clarify / r-clarify_non_cot: use model's clarify/execute
    # - direct / cot / cot_reflect / reflect: force execute only
    allow_clarify = state.mode in {
        "clarify",
        "r-clarify",
        "r-clarify_reflexion",
        "reflect_action_only",
        "non_thinking_clarify",
        "r-clarify_non_cot",
    }
    if allow_clarify and state.clarify_quota_left is not None:
        allow_clarify = state.clarify_quota_left > 0

    if not allow_clarify:
        next_decision = "execute"
        clarification_question = None

    # Update clarification quota; decrement if next decision is Clarify.
    clarify_quota_left = state.clarify_quota_left
    if next_decision == "clarify":
        if clarify_quota_left is not None:
            clarify_quota_left = max(clarify_quota_left - 1, 0)

    # Keep new_messages for future additions, even if none added now.
    new_messages = state.messages

    return {
        "next_decision": next_decision,
        "action_id": action_id,
        "step": state.step,  # Step increments in clarify/execute nodes only.
        "clarify_quota_left": clarify_quota_left,
        "clarify_quota_total": state.clarify_quota_total,
        "messages": new_messages,
        "clarification_question": clarification_question,
        "last_error": resp.parse_error,
        # Record the agent's action and append to trace; trace is used to build prompts.
        "trace": state.trace
        + [
            AgentStep(
                step=state.step + 1,  # Mark the upcoming action step number.
                thought=resp.thought,
                next_decision=next_decision,
                action_id=action_id,
                clarification_question=clarification_question,
                token_usage=token_usage or None,
            )
        ],
    }


# ========== Clarify Node ==========

def clarify_node(state: AgentState) -> AgentState:

    question_text = (state.clarification_question or "").strip()

    new_messages = list(state.messages)
    if not question_text:
        question_text = "エージェントが明確化質問を生成できなかったため、失敗として扱います。"
        return {
            "messages": new_messages + [AIMessage(content=question_text)],
            "next_decision": "execute",
            "action_id": -1,
            "step": state.max_steps,
            "clarification_question": None,
        }

    new_messages.append(AIMessage(content=question_text))

    return {
        "messages": new_messages,
        "step": state.step + 1,
        "next_decision": None,
        "action_id": None,
        "clarification_question": None,
    }


# ========== User Simulation Node ==========

def user_sim_node(state: AgentState) -> AgentState:

    action_id = state.label_action_id
    action_desc = id_to_action.get(action_id, "不明な行動")
    if action_id not in id_to_action:
        # If the label is unknown, fail immediately.
        fallback_msg = "ラベルが不明のため、失敗として扱います。"
        return {
            "messages": state.messages + [HumanMessage(content=fallback_msg)],
            "step": state.max_steps,
            "next_decision": "execute",
            "action_id": -1,
        }

    # Find the last AIMessage as the "clarification question"
    last_ai_msg = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break

    system_prompt = user_sim_system_prompt(
        action_desc=action_desc,
        utterance=state.original_utterance,
        env=state.env,
    )

    if last_ai_msg:
        question_text = last_ai_msg.content
    else:
        # If agent chose clarify but no question was generated, fail immediately.
        question_text = "エージェントが明確化質問が生成されなかったため、失敗として扱います。"
        return {
            "messages": state.messages + [HumanMessage(content=question_text)],
            "step": state.max_steps,       # Force termination on next agent call.
            "next_decision": "execute",
            "action_id": -1,
        }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question_text),
    ]

    resp = user_llm.invoke(messages)

    new_trace = list(state.trace)
    if new_trace:
        last = new_trace[-1]
        if isinstance(last, AgentStep):
            new_trace[-1] = last.copy(
                update={"user_reply": resp.content, "user_source": "simulated"}
            )
        elif isinstance(last, dict):
            last = dict(last)
            last["user_reply"] = resp.content
            last["user_source"] = "simulated"
            new_trace[-1] = last

    return {
        "messages": state.messages + [HumanMessage(content=resp.content)],
        "trace": new_trace,
    }


# ========== Execute Node (END) ==========

def execute_node(state: AgentState) -> AgentState:

    # If action_id is invalid (None, etc.), fall back to -1.
    action_id = state.action_id if state.action_id in id_to_action else -1
    desc = id_to_action[action_id]
    if action_id == -1:
        result_text = "動作を特定できません。"
    else:
        result_text = f"{desc}を執行しました。"

    return {
        "messages": state.messages + [AIMessage(content=result_text)],
        "step": state.step + 1,
    }


# ========== Build Graph ==========

builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("clarify", clarify_node)
builder.add_node("user_sim", user_sim_node)
builder.add_node("execute", execute_node)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    lambda s: s.next_decision if s.next_decision in {"clarify","execute"} else "execute",
    {"clarify": "clarify", "execute": "execute"},
)

builder.add_edge("clarify", "user_sim")
builder.add_edge("user_sim", "agent")

builder.add_edge("execute", END)

graph = builder.compile()
