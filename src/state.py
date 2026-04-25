from typing import Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# Action table shared by graph / utils / main.
id_to_action: dict[int, str] = {
    -1: "動作を特定できない",
    1: "[1]バナナを持ってくる",
    2: "[2]充電ケーブルを持ってくる",
    3: "[3]コップを持ってくる",
    4: "[4]ケチャップを持ってくる",
    5: "[5]宅配便を持ってくる",
    6: "[6]ペットボトルを持ってくる",
    7: "[7]リモコンを持ってくる",
    8: "[8]スマホを持ってくる",
    9: "[9]お菓子を持ってくる",
    10: "[10]ティッシュ箱を持ってくる",
    11: "[11]充電ケーブルを片付ける",
    12: "[12]コップを片付ける",
    13: "[13]ケチャップを片付ける",
    14: "[14]ミニカーを片付ける",
    15: "[15]ペットボトルを片付ける",
    16: "[16]リモコンを片付ける",
    17: "[17]スマホを片付ける",
    18: "[18]お菓子を片付ける",
    19: "[19]ティッシュ箱を片付ける",
    20: "[20]ゴミをゴミ箱に入れる",
    21: "[21]缶切りを持ってくる",
    22: "[22]クッキングシートを持ってくる",
    23: "[23]グラスを持ってくる",
    24: "[24]おろし器を持ってくる",
    25: "[25]キッチンペーパーを持ってくる",
    26: "[26]レモンを持ってくる",
    27: "[27]オリーブオイルを持ってくる",
    28: "[28]じゃがいもを持ってくる",
    29: "[29]サランラップを持ってくる",
    30: "[30]水筒を持ってくる",
    31: "[31]缶切りを棚にしまう",
    32: "[32]クッキングシートを棚にしまう",
    33: "[33]グラスを棚にしまう",
    34: "[34]おろし器を棚にしまう",
    35: "[35]キッチンペーパーを棚にしまう",
    36: "[36]ペットボトルを冷蔵庫にしまう",
    37: "[37]サランラップを棚にしまう",
    38: "[38]タッパーをレンジに入れる",
    39: "[39]タッパーを冷蔵庫にしまう",
    40: "[40]水筒を棚にしまう",
}

action_to_id: dict[str, int] = {v: k for k, v in id_to_action.items()}

# All available decisions.
DecisionType = Literal["clarify", "execute"]

# Agent output for each step.
class AgentStep(BaseModel):
    step: int                       # Step number.
    thought: Optional[str] = None   # LLM thinking; may be None in direct mode.
    next_decision: DecisionType     # clarify / execute
    action_id: Optional[int] = None # Action chosen when executing.
    clarification_question: Optional[str] = None  # Question text generated in clarify.
    user_reply: Optional[str] = None    # User reply text in clarify.
    token_usage: Optional[dict] = None  # Token usage for this agent call (prompt/completion/total).


class AgentOutput(BaseModel):
    thought: Optional[str] = Field(
        default=None,
        description="你对于当前局面，以及对于接下来应该如何决策的思考；direct 模式下可以为 null"
    )
    next_decision: DecisionType = Field(
        description="你接下来要执行的决策，只能是 clarify / execute 之一"
    )
    action_id: Optional[int] = Field(
        default=None,
        description="当你选择 next_decision 为 execute 时，必须是 1 到 40 的整数; clarify 时为 None",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="当你选择 next_decision 为 clarify 时，必须填写 1 条澄清问题; execute 时为 null",
    )


# ========== Graph State ==========

class AgentState(BaseModel):

    # Dialogue history: real/simulated user + agent messages.
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Next decision chosen by the agent.
    next_decision: Optional[DecisionType] = Field(default=None,)

    # If next_decision == "execute", this is 1~40; otherwise None.
    action_id: Optional[int] = Field(default=None)

    # External environment info (sample env), kept for future use.
    env: dict[str, Any] = Field(default_factory=dict)

    # User feedback (may be used for reflection later).
    user_feedback: Optional[str] = Field(default=None)

    # Ground-truth target action for this sample (label).
    label_action_id: int = Field(default=None)

    # Original ambiguous utterance (for user_sim).
    original_utterance: str = Field(default="")

    # Current run mode (clarify / r-clarify / r-clarify_reflexion / reflect / reflect_action_only / non_thinking_clarify / r-clarify_non_cot / cot / cot_reflect / direct).
    mode: str = Field(
        default="clarify",
        description="当前样本使用的运行模式，例如 'clarify' / 'r-clarify' / 'r-clarify_reflexion' / 'reflect' / 'reflect_action_only' / 'non_thinking_clarify' / 'r-clarify_non_cot' / 'cot' / 'cot_reflect' / 'direct'",
    )

    # Step counter.
    step: int = Field(default=0, description="当前执行的 step 计数（从 0 开始）")

    # Maximum steps.
    max_steps: int = Field(default=1000,description="最大允许的 step 数，超过后应强制结束或执行")

    # Trace of each trial.
    trace: list[AgentStep] = Field(default_factory=list, description="agent 每一步的思考和决策轨迹")

    # Whether to run reflection after clarification to summarize user replies.
    enable_reflection: bool = Field(default=False)

    # Preprocessed user context for post-clarification reflection.
    user_context: str = Field(default="")

    # Clarification quota (total and remaining); None means unlimited.
    clarify_quota_total: Optional[int] = Field(
        default=None, description="本次 trial 允许澄清的最大次数（None 表示不限）"
    )
    clarify_quota_left: Optional[int] = Field(
        default=None, description="当前剩余的澄清次数（None 表示不限）"
    )

    # Clarification question generated when agent chooses clarify (recorded by clarify node).
    clarification_question: Optional[str] = Field(
        default=None, description="agent 选择 clarify 时生成的一句话澄清问题"
    )
