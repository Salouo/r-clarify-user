from __future__ import annotations

import argparse
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from human_loop import (
    HumanExperimentState,
    export_logs,
    load_human_subset,
    next_sample,
    start_current_episode,
    submit_answer_for_current,
)


MODES = [
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
]


@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    button_label: str
    mode: str
    use_reflection: bool
    run_suffix: str
    output_stem: str


@dataclass
class HumanAppState:
    section_states: dict[str, HumanExperimentState]
    current_section: str | None = None
    finalized_sections: set[str] = field(default_factory=set)


SECTION_SPECS = (
    SectionSpec(
        key="with_reflection",
        title="Section 1",
        button_label="Section 1",
        mode="r-clarify",
        use_reflection=True,
        run_suffix="with_reflection",
        output_stem="reflection",
    ),
    SectionSpec(
        key="without_reflection",
        title="Section 2",
        button_label="Section 2",
        mode="clarify",
        use_reflection=False,
        run_suffix="without_reflection",
        output_stem="without_reflection",
    ),
)


NEXT_SAMPLE_SOUND_INIT_JS = r"""
(() => {
  if (window.__rClarifySoundInitialized) {
    return;
  }
  window.__rClarifySoundInitialized = true;
  window.__rClarifyLastNextCue = "";

  const getAudioContext = () => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) {
      return null;
    }
    if (!window.__rClarifyAudioContext) {
      window.__rClarifyAudioContext = new AudioContext();
    }
    return window.__rClarifyAudioContext;
  };

  const unlockAudio = () => {
    const ctx = getAudioContext();
    if (ctx && ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }
  };

  window.addEventListener("pointerdown", unlockAudio, { capture: true });
  window.addEventListener("keydown", unlockAudio, { capture: true });

  window.__rClarifyPlayNextSampleCue = (cue) => {
    if (!cue || cue === window.__rClarifyLastNextCue) {
      return;
    }
    window.__rClarifyLastNextCue = cue;

    const ctx = getAudioContext();
    if (!ctx) {
      return;
    }

    const play = () => {
      const start = ctx.currentTime + 0.01;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.16, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.42);
      gain.connect(ctx.destination);

      [880, 1174.66].forEach((frequency, index) => {
        const oscillator = ctx.createOscillator();
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(frequency, start + index * 0.12);
        oscillator.connect(gain);
        oscillator.start(start + index * 0.12);
        oscillator.stop(start + index * 0.12 + 0.18);
      });
    };

    if (ctx.state === "suspended") {
      ctx.resume().then(play).catch(() => {});
    } else {
      play();
    }
  };
})();
"""


NEXT_SAMPLE_SOUND_CHANGE_JS = """
(cue) => {
  if (window.__rClarifyPlayNextSampleCue) {
    window.__rClarifyPlayNextSampleCue(cue);
  }
  return [];
}
"""


def build_app(initial_state: HumanAppState | HumanExperimentState) -> gr.Blocks:
    app_state = _coerce_app_state(initial_state)
    with gr.Blocks(title="R-Clarify Human Experiment") as demo:
        state = gr.State(app_state)

        gr.Markdown(
            "## R-Clarify ヒューマン実験\n\n"
            "画面に表示される状況情報と目標ニーズに基づいて、実際のユーザとしてエージェントの質問に答えてください。"
            "自分の状態・好み・困っていることを自然に説明してください。"
            "質問で明確に求められていない限り、エージェントに具体的な行動を直接命令しないでください。"
        )

        with gr.Column():
            landing_status = gr.Markdown()
            with gr.Row():
                with_reflection_btn = gr.Button(
                    _section_spec("with_reflection").button_label,
                    variant="primary",
                )
                without_reflection_btn = gr.Button(
                    _section_spec("without_reflection").button_label,
                    variant="primary",
                )

        with gr.Column():
            progress = gr.Markdown()

            with gr.Row():
                participant_id = gr.Textbox(label="参加者ID", interactive=False)
                run_id = gr.Textbox(label="実行ID", interactive=False)
                sample_id = gr.Textbox(label="サンプルID", interactive=False)

            with gr.Row():
                initial_utterance = gr.Textbox(
                    label="最初の発話",
                    lines=2,
                    interactive=False,
                )
                hidden_need = gr.Textbox(
                    label="目標ニーズ（エージェントには表示されません）",
                    lines=2,
                    interactive=False,
                    visible=_first_section_state(app_state).config.show_gold_to_user,
                )

            with gr.Row():
                user_info = gr.Textbox(label="ユーザ情報", lines=5, interactive=False)
                env_info = gr.Textbox(label="環境情報", lines=7, interactive=False)

            question = gr.Textbox(
                label="エージェントの明確化質問",
                lines=3,
                interactive=False,
            )
            qa_history = gr.Textbox(
                label="現在の試行内の質問・回答履歴",
                lines=6,
                interactive=False,
            )
            chatbot = _make_chatbot()

            answer = gr.Textbox(
                label="回答",
                placeholder="ここに回答を入力してください",
                lines=3,
            )

            with gr.Row():
                start_btn = gr.Button(
                    _start_button_label(_first_section_state(app_state)),
                    variant="secondary",
                )
                send_btn = gr.Button("回答を送信", variant="secondary")
                next_btn = gr.Button("次のサンプル")
                finish_btn = gr.Button("完了")

            with gr.Row():
                action = gr.Textbox(label="実行された行動", lines=2, interactive=False)
                trial_status = gr.Textbox(label="現在の状態", lines=4, interactive=False)

            status = gr.Textbox(label="メッセージ", lines=3, interactive=False)
            next_sample_audio_cue = gr.Textbox(value="", visible=False)

        outputs = [
            state,
            landing_status,
            with_reflection_btn,
            without_reflection_btn,
            progress,
            participant_id,
            run_id,
            sample_id,
            initial_utterance,
            hidden_need,
            user_info,
            env_info,
            question,
            qa_history,
            chatbot,
            answer,
            start_btn,
            send_btn,
            next_btn,
            finish_btn,
            action,
            trial_status,
            status,
            next_sample_audio_cue,
        ]

        def on_select(app_state: HumanAppState, section_key: str):
            if section_key in app_state.finalized_sections:
                return _render_app(app_state, "このsectionはすでに完了しています。")
            if section_key not in app_state.section_states:
                return _render_app(app_state, "sectionを選択できませんでした。")
            locked_by = _first_unfinished_previous_section(app_state, section_key)
            if locked_by is not None:
                return _render_app(
                    app_state,
                    f"先にSection {_section_number(locked_by.key)}を完了してください。",
                )
            app_state.current_section = section_key
            title = _section_spec(section_key).title
            return _render_app(app_state, f"{title}を選択しました。")

        def on_start(app_state: HumanAppState):
            exp_state = _active_section_state(app_state)
            if exp_state is None:
                return _render_app(app_state, "先にsectionを選択してください。")
            try:
                exp_state = start_current_episode(exp_state, restart=True)
                _set_active_section_state(app_state, exp_state)
                return _render_app(app_state, "現在のサンプルを開始しました。")
            except Exception as exc:
                return _render_app(app_state, f"開始できませんでした: {exc}")

        def on_start_processing():
            return (
                "開始中です。しばらくお待ちください。",
                gr.update(value="", interactive=False),
                gr.update(value="回答を送信", interactive=False, variant="secondary"),
                gr.update(value="開始中...", interactive=False, variant="primary"),
                gr.update(interactive=False, variant="secondary"),
                gr.update(interactive=False, variant="secondary"),
                "処理中です。",
                "サンプルを開始中です。",
            )

        def on_submit(app_state: HumanAppState, human_answer: str):
            exp_state = _active_section_state(app_state)
            if exp_state is None:
                return _render_app(app_state, "先にsectionを選択してください。")
            try:
                exp_state = submit_answer_for_current(exp_state, human_answer)
                _set_active_section_state(app_state, exp_state)
                return _render_app(app_state, "回答を送信しました。")
            except Exception as exc:
                return _render_app(app_state, f"回答を送信できませんでした: {exc}")

        def on_submit_processing():
            return (
                "処理中です。しばらくお待ちください。",
                gr.update(interactive=False),
                gr.update(value="処理中...", interactive=False, variant="primary"),
                gr.update(interactive=False, variant="secondary"),
                gr.update(interactive=False, variant="secondary"),
                gr.update(interactive=False, variant="secondary"),
                "処理中です。",
                "回答を送信中です。",
            )

        def on_next(app_state: HumanAppState):
            exp_state = _active_section_state(app_state)
            if exp_state is None:
                return _render_app(app_state, "先にsectionを選択してください。")
            if exp_state.current_episode is None:
                return _render_app(app_state, "まず現在のサンプルを開始してください。")
            before_pos = exp_state.current_pos
            exp_state = next_sample(exp_state, auto_start=True)
            _set_active_section_state(app_state, exp_state)
            if exp_state.current_pos == before_pos:
                if before_pos + 1 >= len(exp_state.samples) and (
                    exp_state.current_episode is None
                    or exp_state.current_episode.finished
                ):
                    return _render_app(app_state, "すべてのサンプルが完了しました。")
                return _render_app(app_state, "現在のサンプルはまだ完了していません。")
            return _render_app(app_state, "次のサンプルを開始しました。")

        def on_next_processing():
            return (
                "次のサンプルへ移動中です。しばらくお待ちください。",
                gr.update(value="", interactive=False),
                gr.update(value="回答を送信", interactive=False, variant="secondary"),
                gr.update(interactive=False, variant="secondary"),
                gr.update(value="移動中...", interactive=False, variant="primary"),
                gr.update(interactive=False, variant="secondary"),
                "次のサンプルを準備中です。",
                "移動中です。",
            )

        def on_finish(app_state: HumanAppState):
            exp_state = _active_section_state(app_state)
            if exp_state is None:
                return _render_app(app_state, "先にsectionを選択してください。")
            if not _all_samples_done(exp_state):
                return _render_app(app_state, "まだすべてのサンプルが完了していません。")
            try:
                results_path, episode_paths = export_logs(exp_state)
                section_key = app_state.current_section
                if section_key is not None:
                    app_state.finalized_sections.add(section_key)
                app_state.current_section = None
                if _all_sections_finalized(app_state):
                    lead = "2つのsectionが完了しました。この画面を閉じてください。"
                else:
                    lead = "このsectionが完了しました。続けてもう一つのsectionを選択してください。"
                msg = (
                    f"{lead}\n"
                    f"出力JSON: {results_path}\n"
                    f"エピソード数: {len(episode_paths)}"
                )
                return _render_app(app_state, msg)
            except Exception as exc:
                return _render_app(app_state, f"終了処理中にログを書き出せませんでした: {exc}")

        demo.load(
            lambda app_state: _render_app(app_state),
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )
        with_reflection_btn.click(
            lambda app_state: on_select(app_state, "with_reflection"),
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )
        without_reflection_btn.click(
            lambda app_state: on_select(app_state, "without_reflection"),
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )
        busy_outputs = [
            status,
            answer,
            send_btn,
            start_btn,
            next_btn,
            finish_btn,
            action,
            trial_status,
        ]
        start_btn.click(
            on_start_processing,
            inputs=None,
            outputs=busy_outputs,
            queue=False,
        ).then(
            on_start,
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )
        send_btn.click(
            on_submit_processing,
            inputs=None,
            outputs=busy_outputs,
            queue=False,
        ).then(
            on_submit,
            inputs=[state, answer],
            outputs=outputs,
            show_progress_on=[status],
        )
        answer.submit(
            on_submit_processing,
            inputs=None,
            outputs=busy_outputs,
            queue=False,
        ).then(
            on_submit,
            inputs=[state, answer],
            outputs=outputs,
            show_progress_on=[status],
        )
        next_btn.click(
            on_next_processing,
            inputs=None,
            outputs=busy_outputs,
            queue=False,
        ).then(
            on_next,
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )
        next_sample_audio_cue.change(
            fn=None,
            inputs=[next_sample_audio_cue],
            outputs=None,
            js=NEXT_SAMPLE_SOUND_CHANGE_JS,
            queue=False,
            show_progress="hidden",
        )
        finish_btn.click(
            on_finish,
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )

    return demo


def _render_app(app_state: HumanAppState, notice: str = ""):
    has_active_section = app_state.current_section is not None
    if has_active_section:
        exp_state = _active_section_state(app_state) or _first_section_state(app_state)
        _, *experiment_values = _render(exp_state, notice)
    else:
        experiment_values = _empty_experiment_values(app_state, notice)
    landing_notice = notice if not has_active_section else ""

    return (
        app_state,
        _landing_status_text(app_state, landing_notice),
        _section_button_update(app_state, "with_reflection"),
        _section_button_update(app_state, "without_reflection"),
        *experiment_values,
    )


def _empty_experiment_values(app_state: HumanAppState, notice: str = ""):
    exp_state = _first_section_state(app_state)
    status = notice or "sectionを選択してください。"
    return (
        "**Section:** 未選択",
        exp_state.config.participant_id or "",
        "",
        "",
        "",
        gr.update(value="", visible=exp_state.config.show_gold_to_user),
        "",
        "",
        "sectionを選択してください。",
        "まだ履歴はありません。",
        [],
        gr.update(value="", interactive=False),
        gr.update(value="開始", interactive=False, variant="secondary"),
        gr.update(value="回答を送信", interactive=False, variant="secondary"),
        gr.update(interactive=False, variant="secondary"),
        gr.update(interactive=False, variant="secondary"),
        "",
        "サンプルはまだ開始されていません。",
        status,
        "",
    )


def _render(exp_state: HumanExperimentState, notice: str = ""):
    sample = _current_sample(exp_state)
    episode = exp_state.current_episode
    config = exp_state.config
    progress_text = _progress_text(exp_state)

    sample_id = str(sample.get("index", "")) if sample else ""
    initial_utterance = str(sample.get("utterance", "")) if sample else ""
    hidden_need_value = str(sample.get("reflective_action", "")) if sample else ""
    hidden_update = gr.update(
        value=hidden_need_value if config.show_gold_to_user else "",
        visible=config.show_gold_to_user,
    )
    user_info = _format_user_info(sample)
    env_info = _format_env_info(sample)

    question = "まだ質問はありません。"
    qa_history = "まだ履歴はありません。"
    chat_messages: list[dict[str, str]] = []
    action_text = "まだ実行されていません。"
    trial_status = _trial_status(exp_state)
    send_interactive = False

    if episode:
        if episode.current_question:
            question = episode.current_question
            send_interactive = not episode.finished
            if not notice and not episode.finished:
                notice = "エージェントの質問に回答してください。"
        qa_history = _format_qa_history(episode)
        chat_messages = _chat_messages(episode)
        if episode.last_action_desc:
            action_text = episode.last_action_desc
        if episode.error_message:
            notice = f"{notice}\n{episode.error_message}".strip()

    if not notice:
        notice = (
            "すべてのサンプルが完了しました。「完了」を押してください。"
            if _all_samples_done(exp_state)
            else "準備ができています。"
        )

    answer_update = gr.update(value="", interactive=send_interactive)
    all_done = _all_samples_done(exp_state)
    start_label = "開始済み" if episode else _start_button_label(exp_state)
    start_interactive = not all_done and episode is None
    start_update = gr.update(
        value=start_label,
        interactive=start_interactive,
        variant="primary" if start_interactive else "secondary",
    )
    send_update = gr.update(
        value="回答を送信",
        interactive=send_interactive,
        variant="primary" if send_interactive else "secondary",
    )
    next_variant = "primary" if episode and episode.finished else "secondary"
    next_update = gr.update(
        value="次のサンプル",
        interactive=bool(episode and episode.finished and not all_done),
        variant=next_variant,
    )
    finish_update = gr.update(
        interactive=all_done,
        variant="primary" if all_done else "secondary",
    )
    next_sample_audio_cue = _next_sample_audio_cue(exp_state)

    return (
        exp_state,
        progress_text,
        config.participant_id or "",
        config.run_id,
        sample_id,
        initial_utterance,
        hidden_update,
        user_info,
        env_info,
        question,
        qa_history,
        chat_messages,
        answer_update,
        start_update,
        send_update,
        next_update,
        finish_update,
        action_text,
        trial_status,
        notice,
        next_sample_audio_cue,
    )


def _current_sample(exp_state: HumanExperimentState) -> dict[str, Any] | None:
    if not exp_state.samples:
        return None
    pos = min(exp_state.current_pos, len(exp_state.samples) - 1)
    return exp_state.samples[pos]


def _all_samples_done(exp_state: HumanExperimentState) -> bool:
    return bool(exp_state.samples) and len(exp_state.records_by_sample_id) >= len(
        exp_state.samples
    )


def _next_sample_audio_cue(exp_state: HumanExperimentState) -> str:
    episode = exp_state.current_episode
    if not episode or not episode.finished or _all_samples_done(exp_state):
        return ""
    return f"{exp_state.config.run_id}:{exp_state.current_pos}:{episode.sample_id}"


def _start_button_label(exp_state: HumanExperimentState) -> str:
    return "開始" if exp_state.current_pos == 0 else "現在のサンプルを再開始"


def _make_chatbot():
    kwargs = {"label": "対話", "height": 260}
    if "type" in inspect.signature(gr.Chatbot).parameters:
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


def _progress_text(exp_state: HumanExperimentState) -> str:
    total = len(exp_state.samples)
    current = min(exp_state.current_pos + 1, total) if total else 0
    completed = len(exp_state.records_by_sample_id)
    mode_label = _mode_label(exp_state)
    return (
        f"**Section:** {mode_label}　"
        f"**進捗:** {current} / {total}　"
        f"**完了:** {completed}　"
        f"**抽出seed:** {exp_state.config.seed}"
    )


def _trial_status(exp_state: HumanExperimentState) -> str:
    episode = exp_state.current_episode
    if not episode:
        return "サンプルはまだ開始されていません。"
    clarify_count = 0
    if episode.current_agent_state:
        clarify_count = sum(
            1
            for step in episode.current_agent_state.trace
            if _step_get(step, "next_decision") == "clarify"
        )
    result = "成功" if episode.success else "未成功"
    finished = "完了" if episode.finished else "進行中"
    quota = "制限なし" if episode.clarify_quota is None else str(episode.clarify_quota)
    return (
        f"試行: {episode.current_trial_index} / {episode.max_trials}\n"
        f"明確化回数: {clarify_count} / {quota}\n"
        f"結果: {result}\n"
        f"状態: {finished}"
    )


def _format_user_info(sample: dict[str, Any] | None) -> str:
    if not sample:
        return ""
    ann = sample.get("annotations") or {}
    return "\n".join(
        [
            f"位置: {_format_value(ann.get('position'))}",
            f"手にしている物: {_format_value(ann.get('has'))}",
            f"近くにある物: {_format_value(ann.get('near_items'))}",
        ]
    )


def _format_env_info(sample: dict[str, Any] | None) -> str:
    if not sample:
        return ""
    ann = sample.get("annotations") or {}
    return "\n".join(
        [
            f"ソファ前テーブルの物品: {_format_value(ann.get('sofa_front_table_items'))}",
            f"キッチン前テーブルの物品: {_format_value(ann.get('kitchen_front_table_items'))}",
            f"キッチンの物品: {_format_value(ann.get('kitchen_items'))}",
            "表示されている物品は現在見えている物だけです。リストにない物でも、別の場所から持ってくることがあります。",
        ]
    )


def _format_qa_history(episode) -> str:
    rows = []
    if episode.current_agent_state:
        for step in episode.current_agent_state.trace:
            if _step_get(step, "next_decision") != "clarify":
                continue
            question = _step_get(step, "clarification_question")
            reply = _step_get(step, "user_reply")
            if question:
                rows.append(f"Q: {question}")
            if reply:
                rows.append(f"A: {reply}")
    return "\n".join(rows) if rows else "まだ履歴はありません。"


def _chat_messages(episode) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not episode.current_agent_state:
        return messages
    for msg in episode.current_agent_state.messages:
        if isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": str(msg.content)})
        elif isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": str(msg.content)})
    return messages


def _step_get(step, key: str):
    if isinstance(step, dict):
        return step.get(key)
    return getattr(step, key, None)


def _format_value(value) -> str:
    if value is None or value == "" or value == []:
        return "なし"
    if isinstance(value, list):
        return "、".join(str(v) for v in value) if value else "なし"
    return str(value)


def _mode_label(exp_state: HumanExperimentState) -> str:
    for spec in SECTION_SPECS:
        if (
            exp_state.config.mode == spec.mode
            and exp_state.config.use_reflection == spec.use_reflection
        ):
            return spec.title
    return exp_state.config.mode


def _coerce_app_state(
    initial_state: HumanAppState | HumanExperimentState,
) -> HumanAppState:
    if isinstance(initial_state, HumanAppState):
        return initial_state
    key = "with_reflection" if initial_state.config.use_reflection else "without_reflection"
    return HumanAppState(section_states={key: initial_state}, current_section=key)


def _section_spec(section_key: str) -> SectionSpec:
    for spec in SECTION_SPECS:
        if spec.key == section_key:
            return spec
    raise KeyError(f"Unknown section: {section_key}")


def _first_section_state(app_state: HumanAppState) -> HumanExperimentState:
    for spec in SECTION_SPECS:
        state = app_state.section_states.get(spec.key)
        if state is not None:
            return state
    return next(iter(app_state.section_states.values()))


def _active_section_state(app_state: HumanAppState) -> HumanExperimentState | None:
    if app_state.current_section is None:
        return None
    return app_state.section_states.get(app_state.current_section)


def _set_active_section_state(
    app_state: HumanAppState,
    exp_state: HumanExperimentState,
) -> None:
    if app_state.current_section is not None:
        app_state.section_states[app_state.current_section] = exp_state


def _all_sections_finalized(app_state: HumanAppState) -> bool:
    return all(spec.key in app_state.finalized_sections for spec in SECTION_SPECS)


def _section_number(section_key: str) -> int:
    for idx, spec in enumerate(SECTION_SPECS, start=1):
        if spec.key == section_key:
            return idx
    raise KeyError(f"Unknown section: {section_key}")


def _first_unfinished_previous_section(
    app_state: HumanAppState,
    section_key: str,
) -> SectionSpec | None:
    for spec in SECTION_SPECS:
        if spec.key == section_key:
            return None
        if spec.key in app_state.section_states and spec.key not in app_state.finalized_sections:
            return spec
    return None


def _section_button_update(app_state: HumanAppState, section_key: str):
    spec = _section_spec(section_key)
    finalized = section_key in app_state.finalized_sections
    state = app_state.section_states.get(section_key)
    has_active = app_state.current_section is not None
    locked_by = _first_unfinished_previous_section(app_state, section_key)
    label = spec.button_label
    if finalized:
        label = f"{label}（完了）"
    elif locked_by is not None:
        label = f"{label}（Section {_section_number(locked_by.key)}完了後）"
    elif state is not None and len(state.records_by_sample_id) > 0:
        label = f"{label}（途中）"
    return gr.update(
        value=label,
        interactive=(not finalized and not has_active and locked_by is None),
    )


def _landing_status_text(app_state: HumanAppState, notice: str = "") -> str:
    lines = ["### sectionを選択してください"]
    for spec in SECTION_SPECS:
        exp_state = app_state.section_states.get(spec.key)
        if exp_state is None:
            status = "未準備"
            run_id = ""
        else:
            completed = len(exp_state.records_by_sample_id)
            total = len(exp_state.samples)
            if spec.key in app_state.finalized_sections:
                status = f"完了（{completed}/{total}）"
            elif completed:
                status = f"途中（{completed}/{total}）"
            else:
                status = f"未開始（0/{total}）"
            run_id = exp_state.config.run_id
        lines.append(f"- {spec.title}: {status}  \n  実行ID: `{run_id}`")
    if notice:
        lines.append(f"\n{notice}")
    elif _all_sections_finalized(app_state):
        lines.append("\nすべてのsectionが完了しました。")
    return "\n".join(lines)


def _build_app_state(args: argparse.Namespace) -> HumanAppState:
    requested_base_run_id = args.run_id or args.participant_id or "anon"
    base_run_id = _unique_base_run_id(args.output_dir, requested_base_run_id)
    section_states: dict[str, HumanExperimentState] = {}
    selected_sample_ids: list[int] | None = None
    selected_seed: int | None = args.seed

    for spec in SECTION_SPECS:
        subset_output_path = _section_subset_output_path(
            args.subset_output_path,
            spec.run_suffix,
        )
        exp_state = load_human_subset(
            dataset_path=args.dataset_path,
            n_samples=args.n_samples,
            seed=selected_seed,
            sample_ids=selected_sample_ids if selected_sample_ids is not None else args.sample_ids,
            participant_id=args.participant_id,
            subset_output_path=subset_output_path,
            output_dir=args.output_dir,
            run_id=base_run_id,
            mode=spec.mode,
            use_reflection=spec.use_reflection,
            max_trials=args.max_trials,
            clarify_quota=args.clarify_quota,
            memo_window=args.memo_window,
            show_gold_to_user=args.show_gold_to_user,
            output_stem=spec.output_stem,
        )
        selected_sample_ids = exp_state.sample_ids
        selected_seed = exp_state.config.seed
        section_states[spec.key] = exp_state

    return HumanAppState(section_states=section_states)


def _unique_base_run_id(output_dir: str | Path, requested_base_run_id: str) -> str:
    base = _safe_run_slug(requested_base_run_id)
    candidate = base
    counter = 2
    while _base_run_id_exists(output_dir, candidate):
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def _base_run_id_exists(output_dir: str | Path, base_run_id: str) -> bool:
    root = Path(output_dir)
    if (root / base_run_id).exists():
        return True
    return any(
        (root / f"{base_run_id}_{spec.run_suffix}").exists()
        for spec in SECTION_SPECS
    )


def _safe_run_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return slug.strip("_") or "anon"


def _section_subset_output_path(path_text: str | None, suffix: str) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R-Clarify human-in-the-loop Gradio app.")
    parser.add_argument("--dataset_path", default="data/processed_data_expanded.json")
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--participant_id", default=None)
    parser.add_argument("--subset_output_path", default=None)
    parser.add_argument("--output_dir", default="outputs/human_runs")
    parser.add_argument("--run_id", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--show_gold_to_user",
        "--show-gold-to-user",
        dest="show_gold_to_user",
        action="store_true",
    )
    group.add_argument(
        "--hide_gold_to_user",
        "--hide-gold-to-user",
        dest="show_gold_to_user",
        action="store_false",
    )
    parser.set_defaults(show_gold_to_user=True)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="Deprecated for the default two-section UI; accepted but ignored.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use_reflection",
        "--use-reflection",
        dest="use_reflection",
        action="store_true",
        help="Deprecated for the default two-section UI; accepted but ignored.",
    )
    group.add_argument(
        "--no_reflection",
        "--no-reflection",
        dest="use_reflection",
        action="store_false",
        help="Deprecated for the default two-section UI; accepted but ignored.",
    )
    parser.set_defaults(use_reflection=True)
    parser.add_argument("--max_trials", "--max-trials", type=int, default=5)
    parser.add_argument("--clarify_quota", "--clarify-quota", type=int, default=2)
    parser.add_argument("--memo_window", "--memo-window", type=int, default=None)
    parser.add_argument("--server_name", default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        app_state = _build_app_state(args)
    except Exception as exc:
        raise SystemExit(f"起動できませんでした: {exc}") from exc

    demo = build_app(app_state)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        js=NEXT_SAMPLE_SOUND_INIT_JS,
    )


if __name__ == "__main__":
    main()
