from __future__ import annotations

import argparse
import inspect
import json
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
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
    section_started_at: dict[str, float] = field(default_factory=dict)
    section_finished_at: dict[str, float] = field(default_factory=dict)
    experiment_started_at: float | None = None
    experiment_finished_at: float | None = None
    startup_notice: str = ""


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
  window.__rClarifyLastAnswerNeededCue = "";

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

  window.__rClarifyPlayAnswerNeededCue = (cue) => {
    if (!cue || cue === window.__rClarifyLastAnswerNeededCue) {
      return;
    }
    window.__rClarifyLastAnswerNeededCue = cue;

    const ctx = getAudioContext();
    if (!ctx) {
      return;
    }

    const play = () => {
      const start = ctx.currentTime + 0.01;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.14, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.34);
      gain.connect(ctx.destination);

      [659.25, 880].forEach((frequency, index) => {
        const oscillator = ctx.createOscillator();
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(frequency, start + index * 0.1);
        oscillator.connect(gain);
        oscillator.start(start + index * 0.1);
        oscillator.stop(start + index * 0.1 + 0.16);
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


ANSWER_NEEDED_SOUND_CHANGE_JS = """
(cue) => {
  if (window.__rClarifyPlayAnswerNeededCue) {
    window.__rClarifyPlayAnswerNeededCue(cue);
  }
  return [];
}
"""


INFO_PANEL_CSS = """
.info-panel {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 12px 14px;
  background: var(--background-fill-primary);
  min-height: 120px;
}

.info-panel p {
  margin: 0 0 0.35rem;
}

.info-panel p:last-child {
  margin-bottom: 0;
}

.timer-panel {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--background-fill-secondary);
}

.timer-panel p {
  margin: 0;
}
"""

CHECKPOINT_DIRNAME = ".ui_checkpoints"
CHECKPOINT_FILE_NAME = "human_app_state.pkl"


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
            timer_display = gr.Markdown(
                value=_timing_text(app_state),
                elem_classes=["timer-panel"],
            )

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
                with gr.Column():
                    gr.Markdown("### ユーザ情報")
                    user_info = gr.Markdown(
                        value="",
                        show_label=False,
                        container=False,
                        line_breaks=True,
                        min_height=120,
                        elem_classes=["info-panel"],
                    )
                with gr.Column():
                    gr.Markdown("### 環境情報")
                    env_info = gr.Markdown(
                        value="",
                        show_label=False,
                        container=False,
                        line_breaks=True,
                        min_height=120,
                        elem_classes=["info-panel"],
                    )

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
            answer_needed_audio_cue = gr.Textbox(value="", visible=False)

        outputs = [
            state,
            landing_status,
            with_reflection_btn,
            without_reflection_btn,
            progress,
            timer_display,
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
            answer_needed_audio_cue,
        ]
        timer = gr.Timer(1)

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
                section_key = app_state.current_section
                if section_key is not None:
                    _ensure_section_started(app_state, section_key)
                    exp_state.run_start_time = app_state.section_started_at[section_key]
                    _sync_timing_metadata(app_state)
                exp_state = start_current_episode(exp_state, restart=True)
                _set_active_section_state(app_state, exp_state)
                return _render_app(app_state, "現在のサンプルを開始しました。")
            except Exception as exc:
                if exp_state is not None:
                    exp_state.current_episode = None
                    _set_active_section_state(app_state, exp_state)
                return _render_app(
                    app_state,
                    f"開始できませんでした: {exc}\n「開始」を押して現在のサンプルを再開始してください。",
                )

        def on_start_processing(app_state: HumanAppState):
            section_key = app_state.current_section
            exp_state = _active_section_state(app_state)
            if section_key is not None and exp_state is not None:
                _ensure_section_started(app_state, section_key)
                exp_state.run_start_time = app_state.section_started_at[section_key]
                _set_active_section_state(app_state, exp_state)
                _sync_timing_metadata(app_state)
            return (
                app_state,
                _timing_text(app_state),
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
                exp_state.current_episode = None
                _set_active_section_state(app_state, exp_state)
                return _render_app(
                    app_state,
                    f"回答を送信できませんでした: {exc}\n"
                    "「開始」を押して現在のサンプルを再開始してください。",
                )

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
            try:
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
            except Exception as exc:
                exp_state.current_episode = None
                _set_active_section_state(app_state, exp_state)
                return _render_app(
                    app_state,
                    f"次のサンプルを開始できませんでした: {exc}\n"
                    "「開始」を押して現在のサンプルを再開始してください。",
                )

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
                section_key = app_state.current_section
                finished_at = None
                if section_key is not None:
                    finished_at = _mark_section_finished(app_state, section_key)
                    app_state.finalized_sections.add(section_key)
                if _all_sections_finalized(app_state):
                    _mark_experiment_finished(app_state, finished_at)
                _sync_timing_metadata(app_state)
                results_path, episode_paths = export_logs(exp_state)
                if _all_sections_finalized(app_state):
                    _export_finished_section_logs(app_state, skip_section_key=section_key)
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

        def on_load(app_state: HumanAppState):
            restored_state, restore_notice = _load_app_state_checkpoint(app_state)
            notice_parts = []
            if restore_notice:
                notice_parts.append(restore_notice)
            startup_notice = (getattr(restored_state, "startup_notice", "") or "").strip()
            if startup_notice:
                notice_parts.append(startup_notice)
                restored_state.startup_notice = ""
            return _render_app(restored_state, "\n".join(notice_parts).strip())

        demo.load(
            on_load,
            inputs=[state],
            outputs=outputs,
            show_progress_on=[status],
        )
        timer.tick(
            lambda app_state: _timing_text(app_state),
            inputs=[state],
            outputs=[timer_display],
            queue=False,
            show_progress="hidden",
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
        start_busy_outputs = [state, timer_display, *busy_outputs]
        start_btn.click(
            on_start_processing,
            inputs=[state],
            outputs=start_busy_outputs,
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
        answer_needed_audio_cue.change(
            fn=None,
            inputs=[answer_needed_audio_cue],
            outputs=None,
            js=ANSWER_NEEDED_SOUND_CHANGE_JS,
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
    checkpoint_notice = ""
    try:
        _save_app_state_checkpoint(app_state)
    except Exception:
        checkpoint_notice = "進捗の自動保存に失敗しました。"
    if checkpoint_notice:
        notice = f"{notice}\n{checkpoint_notice}".strip()

    has_active_section = app_state.current_section is not None
    if has_active_section:
        exp_state = _active_section_state(app_state) or _first_section_state(app_state)
        _, *experiment_values = _render(exp_state, app_state, notice)
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
        _timing_text(app_state),
        exp_state.config.participant_id or "",
        "",
        "",
        "",
        gr.update(value="", visible=exp_state.config.show_gold_to_user),
        _format_user_info(None),
        _format_env_info(None),
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
        "",
    )


def _render(
    exp_state: HumanExperimentState,
    app_state: HumanAppState,
    notice: str = "",
):
    sample = _current_sample(exp_state)
    episode = exp_state.current_episode
    config = exp_state.config
    progress_text = _progress_text(exp_state)

    sample_id = str(sample.get("index", "")) if sample else ""
    initial_utterance = str(sample.get("utterance", "")) if sample else ""
    hidden_need_value = _format_hidden_need(sample.get("reflective_action")) if sample else ""
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
    answer_needed_audio_cue = _answer_needed_audio_cue(exp_state)

    return (
        exp_state,
        progress_text,
        _timing_text(app_state),
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
        answer_needed_audio_cue,
    )


def _ensure_section_started(app_state: HumanAppState, section_key: str) -> None:
    started_at = time.time()
    if app_state.experiment_started_at is None:
        app_state.experiment_started_at = started_at
    if section_key not in app_state.section_started_at:
        app_state.section_started_at[section_key] = started_at
        app_state.section_finished_at.pop(section_key, None)


def _mark_section_finished(app_state: HumanAppState, section_key: str) -> float:
    if section_key not in app_state.section_started_at:
        _ensure_section_started(app_state, section_key)
    finished_at = app_state.section_finished_at.get(section_key)
    if finished_at is None:
        finished_at = time.time()
        app_state.section_finished_at[section_key] = finished_at
    return finished_at


def _mark_experiment_finished(
    app_state: HumanAppState,
    finished_at: float | None = None,
) -> None:
    if app_state.experiment_started_at is None:
        return
    if app_state.experiment_finished_at is None:
        app_state.experiment_finished_at = finished_at or time.time()


def _sync_timing_metadata(app_state: HumanAppState) -> None:
    section_timings = _section_timings_payload(app_state)
    experiment_total = _duration_seconds(
        app_state.experiment_started_at,
        app_state.experiment_finished_at,
    )
    total_section_duration = _total_section_duration_seconds(app_state)
    experiment_timing = {
        "experiment_started_at": _format_epoch(app_state.experiment_started_at),
        "experiment_finished_at": _format_epoch(app_state.experiment_finished_at),
        "experiment_total_duration_seconds": experiment_total,
        "total_section_duration_seconds": total_section_duration,
        "section_timings": section_timings,
    }
    for spec in SECTION_SPECS:
        exp_state = app_state.section_states.get(spec.key)
        if exp_state is None:
            continue
        section_timing = section_timings[_section_result_id(spec.key)]
        metadata = {
            "section_id": _section_result_id(spec.key),
            "section_key": spec.key,
            "section_title": spec.title,
            "run_started_at": section_timing["started_at"],
            "section_started_at": section_timing["started_at"],
            "section_finished_at": section_timing["finished_at"],
            "section_duration_seconds": section_timing["duration_seconds"],
            "section_timings": section_timings,
            "total_section_duration_seconds": total_section_duration,
            "experiment_started_at": experiment_timing["experiment_started_at"],
            "experiment_finished_at": experiment_timing["experiment_finished_at"],
            "experiment_total_duration_seconds": experiment_total,
            "experiment_timing": experiment_timing,
        }
        if section_timing["finished_at"] is not None:
            metadata.update(
                {
                    "run_finished_at": section_timing["finished_at"],
                    "total_duration_seconds": section_timing["duration_seconds"],
                }
            )
        exp_state.extra_results_metadata = metadata


def _export_finished_section_logs(
    app_state: HumanAppState,
    *,
    skip_section_key: str | None = None,
) -> None:
    for spec in SECTION_SPECS:
        if spec.key == skip_section_key or spec.key not in app_state.finalized_sections:
            continue
        exp_state = app_state.section_states.get(spec.key)
        if exp_state is not None and _all_samples_done(exp_state):
            export_logs(exp_state)


def _timing_text(app_state: HumanAppState) -> str:
    total_elapsed = _experiment_elapsed_seconds(app_state)
    total_text = _format_duration(total_elapsed)
    parts = [f"**経過時間:** 合計 `{total_text}`"]
    for spec in SECTION_SPECS:
        elapsed = _section_elapsed_seconds(app_state, spec.key)
        status = _section_timing_status(app_state, spec.key)
        parts.append(f"{spec.title} `{_format_duration(elapsed)}`（{status}）")
    return "  \n".join(parts)


def _section_timing_status(app_state: HumanAppState, section_key: str) -> str:
    if section_key in app_state.section_finished_at:
        return "完了"
    if section_key in app_state.section_started_at:
        return "進行中"
    return "未開始"


def _section_elapsed_seconds(
    app_state: HumanAppState,
    section_key: str,
    now: float | None = None,
) -> float | None:
    started_at = app_state.section_started_at.get(section_key)
    if started_at is None:
        return None
    finished_at = app_state.section_finished_at.get(section_key)
    return _duration_seconds(started_at, finished_at or now or time.time())


def _experiment_elapsed_seconds(app_state: HumanAppState) -> float | None:
    if app_state.experiment_started_at is None:
        return None
    return _duration_seconds(
        app_state.experiment_started_at,
        app_state.experiment_finished_at or time.time(),
    )


def _section_timings_payload(app_state: HumanAppState) -> dict[str, dict[str, Any]]:
    return {
        _section_result_id(spec.key): _section_timing_payload(app_state, spec)
        for spec in SECTION_SPECS
    }


def _section_timing_payload(
    app_state: HumanAppState,
    spec: SectionSpec,
) -> dict[str, Any]:
    started_at = app_state.section_started_at.get(spec.key)
    finished_at = app_state.section_finished_at.get(spec.key)
    return {
        "section_id": _section_result_id(spec.key),
        "section_key": spec.key,
        "section_title": spec.title,
        "started_at": _format_epoch(started_at),
        "finished_at": _format_epoch(finished_at),
        "duration_seconds": _duration_seconds(started_at, finished_at),
    }


def _total_section_duration_seconds(app_state: HumanAppState) -> float | None:
    durations = [
        _duration_seconds(
            app_state.section_started_at.get(spec.key),
            app_state.section_finished_at.get(spec.key),
        )
        for spec in SECTION_SPECS
    ]
    durations = [duration for duration in durations if duration is not None]
    if not durations:
        return None
    return round(sum(durations), 3)


def _section_result_id(section_key: str) -> str:
    return f"section{_section_number(section_key)}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "未開始"
    whole_seconds = int(max(0, seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_epoch(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _duration_seconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(max(0.0, end - start), 3)


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


def _answer_needed_audio_cue(exp_state: HumanExperimentState) -> str:
    episode = exp_state.current_episode
    if not episode or episode.finished or not episode.current_question:
        return ""
    question_key = abs(hash(episode.current_question))
    return (
        f"{exp_state.config.run_id}:{exp_state.current_pos}:"
        f"{episode.sample_id}:{episode.current_trial_index}:{question_key}"
    )


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
            f"**位置:** {_format_value(ann.get('position'))}",
            f"**手にしている物:** {_format_value(ann.get('has'))}",
            f"**近くにある物:** {_format_value(ann.get('near_items'))}",
        ]
    )


def _format_env_info(sample: dict[str, Any] | None) -> str:
    if not sample:
        return ""
    ann = sample.get("annotations") or {}
    items = "\n".join(
        [
            f"**ソファ前テーブルの物品:** {_format_value(ann.get('sofa_front_table_items'))}",
            f"**キッチン前テーブルの物品:** {_format_value(ann.get('kitchen_front_table_items'))}",
            f"**キッチンの物品:** {_format_value(ann.get('kitchen_items'))}",
        ]
    )
    note = "表示されている物品は現在見えている物だけです。リストにない物でも、別の場所から持ってくることがあります。"
    return f"{items}\n\n{note}"


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


def _format_hidden_need(value) -> str:
    text = "" if value is None else str(value).strip()
    return re.sub(r"^\[\d+\]\s*", "", text)


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


def _checkpoint_path(app_state: HumanAppState) -> Path:
    exp_state = _first_section_state(app_state)
    return (
        Path(exp_state.config.output_dir)
        / CHECKPOINT_DIRNAME
        / _safe_run_slug(exp_state.config.run_id)
        / CHECKPOINT_FILE_NAME
    )


def _save_app_state_checkpoint(app_state: HumanAppState) -> None:
    path = _checkpoint_path(app_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "app_state": app_state,
    }
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _load_app_state_checkpoint(
    fallback_state: HumanAppState,
) -> tuple[HumanAppState, str]:
    path = _checkpoint_path(fallback_state)
    if not path.exists():
        return fallback_state, ""
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        restored = payload.get("app_state")
        if isinstance(restored, HumanAppState):
            if not hasattr(restored, "startup_notice"):
                restored.startup_notice = ""
            return restored, "前回の進捗を復元しました。"
    except Exception:
        return fallback_state, "前回の進捗を復元できませんでした。新しい状態で開始します。"
    return fallback_state, ""


def _resume_results_path_for_section(
    args: argparse.Namespace,
    section_key: str,
) -> str | None:
    if section_key == "with_reflection":
        return args.resume_reflection_json
    if section_key == "without_reflection":
        return args.resume_without_reflection_json
    return None


def _apply_resume_from_results(
    exp_state: HumanExperimentState,
    results_path: str | Path,
) -> tuple[int, int]:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Invalid resume file (missing list 'results'): {path}")

    sample_ids_set = set(exp_state.sample_ids)
    sample_pos_by_id = {sid: idx + 1 for idx, sid in enumerate(exp_state.sample_ids)}
    restored_records: dict[int, dict[str, Any]] = {}
    ignored = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        sample_index = item.get("sample_index")
        try:
            sid = int(sample_index)
        except (TypeError, ValueError):
            continue
        if sid in sample_ids_set:
            normalized = dict(item)
            normalized["sample_position"] = sample_pos_by_id[sid]
            normalized["participant_id"] = exp_state.config.participant_id
            normalized["run_id"] = exp_state.config.run_id
            normalized["mode"] = exp_state.config.mode
            normalized["use_reflection"] = exp_state.config.use_reflection
            restored_records[sid] = normalized
        else:
            ignored += 1

    exp_state.records_by_sample_id = restored_records
    exp_state.current_episode = None
    exp_state.finished_episodes = {}

    done_ids = set(restored_records.keys())
    next_pos = None
    for idx, sid in enumerate(exp_state.sample_ids):
        if sid not in done_ids:
            next_pos = idx
            break
    if next_pos is None:
        exp_state.current_pos = max(len(exp_state.samples) - 1, 0)
    else:
        exp_state.current_pos = next_pos

    return len(restored_records), ignored


def _read_sample_ids_from_results(results_path: str | Path) -> list[int]:
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list):
        raise ValueError(f"Invalid resume file (missing list 'sample_ids'): {path}")
    return [int(x) for x in sample_ids]


def _build_app_state(args: argparse.Namespace) -> HumanAppState:
    requested_base_run_id = args.run_id or args.participant_id or "anon"
    base_run_id = _unique_base_run_id(args.output_dir, requested_base_run_id)
    section_states: dict[str, HumanExperimentState] = {}
    resume_notes: list[str] = []
    bootstrap_sample_ids: str | list[int] | None = args.sample_ids
    if bootstrap_sample_ids is None and args.resume_reflection_json:
        bootstrap_sample_ids = _read_sample_ids_from_results(args.resume_reflection_json)
        resume_notes.append("Section 1 の既存結果から sample_ids を復元しました。")
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
            sample_ids=selected_sample_ids if selected_sample_ids is not None else bootstrap_sample_ids,
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

        resume_path = _resume_results_path_for_section(args, spec.key)
        if resume_path:
            restored_count, ignored_count = _apply_resume_from_results(exp_state, resume_path)
            remaining = len(exp_state.samples) - restored_count
            note = (
                f"{spec.title}: {restored_count} 件を復元、残り {remaining} 件。"
            )
            if ignored_count:
                note += f"（対象外サンプル {ignored_count} 件をスキップ）"
            resume_notes.append(note)

        section_states[spec.key] = exp_state

    startup_notice = "\n".join(resume_notes).strip()
    return HumanAppState(section_states=section_states, startup_notice=startup_notice)


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
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--participant_id", default="chen")
    parser.add_argument("--subset_output_path", default=None)
    parser.add_argument("--output_dir", default="outputs/human_runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument(
        "--resume_reflection_json",
        default=None,
        help="Path to an existing reflection.json to resume Section 1 by skipping completed samples.",
    )
    parser.add_argument(
        "--resume_without_reflection_json",
        default=None,
        help="Path to an existing without_reflection.json to resume Section 2 by skipping completed samples.",
    )
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
    parser.add_argument("--show_error", action="store_true")
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
        show_error=args.show_error,
        css=INFO_PANEL_CSS,
        js=NEXT_SAMPLE_SOUND_INIT_JS,
    )


if __name__ == "__main__":
    main()
