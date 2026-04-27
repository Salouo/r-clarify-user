from __future__ import annotations

import argparse
import inspect
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


def build_app(initial_state: HumanExperimentState) -> gr.Blocks:
    with gr.Blocks(title="R-Clarify Human Experiment") as demo:
        state = gr.State(initial_state)

        gr.Markdown(
            "## R-Clarify ヒューマン実験\n\n"
            "画面に表示される状況情報と目標ニーズに基づいて、実際のユーザとしてエージェントの質問に答えてください。"
            "自分の状態・好み・困っていることを自然に説明してください。"
            "質問で明確に求められていない限り、エージェントに具体的な行動を直接命令しないでください。"
        )

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
                visible=initial_state.config.show_gold_to_user,
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
            start_btn = gr.Button("開始 / 現在のサンプルを再開始")
            send_btn = gr.Button("回答を送信", variant="primary")
            next_btn = gr.Button("次のサンプル")
            memo_btn = gr.Button(
                "現在のReflection memoを見る",
                visible=_has_current_reflection_memo(initial_state),
            )
            finish_btn = gr.Button("実験を終了")

        with gr.Row():
            action = gr.Textbox(label="実行された行動", lines=2, interactive=False)
            trial_status = gr.Textbox(label="現在の状態", lines=4, interactive=False)

        reflection_memo = gr.Textbox(
            label="現在のReflection memo",
            lines=8,
            interactive=False,
            visible=False,
        )
        status = gr.Textbox(label="メッセージ", lines=3, interactive=False)

        outputs = [
            state,
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
            send_btn,
            next_btn,
            memo_btn,
            finish_btn,
            action,
            trial_status,
            reflection_memo,
            status,
        ]

        def on_start(exp_state: HumanExperimentState):
            try:
                exp_state = start_current_episode(exp_state, restart=True)
                return _render(exp_state, "現在のサンプルを開始しました。")
            except Exception as exc:
                return _render(exp_state, f"開始できませんでした: {exc}")

        def on_submit(exp_state: HumanExperimentState, human_answer: str):
            try:
                exp_state = submit_answer_for_current(exp_state, human_answer)
                return _render(exp_state, "回答を送信しました。")
            except Exception as exc:
                return _render(exp_state, f"回答を送信できませんでした: {exc}")

        def on_next(exp_state: HumanExperimentState):
            if exp_state.current_episode is None:
                return _render(exp_state, "まず現在のサンプルを開始してください。")
            before_pos = exp_state.current_pos
            exp_state = next_sample(exp_state, auto_start=True)
            if exp_state.current_pos == before_pos:
                if before_pos + 1 >= len(exp_state.samples) and (
                    exp_state.current_episode is None
                    or exp_state.current_episode.finished
                ):
                    return _render(exp_state, "すべてのサンプルが完了しました。")
                return _render(exp_state, "現在のサンプルはまだ完了していません。")
            return _render(exp_state, "次のサンプルを開始しました。")

        def on_finish(exp_state: HumanExperimentState):
            if not _all_samples_done(exp_state):
                return _render(exp_state, "まだすべてのサンプルが完了していません。")
            try:
                results_path, episode_paths = export_logs(exp_state)
                msg = (
                    "すべてのサンプルが完了しました。この画面を閉じてください。\n"
                    f"出力JSON: {results_path}\n"
                    f"エピソード数: {len(episode_paths)}"
                )
                return _render(exp_state, msg)
            except Exception as exc:
                return _render(exp_state, f"終了処理中にログを書き出せませんでした: {exc}")

        def on_show_memo(exp_state: HumanExperimentState):
            memo = _format_current_reflection_memo(exp_state)
            return gr.update(value=memo, visible=True)

        demo.load(lambda exp_state: _render(exp_state), inputs=[state], outputs=outputs)
        start_btn.click(on_start, inputs=[state], outputs=outputs)
        send_btn.click(on_submit, inputs=[state, answer], outputs=outputs)
        answer.submit(on_submit, inputs=[state, answer], outputs=outputs)
        next_btn.click(on_next, inputs=[state], outputs=outputs)
        memo_btn.click(on_show_memo, inputs=[state], outputs=[reflection_memo])
        finish_btn.click(on_finish, inputs=[state], outputs=outputs)

    return demo


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
        qa_history = _format_qa_history(episode)
        chat_messages = _chat_messages(episode)
        if episode.last_action_desc:
            action_text = episode.last_action_desc
        if episode.error_message:
            notice = f"{notice}\n{episode.error_message}".strip()

    if not notice:
        notice = (
            "すべてのサンプルが完了しました。「実験を終了」を押してください。"
            if _all_samples_done(exp_state)
            else "準備ができています。"
        )

    answer_update = gr.update(value="", interactive=send_interactive)
    send_update = gr.update(interactive=send_interactive)
    next_variant = "primary" if episode and episode.finished else "secondary"
    next_update = gr.update(variant=next_variant)
    memo_update = gr.update(visible=_has_current_reflection_memo(exp_state))
    finish_update = gr.update(
        interactive=_all_samples_done(exp_state),
        variant="primary" if _all_samples_done(exp_state) else "secondary",
    )
    reflection_memo_update = gr.update(value="", visible=False)

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
        send_update,
        next_update,
        memo_update,
        finish_update,
        action_text,
        trial_status,
        reflection_memo_update,
        notice,
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


def _current_reflection_memos(exp_state: HumanExperimentState) -> list[str]:
    if not exp_state.config.use_reflection:
        return []
    episode = exp_state.current_episode
    if not episode:
        return []

    prior_memos: list[str] = []
    if episode.trial_contexts:
        context = episode.trial_contexts[-1]
        raw_memos = context.get("prior_memos") or []
        prior_memos = [str(memo).strip() for memo in raw_memos if str(memo).strip()]
    if prior_memos:
        return prior_memos

    return [
        str(memo).strip()
        for memo in episode.reflection_memos[-episode.memo_window :]
        if str(memo).strip()
    ]


def _has_current_reflection_memo(exp_state: HumanExperimentState) -> bool:
    return bool(_current_reflection_memos(exp_state))


def _format_current_reflection_memo(exp_state: HumanExperimentState) -> str:
    memos = _current_reflection_memos(exp_state)
    if not memos:
        return "現在表示できるReflection memoはありません。"
    if len(memos) == 1:
        return memos[0]
    return "\n\n".join(f"[{idx}]\n{memo}" for idx, memo in enumerate(memos, start=1))


def _make_chatbot():
    kwargs = {"label": "対話", "height": 260}
    if "type" in inspect.signature(gr.Chatbot).parameters:
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


def _progress_text(exp_state: HumanExperimentState) -> str:
    total = len(exp_state.samples)
    current = min(exp_state.current_pos + 1, total) if total else 0
    completed = len(exp_state.records_by_sample_id)
    reflection = "有効" if exp_state.config.use_reflection else "無効"
    return (
        f"**進捗:** {current} / {total}　"
        f"**完了:** {completed}　"
        f"**Reflection memo:** {reflection}　"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R-Clarify human-in-the-loop Gradio app.")
    parser.add_argument("--dataset_path", default="data/processed_data_expanded.json")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--participant_id", default=None)
    parser.add_argument("--subset_output_path", default=None)
    parser.add_argument("--output_dir", default="outputs/human_runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--show_gold_to_user", action="store_true")
    parser.add_argument("--mode", choices=MODES, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_reflection", "--use-reflection", dest="use_reflection", action="store_true")
    group.add_argument("--no_reflection", "--no-reflection", dest="use_reflection", action="store_false")
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
    mode = args.mode
    if mode is None:
        mode = "r-clarify" if args.use_reflection else "clarify"
    try:
        exp_state = load_human_subset(
            dataset_path=args.dataset_path,
            n_samples=args.n_samples,
            seed=args.seed,
            sample_ids=args.sample_ids,
            participant_id=args.participant_id,
            subset_output_path=args.subset_output_path,
            output_dir=args.output_dir,
            run_id=args.run_id,
            mode=mode,
            use_reflection=args.use_reflection,
            max_trials=args.max_trials,
            clarify_quota=args.clarify_quota,
            memo_window=args.memo_window,
            show_gold_to_user=args.show_gold_to_user,
        )
    except Exception as exc:
        raise SystemExit(f"起動できませんでした: {exc}") from exc

    demo = build_app(exp_state)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
