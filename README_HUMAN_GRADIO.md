# Human-in-the-loop R-Clarify Gradio 実験说明

这个 human mode 是当前模拟实验的最小侵入式扩展：保留 agent、Clarify/Execute、trial、reflection memo、success/failure 判断、episode/output 记录格式，只把原来的 `user_sim` 回答替换成真实实验者在 Gradio 页面中输入的回答。

运行时面向实验者显示的 UI 文本均为日语。README 本身使用中文说明；如果这里引用 UI 文本示例，示例也保持日语。

## 与 simulated mode 的关系

- simulated mode：`agent -> clarify -> user_sim -> agent -> execute`
- human mode：`agent -> clarify -> human input via Gradio -> agent -> execute`

human answer 会以与 simulated user answer 相同的位置写入 graph state：

- 追加为 `HumanMessage(content=<human answer>)`
- 写入对应 `AgentStep.user_reply`
- JSON 日志中标记 `user_source: "human"`

原自动模拟入口 `python -m src.run ...` 保持可用。新加的 simulated 日志字段是向后兼容扩展，不会删除或重命名原字段。

## 启动方式

默认每位实验者抽 30 个样本，并使用 reflection memo：

```bash
uv run python app_gradio_human.py \
  --dataset_path data/processed_data_expanded.json \
  --n_samples 30 \
  --participant_id P001
```

指定 seed 以复现本次抽样：

```bash
uv run python app_gradio_human.py \
  --dataset_path data/processed_data_expanded.json \
  --n_samples 30 \
  --seed 42 \
  --participant_id P001
```

严格指定样本：

```bash
uv run python app_gradio_human.py \
  --sample_ids 1,5,9 \
  --participant_id P001
```

默认会显示 gold action / hidden need 给真实用户：

```bash
uv run python app_gradio_human.py \
  --n_samples 30 \
  --participant_id P001
```

如需在特殊对照中隐藏该信息：

```bash
uv run python app_gradio_human.py \
  --n_samples 30 \
  --participant_id P001 \
  --hide_gold_to_user
```

UI 中对应区域标题为：

```text
目標ニーズ（エージェントには表示されません）
```

这个设置的目的不是把答案泄露给 agent，而是让真实实验者在与 simulated user 尽量一致的信息条件下扮演目标用户。该信息不会被加入 agent context；它只用于 human role-play、success/failure 判断与日志审计。

## Reflection memo 开关

默认开启 reflection memo，用于 R-Clarify：

```bash
uv run python app_gradio_human.py \
  --participant_id P001 \
  --use_reflection
```

关闭 reflection memo，用于对照实验：

```bash
uv run python app_gradio_human.py \
  --participant_id P001 \
  --no_reflection
```

如果没有显式传入 `--mode`：

- `--use_reflection` 默认使用 `r-clarify`
- `--no_reflection` 默认使用 `clarify`

如果你显式传入 `--mode r-clarify --no_reflection`，程序会尊重该 mode，但不会在失败后生成或传递 reflection memo。更推荐的 no-reflection 对照是直接使用默认的 `--no_reflection`，即 `clarify` mode。

## 抽样设计

每个 participant / run 都从完整 `data/processed_data_expanded.json` 重新随机抽样：

- 跨 run 有放回：不同实验者允许抽到相同样本。
- 单个 run 内默认不重复：使用类似 `random.sample(all_sample_ids, 30)` 的逻辑。
- 不维护全局 used set。
- 不排除上一次实验者抽到过的样本。
- 指定 `--sample_ids` 时严格使用给定样本，不随机抽样。
- 未指定 `--seed` 时自动生成 seed。

抽样信息会写入最终 results JSON 顶层，其中至少包含：

- `run_id`
- `participant_id`
- `timestamp`
- `dataset_path`
- `n_samples`
- `seed`
- `sample_ids`
- `sampling_scope: "full_dataset_each_run"`
- `within_run_replacement: false`
- `across_run_replacement: true`

默认不再生成单独的 `sample_subset.json`。如果需要额外保存抽样 manifest，可以显式指定 `--subset_output_path`。

## 输出位置

human run 的输出默认位于：

```text
outputs/human_runs/<run_id>/
```

主要文件：

```text
outputs/human_runs/<run_id>/results/human_<mode>_n<N>_<participant_id>_<run_id>.json
outputs/human_runs/<run_id>/episodes/sample<sample_id>_human_<run_id>.txt
```

results JSON 保留 simulated experiment 的核心字段：

- `n_processed`
- `clarify_quota`
- `memo_window`
- `results`
- `sample_index`
- `max_trials`
- `num_trials`
- `correct`
- `steps_total`
- `steps_per_trial`
- `steps_detail_per_trial`
- `clarify_turns_per_trial`
- `gold_action`

human mode 追加向后兼容字段：

- `user_source: "human"`
- `participant_id`
- `run_id`
- `mode`
- `use_reflection`
- `sampling_scope`
- `response_latencies`
- `human_visible_context`

这些新增字段不会影响只读取旧字段的 evaluation 脚本。

## Gradio 页面行为

页面会显示：

- 进度：第几个样本 / 总样本数
- 参加者ID、実行ID、サンプルID
- 最初の発話
- ユーザ情報
- 環境情報
- 目标ニーズ（默认显示，可用 `--hide_gold_to_user` 隐藏）
- エージェントの明確化質問
- 現在の試行内の質問・回答履歴
- 対話
- 回答输入框
- 実行された行動
- 当前 trial / clarify count / success / finished 状态

顶部说明文字为：

```text
画面に表示される状況情報と目標ニーズに基づいて、実際のユーザとしてエージェントの質問に答えてください。自分の状態・好み・困っていることを自然に説明してください。質問で明確に求められていない限り、エージェントに具体的な行動を直接命令しないでください。
```

如果当前 sample 已完成，继续点击发送回答会显示日语提示，并要求进入下一个 sample。

## 恢复与复现实验

当前实现保存本次 run 的 seed 和 sample ids。要复现实验样本顺序：

```bash
uv run python app_gradio_human.py \
  --run_id replay_P001 \
  --seed <results JSON中的seed> \
  --n_samples <results JSON中的n_samples> \
  --participant_id P001
```

或者直接使用已保存的 sample ids：

```bash
uv run python app_gradio_human.py \
  --sample_ids 1,5,9 \
  --participant_id P001
```

注意：这会复现样本集合与顺序，但不会自动恢复 Gradio 会话中的中途输入。当前实现会在每完成一个 sample 后自动落盘。

## 常见错误

- API key 缺失：确认 `.env` 中存在 `OPENAI_API_KEY=...`。
- dataset path 错误：确认 `--dataset_path` 指向包含 `samples` 列表的 JSON。
- sample_ids 错误：确认每个 ID 都存在于 expanded dataset 的 `index` 字段中。
- output_dir 无法写入：确认目录权限，或使用 `--output_dir` 指向可写目录。
- no-reflection 对照：使用 `--no_reflection`，不要手动维护或删除 memo 文件。
