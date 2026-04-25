# Clarification for Reflective Action

A research-style benchmark pipeline for **ambiguous household utterance understanding**.  
The agent must choose exactly one “reflective action” from a fixed 40-action space, optionally ask clarification questions, and improve across trials using reflection memos.

## What This Project Does

- Models the task as a LangGraph state machine:
  - `agent` decides `clarify` or `execute`
  - `clarify` asks one question
  - `user_sim` answers as a simulated user
  - `execute` commits to one action ID (`1..40`)
- Supports multiple decision styles:
  - clarify-first, reflection-augmented, CoT-like, direct, and no-thought variants
- Runs multi-trial inference per sample, with optional reflection memo carryover.
- Logs rich artifacts for analysis:
  - per-trial steps
  - clarification turns and word counts
  - token usage (agent + reflection)
  - full per-sample episode transcripts

## Task and Data

The task is to infer a latent user need from:

- user utterance
- user position and held items
- nearby items and item placement in the room

Then execute one action from `data/options.txt` (40 candidate actions).

### Data Files

- `data/raw_data.json`: original paired data with `reflective_actions` (possibly multiple labels per utterance)
- `data/processed_data.json`: normalized structure with `annotations` + multi-label `reflective_actions`
- `data/processed_data_expanded.json`: expanded to single-label samples (`reflective_action`), used by `src.run`
- `data/processed_data_expanded_en.json`: same size/structure variant

Current dataset sizes in this repo:

- base samples: `355`
- expanded single-label samples: `545`

## Modes

Supported `--mode` values in `src/run.py`:

- `clarify`: clarify-enabled, no reflection
- `r-clarify`: clarify + reflection memo (window default `1`)
- `r-clarify_reflexion`: clarify + multi-memo reflection (window default `3`)
- `r-clarify_non_cot`: clarify + reflection, but `thought` output disabled
- `reflect`: reflection-enabled execute-only (no clarify)
- `reflect_action_only`: reflection stores wrong actions only
- `non_thinking_clarify`: clarify-enabled, `thought` forced to `null`
- `cot`: single-step execute with thought (no clarify)
- `cot_reflect`: `cot` + reflection memo
- `direct`: single-step execute, no thought

## Project Structure

```text
src/
  run.py            # dataset runner, resume logic, result writing
  reflection.py     # multi-trial loop + memo generation + episode export
  graph.py          # LangGraph nodes and transitions
  prompts.py        # agent/user-sim/reflection prompts
  state.py          # action table + Pydantic state models
  llm_factory.py    # GPT model selection
  gpt_llm.py        # OpenAI chat wrapper
  token_usage.py    # context-local token collector
  utils.py          # prompt formatting + step extraction
  eval.py           # evaluation metrics on results JSON
```

## Environment Setup

### 1) Python

- Python `>=3.11` (as defined in `pyproject.toml`)

### 2) Install Dependencies

```bash
uv sync
```

### 3) API Key

Create `.env` in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

Important: this project now uses GPT/OpenAI only, so OpenAI credentials are required.

### 4) Choose GPT Model

Edit `src/llm_factory.py`:

- adjust `GPT_MODEL_NAME` as needed

## Running Experiments

### Single Run

```bash
uv run python -m src.run \
  --mode r-clarify_non_cot \
  --max-trials 5 \
  --clarify-quota 2 \
  --memo-window 1 \
  --start 0 \
  --end 545
```

Key arguments:

- `--mode`: one of the modes above
- `--max-trials` (required): max trials per sample
- `--clarify-quota`: max clarification turns per trial (`None` => unlimited)
- `--memo-window`: number of latest memos passed into next trial
- `--start`, `--end`: sample index range `[start, end)`

### Batch Script

Use `run_all_modes.sh` to run one or multiple modes:

```bash
bash run_all_modes.sh -g 0,1,2,3 -t 5 -c 2 -s 0 -e 546 -m r-clarify_non_cot
```

`-m` accepts CSV, e.g. `-m clarify,r-clarify,reflect`.

## Outputs

Outputs are grouped by model directory name:

- Results JSON:
  - `outputs/<model_dir>/results/<mode>.json`
  - or sharded: `outputs/<model_dir>/results/<mode>_<start>_<end>.json`
- Episode transcripts:
  - `outputs/<model_dir>/episodes/<mode>/sample<index>.txt`

### Result Record Fields (per sample)

Each entry in `results` includes:

- `sample_index`
- `num_trials`
- `correct`
- `steps_total`
- `steps_per_trial`
- `steps_detail_per_trial` (serialized step logs, token usage included)
- `clarify_turns_per_trial`
- `gold_action`

## Evaluation

`src/eval.py` provides:

- `Pass@k`-style accuracy by allowed trial count
- average steps for successful trials
- `DPass@k(alpha)` (step-discounted success score)
- token usage breakdown before success (`reflect` / `execute` / `clarify`)

Run:

```bash
uv run python -m src.eval
```

Before running, update `metrics_path` inside `src/eval.py` to your target result file.

## Utility Scripts

- `merge_results.py`: merge multiple sharded result files into one
- `process_dataset.py`: expand `reflective_actions` list into single-label samples
- `test.py`: compute clarify usage ratio among successful trials

Note: some script paths are hardcoded; adjust paths before use.

## Current Limitations

- Prompts and action descriptions are primarily Japanese.
- Thread pool in `src/run.py` is currently `max_workers=1` (sequential sample processing).
- Agent backend uses the GPT model configured in `src/llm_factory.py`.
- User simulator model is fixed to OpenAI `gpt-5.2` in current code.
