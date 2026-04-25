set -euo pipefail

# Set defaults here or override via CLI args.
max_trials=5
memo_window=3      # only used for r-clarify_reflexion unless overridden
clarify_quota=2    # Default clarify count: 2
start_idx="0"
end_idx="10"
mode_list="r-clarify"

# modes setting (clarify, r-clarify, r-clarify_reflexion, r-clarify_non_cot, reflect, non_thinking_clarify, cot, cot_reflect, direct, reflect_action_only)

while getopts ":t:w:c:s:e:m:h" opt; do
  case "$opt" in
    t) max_trials="$OPTARG" ;;
    w) memo_window="$OPTARG" ;;
    c) clarify_quota="$OPTARG" ;;
    s) start_idx="$OPTARG" ;;
    e) end_idx="$OPTARG" ;;
    m) mode_list="$OPTARG" ;;
    h)
      echo "Usage: $0 [-t <max_trials>] [-w <memo_window_for_r-clarify_reflexion>] [-c <clarify_quota>] [-s <start_idx>] [-e <end_idx>] [-m <mode_or_csv>]"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

runner_cmd=(uv run python)

# modes setting (clarify, r-clarify, r-clarify_reflexion, r-clarify_non_cot, reflect, non_thinking_clarify, cot, cot_reflect, direct, reflect_action_only)
IFS=',' read -r -a modes <<< "$mode_list"

for mode in "${modes[@]}"; do
  cmd=("${runner_cmd[@]}" -m src.run --mode "$mode" --max-trials "$max_trials")
  cmd+=(--clarify-quota "$clarify_quota")
  if [[ -n "$start_idx" ]]; then
    cmd+=(--start "$start_idx")
  fi
  if [[ -n "$end_idx" ]]; then
    cmd+=(--end "$end_idx")
  fi

  # memo_window setting
  if [[ "$mode" == "r-clarify_reflexion" ]]; then
    cmd+=(--memo-window "$memo_window")
  else
    cmd+=(--memo-window 1)
  fi

  echo "Command: ${cmd[*]}"
  "${cmd[@]}"
done
