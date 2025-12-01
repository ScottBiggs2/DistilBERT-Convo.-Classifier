#!/usr/bin/env bash
set -euo pipefail

# run_both.sh
# Starts both labeling scripts in the background using nohup
# Writes per-script stdout/stderr logs and pid files to ./logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# If you use a virtualenv at .venv, activate it automatically
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Start Gemini labeling script
# Optionally auto-confirm the interactive prompt in the Gemini script by
# setting AUTO_CONFIRM_GEMINI=1 in the environment that runs this script.
AUTO_CONFIRM_GEMINI="${AUTO_CONFIRM_GEMINI:-0}"
if [ "$AUTO_CONFIRM_GEMINI" = "1" ] || [ "$AUTO_CONFIRM_GEMINI" = "true" ]; then
  # Pipe a single 'y' followed by newline into the script so it won't block on input()
  printf 'y\n' | nohup python3 "$SCRIPT_DIR/src/get_gemini_flash_labels.py" \
    > "$LOG_DIR/gemini.out" 2> "$LOG_DIR/gemini.err" &
else
  nohup python3 "$SCRIPT_DIR/src/get_gemini_flash_labels.py" \
    > "$LOG_DIR/gemini.out" 2> "$LOG_DIR/gemini.err" &
fi
echo $! > "$LOG_DIR/gemini.pid"
echo "Started gemini labeling (pid $(cat \"$LOG_DIR/gemini.pid\")), logs: $LOG_DIR/gemini.* (AUTO_CONFIRM_GEMINI=$AUTO_CONFIRM_GEMINI)"

# Start GPT-4o-mini unverified logprobs script
nohup python3 "$SCRIPT_DIR/src/get_gpt_4o_mini_logprobs_unverified.py" \
  > "$LOG_DIR/gpt4o_unverified.out" 2> "$LOG_DIR/gpt4o_unverified.err" &
echo $! > "$LOG_DIR/gpt4o_unverified.pid"
echo "Started gpt4o_unverified labeling (pid $(cat \"$LOG_DIR/gpt4o_unverified.pid\")), logs: $LOG_DIR/gpt4o_unverified.*"

echo "\n✅ All processes launched. Use 'tail -f logs/...'' to monitor logs or run './stop_both.sh' to stop."