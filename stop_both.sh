#!/usr/bin/env bash
set -euo pipefail

# stop_both.sh
# Stops processes started by run_both.sh by reading pid files in ./logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

PIDFILES=("$LOG_DIR/gemini.pid" "$LOG_DIR/gpt4o_unverified.pid")

for pidfile in "${PIDFILES[@]}"; do
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping pid $pid (from $pidfile)"
      kill "$pid" || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        echo "PID $pid still alive; sending SIGKILL"
        kill -9 "$pid" || true
      fi
    else
      echo "PID $pid (from $pidfile) is not running"
    fi
    rm -f "$pidfile"
  else
    echo "No pidfile: $pidfile"
  fi
done

echo "✅ stop_both.sh complete."