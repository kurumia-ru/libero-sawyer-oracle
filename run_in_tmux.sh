#!/usr/bin/env bash
# Generic tmux launcher for long openpi_eval jobs.
# Usage: ./run_in_tmux.sh SESSION_NAME "command to run"
set -euo pipefail

SESSION="${1:?session name}"
shift
CMD="${*:?command}"
WORKDIR="${OPENPI_XEMB_ROOT:-$HOME/openpi_ws/openpi-xemb}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session exists: $SESSION (tmux attach -t $SESSION)"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$WORKDIR" "$CMD"
echo "started: $SESSION"
echo "attach: tmux attach -t $SESSION"
