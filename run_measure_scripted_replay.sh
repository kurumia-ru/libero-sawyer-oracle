#!/usr/bin/env bash# Measure open-loop replay success for all scripted Sawyer episodes.
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
LOG=$EVAL/logs_sawyer_oracle
REPO="${REPO:-$HOME/openpi_data/libero_sawyer_scripted_v1}"
OUT="$EVAL/scripted_replay_all.json"
RUNLOG="$EVAL/measure_scripted_replay.log"

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export LIBERO_DATA_ROOT=$HOME/openpi_data
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL

mkdir -p "$LOG"

log() { echo "[$(date -Iseconds)] [scripted-replay] $*" | tee -a "$LOG/measure_scripted_replay.log"; }

if [[ -f "$OUT" ]] && python3 -c "import json; d=json.load(open('$OUT')); exit(0 if d.get('n_episodes',0)>=500 else 1)" 2>/dev/null; then
  log "SKIP (done): $OUT"
  python3 -c "import json; d=json.load(open('$OUT')); print(f\"SUCCESS {d['success']}/{d['n_episodes']} ({100*d['success_rate']:.1f}%)\")"
  exit 0
fi

log "START all-ep replay repo=$REPO"
cd "$XEMB"
uv run python "$EVAL/measure_scripted_replay.py" \
  --repo "$REPO" \
  --suite object \
  --every 50 \
  --track-drift \
  --out "$OUT" \
  2>&1 | tee "$RUNLOG"

log "DONE -> $OUT"
