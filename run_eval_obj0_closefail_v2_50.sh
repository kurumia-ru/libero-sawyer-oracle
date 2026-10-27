#!/usr/bin/env bash
# Eval-only: closefail_v2 merged ckpt, 50 episodes task0.
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data

GPU_EVAL="${GPU_EVAL:-4}"
PORT="${PORT:-8106}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-50}"

MERGED_CKPT="${MERGED_CKPT:-checkpoints/pi05_lora_b_object_sawyer_merged_bs4_10000_scripted_obj0_osc_v8_dagger_closefail_v2}"
ASSETS="${ASSETS:-$DATA/libero_sawyer_scripted_obj0_osc_v8_dagger_closefail_v2}"
TAG=scripted_obj0_osc_v8_dagger_closefail_v2_eval50

LOGDIR=$EVAL/logs_sawyer_oracle
LOG=$LOGDIR/${TAG}.nohup.log
mkdir -p "$LOGDIR"

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export OPENPI_XEMB_ROOT=$XEMB
export LIBERO_DATA_ROOT=$DATA
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}

log() { echo "[$(date -Iseconds)] [obj0-v2-eval50] $*" | tee -a "$LOG"; }

patch_config() {
  local target="$1"
  local cfg="$XEMB/src/openpi/training/config.py"
  log "PATCH b_sawyer repo -> $target"
  python3 - <<PY
from pathlib import Path
import re
cfg = Path("$cfg")
text = cfg.read_text()
text2 = re.sub(r'("b_sawyer"\s*:\s*)"[^"]+"', r'\1"$target"', text)
cfg.write_text(text2)
m = re.search(r'"b_sawyer":\s*"[^"]+"', text2)
print("[patch]", m.group(0) if m else "missing")
PY
}

cd "$XEMB"
[[ -d "$XEMB/$MERGED_CKPT" ]] || { log "FATAL missing $XEMB/$MERGED_CKPT"; exit 1; }
[[ -d "$ASSETS" ]] || { log "FATAL missing assets $ASSETS"; exit 1; }

patch_config "$ASSETS"
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
sleep 1

log "SERVE $MERGED_CKPT port=$PORT gpu=$GPU_EVAL trials=$NUM_TRIALS"
SLOG=$EVAL/serve_${TAG}.log
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
  policy:checkpoint --policy.config pi05_lora_b_object_sawyer \
  --policy.dir "$XEMB/$MERGED_CKPT" > "$SLOG" 2>&1 &
SPID=$!
for _ in $(seq 1 90); do
  grep -q "server listening" "$SLOG" 2>/dev/null && break
  sleep 2
done
grep -E "Loaded norm stats|listening|Error|Traceback" "$SLOG" | tee -a "$LOG" || true
grep -q "server listening" "$SLOG" || { log "FATAL serve failed"; kill $SPID 2>/dev/null || true; exit 1; }

cd "$ROOT/openpi/examples/libero"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL
VDIR=$EVAL/videos_${TAG}
mkdir -p "$VDIR"
log "EVAL start -> $EVAL/eval_${TAG}.log videos=$VDIR"
python "$EVAL/eval_libero_sawyer.py" \
  --args.host localhost --args.port "$PORT" \
  --args.task-suite-name libero_object --args.task-ids "$TASK_ID" \
  --args.num-trials-per-task "$NUM_TRIALS" --args.seed 7 \
  --args.robot-variant sawyer --args.sawyer-home-w30 \
  --args.control-freq 20 \
  --args.video-out-path "$VDIR" \
  > "$EVAL/eval_${TAG}.log" 2>&1 || true

kill "$SPID" 2>/dev/null || true
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
grep -E "SUCCESS_RATE|STAGE_RATES|Task 0 success" "$EVAL/eval_${TAG}.log" | tee -a "$LOG" || true
log "ALL DONE"
echo OBJ0_CLOSEFAIL_V2_EVAL50_DONE
