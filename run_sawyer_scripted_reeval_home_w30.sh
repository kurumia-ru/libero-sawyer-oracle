#!/usr/bin/env bash
# Re-eval Sawyer scripted BO with home_w30 setup (matches training data export).
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
CLIENT=$ROOT/openpi/examples/libero
LOG=$EVAL/logs_sawyer_oracle
STEP="${STEP:-30000}"
BS="${BS:-4}"
TAG="bs${BS}_${STEP}_scripted_v1"
CONFIG=pi05_lora_b_object_sawyer
MERGED_REL="checkpoints/pi05_lora_b_object_sawyer_merged_${TAG}"
SEED="${SEED:-7}"
PORT="${PORT:-8008}"
GPU_EVAL="${GPU_EVAL:-0}"
EVAL_TAG="bo_object_scripted_home_w30"

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export LIBERO_DATA_ROOT=$HOME/openpi_data

mkdir -p "$LOG"
SUMMARY="$LOG/sawyer_scripted_home_w30_summary.log"

log() { echo "[$(date -Iseconds)] [scripted-home-w30] $*" | tee -a "$LOG/sawyer_scripted_home_w30.log" "$SUMMARY"; }

wait_port_free() {
  while pgrep -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" >/dev/null 2>&1 \
    || pgrep -f "eval_libero_sawyer.py --args.host localhost --args.port ${PORT}" >/dev/null 2>&1; do
    log "WAIT port $PORT"
    sleep 20
  done
}

main() {
  [[ -f "$XEMB/$MERGED_REL/model.safetensors" ]] || {
    log "FATAL missing merged ckpt: $XEMB/$MERGED_REL"
    exit 1
  }

  local elog="$EVAL/eval_sawyer_scripted_${EVAL_TAG}.log"
  local vdir="$EVAL/videos_sawyer_scripted_${EVAL_TAG}"

  if [[ -f "$elog" ]] && grep -q "SUCCESS_RATE" "$elog"; then
    log "SKIP eval (done): $EVAL_TAG"
    grep "SUCCESS_RATE" "$elog" | tee -a "$SUMMARY"
    exit 0
  fi

  wait_port_free
  mkdir -p "$vdir"
  log "START eval tag=$EVAL_TAG gpu=$GPU_EVAL port=$PORT"

  pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
  sleep 2

  cd "$XEMB"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
    policy:checkpoint --policy.config "$CONFIG" \
    --policy.dir "$XEMB/$MERGED_REL" > "$EVAL/serve_sawyer_scripted_${EVAL_TAG}.log" 2>&1 &
  local spid=$!
  for _ in $(seq 1 90); do
    grep -q "server listening" "$EVAL/serve_sawyer_scripted_${EVAL_TAG}.log" 2>/dev/null && break
    sleep 2
  done
  if ! grep -q "server listening" "$EVAL/serve_sawyer_scripted_${EVAL_TAG}.log" 2>/dev/null; then
    log "FATAL serve failed"
    tail -30 "$EVAL/serve_sawyer_scripted_${EVAL_TAG}.log" | tee -a "$LOG/sawyer_scripted_home_w30.log"
    kill "$spid" 2>/dev/null || true
    exit 1
  fi

  cd "$CLIENT" && source .venv/bin/activate
  export PYTHONPATH="$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export LIBERO_CONFIG_PATH="$EVAL/libero_config"
  python "$EVAL/eval_libero_sawyer.py" \
    --args.host localhost --args.port "$PORT" \
    --args.task-suite-name libero_object \
    --args.num-trials-per-task 10 \
    --args.seed "$SEED" \
    --args.robot-variant sawyer \
    --args.sawyer-home-w30 \
    --args.video-out-path "$vdir" > "$elog" 2>&1

  kill "$spid" 2>/dev/null || true
  grep "SUCCESS_RATE" "$elog" | tee -a "$SUMMARY" "$LOG/sawyer_scripted_home_w30.log"
  log "DONE eval -> $elog videos -> $vdir"
}

main "$@"
