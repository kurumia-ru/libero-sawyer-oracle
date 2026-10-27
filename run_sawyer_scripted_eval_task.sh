#!/usr/bin/env bash
# Merge LoRA + eval on one object task.
# Usage: TASK_ID=1 bash run_sawyer_scripted_eval_task.sh
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

TASK_ID="${TASK_ID:?set TASK_ID}"
ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
CLIENT=$ROOT/openpi/examples/libero
LOG=$EVAL/logs_sawyer_oracle
STEP="${STEP:-30000}"
BS="${BS:-4}"
TAG="bs${BS}_${STEP}_scripted_obj${TASK_ID}"
EXP="sawyer_obj_${TAG}"
CONFIG=pi05_lora_b_object_sawyer
LORA_REL="checkpoints/pi05_lora_b_object_sawyer/${EXP}/${STEP}"
MERGED_REL="checkpoints/pi05_lora_b_object_sawyer_merged_${TAG}"
CKPT="$XEMB/$LORA_REL/model.safetensors"
SEED="${SEED:-7}"
PORT="${PORT:-8007}"
GPU_EVAL="${GPU_EVAL:-6}"
NUM_TRIALS="${NUM_TRIALS:-10}"
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export LIBERO_DATA_ROOT=$HOME/openpi_data

mkdir -p "$LOG"
SUMMARY="$LOG/sawyer_scripted_obj${TASK_ID}_eval_summary.log"
EVAL_LOG="$LOG/sawyer_scripted_eval_obj${TASK_ID}.log"
touch "$SUMMARY"

log() { echo "[$(date -Iseconds)] [eval-obj${TASK_ID}] $*" | tee -a "$EVAL_LOG" "$SUMMARY"; }

wait_for_train_ckpt() {
  log "WAIT ckpt $CKPT"
  for _ in $(seq 1 7200); do
    [[ -f "$CKPT" ]] && { log "FOUND ckpt"; return 0; }
    if ! pgrep -f "train_pytorch_lora.py pi05_lora_b_object_sawyer.*${EXP}" >/dev/null 2>&1; then
      [[ -f "$CKPT" ]] && { log "FOUND ckpt (train exited)"; return 0; }
      log "FATAL train died without ckpt"
      exit 1
    fi
    sleep 30
  done
  log "FATAL timeout waiting for ckpt"
  exit 1
}

merge_ckpt() {
  if [[ -f "$XEMB/$MERGED_REL/model.safetensors" ]]; then
    log "SKIP merge (exists): $MERGED_REL"
    return 0
  fi
  log "MERGE $LORA_REL -> $MERGED_REL gpu=$GPU_EVAL"
  cd "$XEMB"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  uv run scripts/merge_lora_checkpoint.py \
    --config-name "$CONFIG" \
    --lora-checkpoint-dir "$LORA_REL" \
    --output-dir "$MERGED_REL" \
    2>&1 | tee -a "$LOG/merge_sawyer_scripted_obj${TASK_ID}_${TAG}.log"
}

run_eval() {
  local tag="bo_obj${TASK_ID}_scripted"
  local elog="$EVAL/eval_sawyer_scripted_${tag}.log"
  local vdir="$EVAL/videos_sawyer_scripted_obj${TASK_ID}"
  if [[ -f "$elog" ]] && grep -q "SUCCESS_RATE" "$elog"; then
    log "SKIP eval (done)"
    grep "SUCCESS_RATE" "$elog" | tee -a "$SUMMARY"
    return 0
  fi
  while pgrep -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" >/dev/null 2>&1; do
    sleep 10
  done
  mkdir -p "$vdir"
  log "EVAL task_id=$TASK_ID trials=$NUM_TRIALS gpu=$GPU_EVAL"
  pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
  sleep 2
  cd "$XEMB"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
    policy:checkpoint --policy.config "$CONFIG" \
    --policy.dir "$XEMB/$MERGED_REL" > "$EVAL/serve_sawyer_scripted_obj${TASK_ID}.log" 2>&1 &
  local spid=$!
  for _ in $(seq 1 90); do
    grep -q "server listening" "$EVAL/serve_sawyer_scripted_obj${TASK_ID}.log" 2>/dev/null && break
    sleep 2
  done
  grep -q "server listening" "$EVAL/serve_sawyer_scripted_obj${TASK_ID}.log" || {
    log "FATAL serve failed"; tail -20 "$EVAL/serve_sawyer_scripted_obj${TASK_ID}.log"; exit 1
  }
  cd "$CLIENT" && source .venv/bin/activate
  export PYTHONPATH="$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export LIBERO_CONFIG_PATH="$EVAL/libero_config"
  python "$EVAL/eval_libero_sawyer.py" \
    --args.host localhost --args.port "$PORT" \
    --args.task-suite-name libero_object \
    --args.task-ids "$TASK_ID" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.seed "$SEED" \
    --args.robot-variant sawyer \
    --args.sawyer-home-w30 \
    --args.video-out-path "$vdir" > "$elog" 2>&1
  kill "$spid" 2>/dev/null || true
  grep "SUCCESS_RATE" "$elog" | tee -a "$SUMMARY" "$EVAL_LOG"
  grep "Task ${TASK_ID} success" "$elog" | tee -a "$SUMMARY" || true
}

main() {
  log "START eval tag=$TAG"
  wait_for_train_ckpt
  merge_ckpt
  run_eval
  log "ALL DONE eval obj${TASK_ID}"
}

main "$@"
