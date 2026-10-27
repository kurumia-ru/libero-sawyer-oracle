#!/usr/bin/env bash
# Control experiment: same LoRA targets as the 28% run, but init from pi05_libero.
#
# The 28% baseline was: pi05_base init + Full LoRA (leaf-name targets ->
# vision 81 / language 126 / action 126) on the Sawyer dagger dataset.
# This run changes exactly one variable: the initialization.
#
# Usage:
#   ONLY=train bash run_pi05_libero_sawyer_fulllora.sh
#   ONLY=eval  bash run_pi05_libero_sawyer_fulllora.sh
#   ONLY=all   bash run_pi05_libero_sawyer_fulllora.sh
#
#   INIT=base  bash run_pi05_libero_sawyer_fulllora.sh   # reproduce the 28% control
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
CLIENT=$ROOT/openpi/examples/libero
DATA=$HOME/openpi_data
CKPT=$XEMB/checkpoints
LOGDIR=$EVAL/logs_sawyer_oracle

STEP="${STEP:-10000}"
BS="${BS:-4}"
NPROC="${NPROC:-2}"
GPU_TRAIN="${GPU_TRAIN:-0,1}"
GPU_EVAL="${GPU_EVAL:-0}"
NUM_TRIALS="${NUM_TRIALS:-50}"
CONTROL_FREQ="${CONTROL_FREQ:-20}"
SEED="${SEED:-7}"
PORT="${PORT:-8862}"
ONLY="${ONLY:-all}"  # train|eval|all
TASK_ID="${TASK_ID:-0}"
CONFIG="${CONFIG:-pi05_lora_b_object_sawyer}"

# INIT=libero (default) reuses the 28% recipe with a different starting point.
INIT="${INIT:-libero}"
case "$INIT" in
  libero) INIT_CKPT="${INIT_CKPT:-$ROOT/openpi/checkpoints/pi05_libero_pytorch}" ;;
  base)   INIT_CKPT="${INIT_CKPT:-$ROOT/openpi/checkpoints/pi05_base_pytorch}" ;;
  *)      echo "INIT must be libero|base, got $INIT" >&2; exit 1 ;;
esac

# Match the 28% run's dataset (Sawyer Task0 scripted + closefail dagger).
REPO="${REPO:-$DATA/libero_sawyer_scripted_obj${TASK_ID}_osc_v8_dagger_closefail_v2}"
DATA_TAG="${DATA_TAG:-$(basename "$REPO" | sed -E 's/^libero_sawyer_//')}"

EXP="${EXP:-${INIT}_init_sawyer_obj${TASK_ID}_fulllora_s${STEP}_${DATA_TAG}}"
TRAIN_DIR=$CKPT/$CONFIG/$EXP
MERGED=$CKPT/${CONFIG}_merged_${EXP}
ETAG="sawyer_fulllora_${INIT}init_t${TASK_ID}_${DATA_TAG}_s${SEED}"

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export OPENPI_XEMB_ROOT=$XEMB
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

mkdir -p "$LOGDIR"
SUMMARY=$LOGDIR/pi05_sawyer_fulllora_summary.log
LOG=$LOGDIR/pi05_sawyer_fulllora_${INIT}init.nohup.log
touch "$SUMMARY" "$LOG"
log() { echo "[$(date -Iseconds)] [sawyer-fulllora-${INIT}] $*" | tee -a "$LOG" "$SUMMARY" >&2; }

patch_repo() {
  local cfg=$XEMB/src/openpi/training/config.py
  sed -i "s|\"b_sawyer\": \"[^\"]*\"|\"b_sawyer\": \"${REPO}\"|" "$cfg"
  grep -n '"b_sawyer":' "$cfg" | head -2 | tee -a "$LOG" >/dev/null
}

train_full() {
  [[ -f "$INIT_CKPT/model.safetensors" ]] || { log "FATAL missing INIT_CKPT=$INIT_CKPT"; exit 1; }
  [[ -d "$REPO" ]] || { log "FATAL missing REPO=$REPO"; exit 1; }
  if [[ -f "$TRAIN_DIR/$STEP/model.safetensors" ]]; then
    log "SKIP train (have $TRAIN_DIR/$STEP)"
    return 0
  fi

  patch_repo
  log "TRAIN full-LoRA init=$INIT_CKPT repo=$REPO exp=$EXP"

  cd "$XEMB"
  if [[ ! -f "$REPO/norm_stats.json" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_TRAIN%%,*}"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG" \
      2>&1 | tee -a "$LOGDIR/compute_norm_stats_${ETAG}.log"
  fi

  export CUDA_VISIBLE_DEVICES="$GPU_TRAIN"
  local resume_flag=()
  if [[ -d "$TRAIN_DIR" ]] && ls "$TRAIN_DIR"/*/model.safetensors >/dev/null 2>&1; then
    resume_flag=(--resume)
    log "RESUME under $TRAIN_DIR"
  fi

  if ! uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/train_pytorch_lora.py "$CONFIG" \
    --exp-name "$EXP" \
    --batch-size "$BS" \
    --num-train-steps "$STEP" \
    --save-interval 1000 \
    --no-wandb-enabled \
    --pytorch-weight-path "$INIT_CKPT" \
    "${resume_flag[@]}" \
    2>&1 | tee -a "$LOGDIR/train_${ETAG}.log"; then
    log "WARN train CLI may have failed; check log"
  fi

  [[ -f "$TRAIN_DIR/$STEP/model.safetensors" ]] || { log "FATAL missing $TRAIN_DIR/$STEP"; exit 1; }
  log "DONE train -> $TRAIN_DIR/$STEP"
}

merge_full() {
  if [[ -f "$MERGED/model.safetensors" ]]; then
    log "SKIP merge exists $(basename "$MERGED")"
    return 0
  fi
  log "MERGE -> $(basename "$MERGED")"
  cd "$XEMB"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  uv run scripts/merge_lora_checkpoint.py \
    --config-name "$CONFIG" \
    --lora-checkpoint-dir "$TRAIN_DIR/$STEP" \
    --output-dir "$MERGED" \
    --base-weights-dir "$INIT_CKPT" \
    2>&1 | tee -a "$LOGDIR/merge_${ETAG}.log"
  [[ -f "$MERGED/model.safetensors" ]] || { log "FATAL merge failed"; exit 1; }
  log "DONE merge"
}

eval_sawyer() {
  local elog=$EVAL/eval_${ETAG}.log
  local slog=$EVAL/serve_${ETAG}.log
  local vdir=$EVAL/videos_${ETAG}

  if grep -q SUCCESS_RATE "$elog" 2>/dev/null; then
    log "SKIP eval :: $(grep SUCCESS_RATE "$elog" | tail -1)"
    return 0
  fi
  [[ -f "$MERGED/model.safetensors" ]] || { log "FATAL need $MERGED"; exit 1; }

  mkdir -p "$vdir"
  log "EVAL merged=$(basename "$MERGED") port=$PORT"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
  sleep 2
  cd "$XEMB"
  uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$MERGED" \
    >"$slog" 2>&1 &
  local spid=$!
  for _ in $(seq 1 180); do
    grep -q "server listening" "$slog" 2>/dev/null && break
    sleep 2
  done
  if ! grep -q "server listening" "$slog" 2>/dev/null; then
    log "FATAL serve failed"
    kill "$spid" 2>/dev/null || true
    tail -40 "$slog" | tee -a "$SUMMARY"
    return 1
  fi
  (
    cd "$CLIENT"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL
    export LIBERO_CONFIG_PATH=$EVAL/libero_config
    python "$EVAL/eval_libero_sawyer.py" \
      --args.host localhost --args.port "$PORT" \
      --args.task-suite-name libero_object --args.task-ids "$TASK_ID" \
      --args.num-trials-per-task "$NUM_TRIALS" --args.seed "$SEED" \
      --args.robot-variant sawyer --args.sawyer-home-w30 \
      --args.control-freq "$CONTROL_FREQ" \
      --args.video-out-path "$vdir" \
      >"$elog" 2>&1
  )
  kill "$spid" 2>/dev/null || true
  pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
  local rate
  rate=$(grep SUCCESS_RATE "$elog" | tail -1 || true)
  log "DONE eval $rate"
  echo "tag=${ETAG} $rate" >>"$SUMMARY"
}

log "START ONLY=$ONLY INIT=$INIT TASK_ID=$TASK_ID STEP=$STEP REPO=$REPO INIT_CKPT=$INIT_CKPT"
if [[ "$ONLY" == "train" || "$ONLY" == "all" ]]; then
  train_full
  merge_full
fi
if [[ "$ONLY" == "eval" || "$ONLY" == "all" ]]; then
  eval_sawyer
fi
log "ALL DONE — see $SUMMARY"
