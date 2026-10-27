#!/usr/bin/env bash
# Sanity: full fine-tune (NOT LoRA) on Sawyer Task0 ~50ep → home_w30 eval.
# Default init: pi05_libero (override with INIT=base).
#
# If this fails, the train/eval pipeline or data is suspect (1-task FT should work).
#
# Usage:
#   ONLY=all bash run_pi05_libero_sawyer_fullft_50ep.sh
#   INIT=libero ONLY=train GPU_TRAIN=4,5 NPROC=2 BS=2 bash run_pi05_libero_sawyer_fullft_50ep.sh
#   ONLY=eval  GPU_EVAL=4 PORT=8878 bash run_pi05_libero_sawyer_fullft_50ep.sh
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
CLIENT=$ROOT/openpi/examples/libero
DATA=$HOME/openpi_data
CKPT=$XEMB/checkpoints
LOGDIR=$EVAL/logs_sawyer_oracle
LOCK=$LOGDIR/config_b_sawyer.lock

GPU_TRAIN="${GPU_TRAIN:-4,5}"
GPU_EVAL="${GPU_EVAL:-4}"
PORT="${PORT:-8878}"
TASK_ID=0
STEP="${STEP:-5000}"
BS="${BS:-2}"
NPROC="${NPROC:-2}"
NUM_EP="${NUM_EP:-50}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED=7
CONTROL_FREQ=20
ONLY="${ONLY:-all}"
INIT="${INIT:-libero}"  # libero|base
# Same data/transforms as LoRA sawyer runs; train_pytorch.py = all params (no PEFT).
CONFIG="${CONFIG:-pi05_lora_b_object_sawyer}"

case "$INIT" in
  libero) INIT_CKPT="${INIT_CKPT:-$ROOT/openpi/checkpoints/pi05_libero_pytorch}" ;;
  base)   INIT_CKPT="${INIT_CKPT:-$ROOT/openpi/checkpoints/pi05_base_pytorch}" ;;
  *) echo "INIT must be libero|base, got $INIT" >&2; exit 1 ;;
esac

SRC_REPO=$DATA/libero_sawyer_scripted_obj0_osc_v8
REPO=$DATA/libero_sawyer_scripted_obj0_osc_v8_subset${NUM_EP}_s${SEED}
TAG=${INIT}init_fullft_${NUM_EP}ep
EXP=${INIT}_init_sawyer_obj0_fullft_s${STEP}_subset${NUM_EP}_s${SEED}
TRAIN_DIR=$CKPT/$CONFIG/$EXP
ETAG=sawyer_fullft_${INIT}init_t0_subset${NUM_EP}_s${SEED}

LOG=$LOGDIR/${TAG}_pipeline.nohup.log
mkdir -p "$LOGDIR"

export WANDB_PROJECT="${WANDB_PROJECT:-openpi-sawyer-oracle}"
# no-tty servers often lack WANDB_API_KEY; disable unless explicitly enabled.
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL OPENPI_XEMB_ROOT=$XEMB
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date -Iseconds)] [fullft-50ep] $*" | tee -a "$LOG"; }

patch_config() {
  local target="$1"
  python3 - <<PY
from pathlib import Path
import re
cfg=Path("$XEMB/src/openpi/training/config.py")
t=re.sub(r'("b_sawyer"\s*:\s*)"[^"]+"', r'\1"$target"', cfg.read_text())
cfg.write_text(t)
print(re.search(r'"b_sawyer":\s*"[^"]+"', t).group(0))
PY
}

ensure_subset() {
  if [[ -f "$REPO/meta/info.json" ]]; then
    local n
    n=$(python3 -c "import json;print(json.load(open('$REPO/meta/info.json'))['total_episodes'])")
    if [[ "$n" -eq "$NUM_EP" ]]; then
      log "SKIP subset (have $REPO ep=$n)"
      return 0
    fi
  fi
  log "SUBSET first $NUM_EP from $SRC_REPO -> $REPO"
  python3 "$EVAL/subset_lerobot_episodes.py" \
    --src "$SRC_REPO" --dst "$REPO" --num-episodes "$NUM_EP" --overwrite
}

train_fullft() {
  [[ -f "$INIT_CKPT/model.safetensors" ]] || { log "FATAL missing INIT $INIT_CKPT"; exit 1; }
  [[ -d "$REPO" ]] || { log "FATAL missing REPO $REPO"; exit 1; }
  if [[ -f "$TRAIN_DIR/$STEP/model.safetensors" ]]; then
    log "SKIP train (have $TRAIN_DIR/$STEP)"
    return 0
  fi

  (
    flock -w 7200 9 || { log "FATAL flock"; exit 1; }
    patch_config "$REPO"
    # Reuse parent norm_stats if present; else compute on subset.
    if [[ ! -f "$REPO/norm_stats.json" ]]; then
      log "compute_norm_stats"
      CUDA_VISIBLE_DEVICES="${GPU_TRAIN%%,*}" uv run scripts/compute_norm_stats.py --config-name "$CONFIG" \
        2>&1 | tee -a "$LOGDIR/compute_norm_stats_${TAG}.log"
    else
      log "reuse norm_stats $REPO/norm_stats.json"
    fi

    log "TRAIN full-FT init=$INIT_CKPT repo=$REPO exp=$EXP bs=$BS nproc=$NPROC step=$STEP"
    cd "$XEMB"
    export CUDA_VISIBLE_DEVICES="$GPU_TRAIN"
    local resume_flag=()
    if [[ -d "$TRAIN_DIR" ]] && ls "$TRAIN_DIR"/*/model.safetensors >/dev/null 2>&1; then
      resume_flag=(--resume)
      log "RESUME under $TRAIN_DIR"
    fi
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
      scripts/train_pytorch.py "$CONFIG" \
      --exp-name "$EXP" \
      --batch-size "$BS" \
      --num-train-steps "$STEP" \
      --save-interval 1000 \
      --no-wandb-enabled \
      --pytorch-weight-path "$INIT_CKPT" \
      "${resume_flag[@]}" \
      2>&1 | tee -a "$LOGDIR/train_${ETAG}.log"
  ) 9>"$LOCK"

  [[ -f "$TRAIN_DIR/$STEP/model.safetensors" ]] || { log "FATAL missing $TRAIN_DIR/$STEP"; exit 1; }
  log "DONE train -> $TRAIN_DIR/$STEP"
}

eval_sawyer() {
  local elog=$EVAL/eval_${ETAG}.log
  local slog=$EVAL/serve_${ETAG}.log
  local vdir=$EVAL/videos_${ETAG}

  if grep -q SUCCESS_RATE "$elog" 2>/dev/null; then
    log "SKIP eval :: $(grep SUCCESS_RATE "$elog" | tail -1)"
    return 0
  fi
  [[ -f "$TRAIN_DIR/$STEP/model.safetensors" ]] || { log "FATAL need $TRAIN_DIR/$STEP"; exit 1; }

  (
    flock -w 7200 9 || { log "FATAL flock"; exit 1; }
    patch_config "$REPO"
  ) 9>"$LOCK"

  mkdir -p "$vdir"
  log "EVAL ckpt=$TRAIN_DIR/$STEP port=$PORT (no LoRA merge)"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
  sleep 2
  cd "$XEMB"
  uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$TRAIN_DIR/$STEP" \
    >"$slog" 2>&1 &
  local spid=$!
  for _ in $(seq 1 180); do
    grep -q "server listening" "$slog" 2>/dev/null && break
    sleep 2
  done
  if ! grep -q "server listening" "$slog" 2>/dev/null; then
    log "FATAL serve failed"
    kill "$spid" 2>/dev/null || true
    tail -50 "$slog" | tee -a "$LOG"
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
  echo "tag=${ETAG} $rate" >>"$LOGDIR/pi05_sawyer_fullft_summary.log"
}

log "START ONLY=$ONLY STEP=$STEP BS=$BS NPROC=$NPROC NUM_EP=$NUM_EP REPO=$REPO"
ensure_subset
cd "$XEMB"
if [[ "$ONLY" == "train" || "$ONLY" == "all" ]]; then
  train_fullft
fi
if [[ "$ONLY" == "eval" || "$ONLY" == "all" ]]; then
  eval_sawyer
fi
log "ALL DONE"
