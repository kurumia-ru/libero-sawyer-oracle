#!/usr/bin/env bash
# Resume closefail_strict from compute_norm_stats (merge already done).
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data

GPU_TRAIN="${GPU_TRAIN:-5}"
GPU_EVAL="${GPU_EVAL:-4}"
PORT="${PORT:-8105}"
TASK_ID="${TASK_ID:-0}"

STEP="${STEP:-10000}"
BS="${BS:-4}"
NPROC="${NPROC:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"

MERGED_REPO="${MERGED_REPO:-$DATA/libero_sawyer_scripted_obj0_osc_v8_dagger_closefail_strict}"
TAG_DATA=scripted_obj0_osc_v8_dagger_closefail_strict
TRAIN_TAG="bs${BS}_${STEP}_${TAG_DATA}"

LOGDIR=$EVAL/logs_sawyer_oracle
LOG=$LOGDIR/${TAG_DATA}_resume.nohup.log
mkdir -p "$LOGDIR"

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export OPENPI_XEMB_ROOT=$XEMB
export LIBERO_DATA_ROOT=$DATA
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}

log() { echo "[$(date -Iseconds)] [obj0-closefail-strict-resume] $*" | tee -a "$LOG"; }

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
[[ -d "$MERGED_REPO" ]] || { log "FATAL missing merged repo $MERGED_REPO"; exit 1; }
n_dagger=$(python3 -c "import json; print(json.load(open('$DATA/libero_sawyer_dagger_obj0_closefail_strict/meta/info.json'))['total_episodes'])")
log "RESUME from norm stats; dagger_eps=$n_dagger merged=$MERGED_REPO"

patch_config "$MERGED_REPO"

log "compute_norm_stats"
CUDA_VISIBLE_DEVICES=$GPU_TRAIN uv run scripts/compute_norm_stats.py \
  --config-name pi05_lora_b_object_sawyer \
  2>&1 | tee -a "$LOGDIR/compute_norm_stats_${TAG_DATA}.log"

log "TRAIN start tag=$TRAIN_TAG gpu=$GPU_TRAIN"
rm -rf "$CKPT_BASE/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}"
export CUDA_VISIBLE_DEVICES=$GPU_TRAIN NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_pytorch_lora.py pi05_lora_b_object_sawyer \
  --exp-name "sawyer_obj_${TRAIN_TAG}" --batch-size "$BS" --num-train-steps "$STEP" \
  --save-interval 1000 --no-wandb-enabled \
  2>&1 | tee -a "$LOGDIR/train_sawyer_obj_${TRAIN_TAG}.log"
log "DONE TRAIN"

TRAIN_DIR="$CKPT_BASE/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}"
if [[ -d "$TRAIN_DIR/${STEP}" ]]; then
  log "PRUNE intermediate ckpts, keep ${STEP}"
  find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "${STEP}" -exec rm -rf {} +
fi

LORA="checkpoints/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}/${STEP}"
MERGED_CKPT="checkpoints/pi05_lora_b_object_sawyer_merged_${TRAIN_TAG}"
log "MERGE LoRA $LORA -> $MERGED_CKPT"
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/merge_lora_checkpoint.py \
  --config-name pi05_lora_b_object_sawyer \
  --lora-checkpoint-dir "$LORA" --output-dir "$MERGED_CKPT" \
  2>&1 | tee -a "$LOGDIR/merge_ckpt_${TAG_DATA}.log"

log "EVAL trials=$NUM_TRIALS task=$TASK_ID"
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
sleep 1
SLOG=$EVAL/serve_dagger_obj0_closefail_strict_eval.log
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
  policy:checkpoint --policy.config pi05_lora_b_object_sawyer \
  --policy.dir "$XEMB/$MERGED_CKPT" > "$SLOG" 2>&1 &
SPID=$!
for _ in $(seq 1 90); do
  grep -q "server listening" "$SLOG" 2>/dev/null && break
  sleep 2
done
grep -E "Loaded norm stats|listening|Error|Traceback" "$SLOG" | tee -a "$LOG" || true
grep -q "server listening" "$SLOG" || { log "FATAL eval serve failed"; kill $SPID 2>/dev/null || true; exit 1; }

cd "$ROOT/openpi/examples/libero"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL
VDIR=$EVAL/videos_${TAG_DATA}_eval
mkdir -p "$VDIR"
python "$EVAL/eval_libero_sawyer.py" \
  --args.host localhost --args.port "$PORT" \
  --args.task-suite-name libero_object --args.task-ids "$TASK_ID" \
  --args.num-trials-per-task "$NUM_TRIALS" --args.seed 7 \
  --args.robot-variant sawyer --args.sawyer-home-w30 \
  --args.control-freq 20 \
  --args.video-out-path "$VDIR" \
  > "$EVAL/eval_${TAG_DATA}.log" 2>&1 || true
kill "$SPID" 2>/dev/null || true
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
grep -E "SUCCESS_RATE|STAGE_RATES" "$EVAL/eval_${TAG_DATA}.log" | tee -a "$LOG" || true
log "ALL DONE dagger_eps=$n_dagger train=$TRAIN_TAG"
echo DAGGER_OBJ0_CLOSEFAIL_STRICT_DONE
