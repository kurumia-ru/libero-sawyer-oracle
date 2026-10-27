#!/usr/bin/env bash
# Close-fail DAgger: takeover only after near + close-cmd without lift,
# expert continues from student pose (no rise/clear). Student = early (best so far).
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data

GPU_COLLECT="${GPU_COLLECT:-0}"
GPU_TRAIN="${GPU_TRAIN:-1}"
GPU_EVAL="${GPU_EVAL:-0}"
PORT="${PORT:-8098}"

N_EP="${N_EP:-60}"
INIT0="${INIT0:-40}"
NEAR_THRESH="${NEAR_THRESH:-0.08}"
# Linger after first close cmd while still near (no lift).
LINGER_STEPS="${LINGER_STEPS:-20}"
HARD_T="${HARD_T:-110}"
APPROACH_STALL_STEPS="${APPROACH_STALL_STEPS:-0}"
APPROACH_ACTIVE_THRESH="${APPROACH_ACTIVE_THRESH:-0.22}"
CLOSE_CMD_THRESH="${CLOSE_CMD_THRESH:-0.2}"
# Thicken reopen/slide/descend→lift contact labels
GRASP_REPEAT="${GRASP_REPEAT:-4}"
HEAD_FRAMES="${HEAD_FRAMES:-0}"
HEAD_REPEAT="${HEAD_REPEAT:-1}"
DAGGER_REPEAT="${DAGGER_REPEAT:-10}"

STEP="${STEP:-10000}"
BS="${BS:-4}"
NPROC="${NPROC:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"

STUDENT_MERGED="${STUDENT_MERGED:-checkpoints/pi05_lora_b_object_sawyer_merged_bs4_10000_scripted_obj1_osc_v8_dagger_r2_early}"
STUDENT_ASSETS="${STUDENT_ASSETS:-$DATA/libero_sawyer_scripted_obj1_osc_v8_dagger_r2_early}"
BASE_REPO="${BASE_REPO:-$DATA/libero_sawyer_scripted_obj1_osc_v8_dagger_r2_early}"
DAGGER_REPO="${DAGGER_REPO:-$DATA/libero_sawyer_dagger_r2_closefail}"
MERGED_REPO="${MERGED_REPO:-$DATA/libero_sawyer_scripted_obj1_osc_v8_dagger_r2_closefail}"
VIDEO_DIR="${VIDEO_DIR:-$EVAL/videos_dagger_r2_closefail}"
TAG_DATA=scripted_obj1_osc_v8_dagger_r2_closefail
TRAIN_TAG="bs${BS}_${STEP}_${TAG_DATA}"

LOGDIR=$EVAL/logs_sawyer_oracle
LOG=$LOGDIR/${TAG_DATA}_pipeline.nohup.log
mkdir -p "$LOGDIR" "$VIDEO_DIR"

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export OPENPI_XEMB_ROOT=$XEMB
export LIBERO_DATA_ROOT=$DATA
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}

log() { echo "[$(date -Iseconds)] [dagger-closefail] $*" | tee -a "$LOG"; }

patch_config() {
  local target="$1"
  local cfg="$XEMB/src/openpi/training/config.py"
  log "PATCH b_sawyer repo -> $target"
  python3 - <<PY
from pathlib import Path
import re
cfg = Path("$cfg")
text = cfg.read_text()
text2 = re.sub(
    r'("b_sawyer"\s*:\s*)"[^"]+"',
    r'\1"$target"',
    text,
)
cfg.write_text(text2)
m = re.search(r'"b_sawyer":\s*"[^"]+"', text2)
print("[patch]", m.group(0) if m else "missing")
PY
}

cd "$XEMB"
log "START closefail n=$N_EP init0=$INIT0 linger=$LINGER_STEPS near=$NEAR_THRESH close_thresh=$CLOSE_CMD_THRESH grasp_rep=$GRASP_REPEAT ep_rep=$DAGGER_REPEAT"

patch_config "$STUDENT_ASSETS"
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
sleep 1

log "SERVE student=$STUDENT_MERGED port=$PORT gpu=$GPU_COLLECT"
SLOG=$EVAL/serve_dagger_r2_closefail_collect.log
CUDA_VISIBLE_DEVICES=$GPU_COLLECT uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
  policy:checkpoint --policy.config pi05_lora_b_object_sawyer \
  --policy.dir "$XEMB/$STUDENT_MERGED" > "$SLOG" 2>&1 &
SPID=$!
for _ in $(seq 1 90); do
  grep -q "server listening" "$SLOG" 2>/dev/null && break
  sleep 2
done
grep -E "Loaded norm stats|listening|Error|Traceback" "$SLOG" | tee -a "$LOG" || true
grep -q "server listening" "$SLOG" || { log "FATAL serve failed"; kill $SPID 2>/dev/null || true; exit 1; }

log "COLLECT -> $DAGGER_REPO (require-close-fail + keep-student-pose)"
CUDA_VISIBLE_DEVICES=$GPU_COLLECT PYTHONUNBUFFERED=1 uv run python "$EVAL/collect_dagger_takeover.py" \
  --host 127.0.0.1 --port "$PORT" \
  --num-episodes "$N_EP" --init-start "$INIT0" \
  --near-thresh "$NEAR_THRESH" --linger-steps "$LINGER_STEPS" --hard-t "$HARD_T" \
  --approach-stall-steps "$APPROACH_STALL_STEPS" \
  --approach-active-thresh "$APPROACH_ACTIVE_THRESH" \
  --require-close-fail --keep-student-pose \
  --close-cmd-thresh "$CLOSE_CMD_THRESH" \
  --head-frames "$HEAD_FRAMES" --head-repeat "$HEAD_REPEAT" \
  --grasp-repeat "$GRASP_REPEAT" \
  --out-repo "$DAGGER_REPO" --video-dir "$VIDEO_DIR" --overwrite \
  2>&1 | tee -a "$LOGDIR/dagger_r2_closefail_collect.log"

kill "$SPID" 2>/dev/null || true
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true

n_dagger=$(python3 -c "import json; print(json.load(open('$DAGGER_REPO/meta/info.json'))['total_episodes'])")
log "COLLECT done episodes=$n_dagger"
[[ "$n_dagger" -ge 10 ]] || { log "FATAL too few dagger episodes ($n_dagger)"; exit 1; }

log "MERGE base=$BASE_REPO + closefail dagger x$DAGGER_REPEAT -> $MERGED_REPO"
CUDA_VISIBLE_DEVICES=$GPU_TRAIN PYTHONUNBUFFERED=1 uv run python "$EVAL/merge_lerobot_repos.py" \
  --base "$BASE_REPO" --extra "$DAGGER_REPO" --out "$MERGED_REPO" \
  --extra-repeat "$DAGGER_REPEAT" --require-expert-success --overwrite \
  2>&1 | tee -a "$LOGDIR/dagger_r2_closefail_merge.log"

patch_config "$MERGED_REPO"

log "compute_norm_stats"
CUDA_VISIBLE_DEVICES=$GPU_TRAIN uv run scripts/compute_norm_stats.py \
  --config-name pi05_lora_b_object_sawyer \
  2>&1 | tee -a "$LOGDIR/compute_norm_stats_${TAG_DATA}.log"

log "TRAIN start tag=$TRAIN_TAG gpu=$GPU_TRAIN"
rm -rf "$XEMB/checkpoints/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}"
export CUDA_VISIBLE_DEVICES=$GPU_TRAIN NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_pytorch_lora.py pi05_lora_b_object_sawyer \
  --exp-name "sawyer_obj_${TRAIN_TAG}" --batch-size "$BS" --num-train-steps "$STEP" \
  --save-interval 1000 --no-wandb-enabled \
  2>&1 | tee -a "$LOGDIR/train_sawyer_obj_${TRAIN_TAG}.log"
log "DONE TRAIN"

LORA="checkpoints/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}/${STEP}"
MERGED_CKPT="checkpoints/pi05_lora_b_object_sawyer_merged_${TRAIN_TAG}"
log "MERGE LoRA $LORA -> $MERGED_CKPT"
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/merge_lora_checkpoint.py \
  --config-name pi05_lora_b_object_sawyer \
  --lora-checkpoint-dir "$LORA" --output-dir "$MERGED_CKPT" \
  2>&1 | tee -a "$LOGDIR/merge_ckpt_${TAG_DATA}.log"

log "EVAL trials=$NUM_TRIALS"
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
sleep 1
SLOG=$EVAL/serve_dagger_r2_closefail_eval.log
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
  --args.task-suite-name libero_object --args.task-ids 1 \
  --args.num-trials-per-task "$NUM_TRIALS" --args.seed 7 \
  --args.robot-variant sawyer --args.sawyer-home-w30 \
  --args.control-freq 20 \
  --args.video-out-path "$VDIR" \
  > "$EVAL/eval_${TAG_DATA}.log" 2>&1 || true
kill "$SPID" 2>/dev/null || true
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
grep -E "SUCCESS_RATE|STAGE_RATES" "$EVAL/eval_${TAG_DATA}.log" | tee -a "$LOG" || true
log "ALL DONE dagger_eps=$n_dagger train=$TRAIN_TAG"
echo DAGGER_R2_CLOSEFAIL_DONE
