#!/usr/bin/env bash
# osc_v8 for LIBERO object task0 (same recipe as obj1 osc_v8):
#   compact (no oversample) + hz20 + stronger approach detour (1.5cm x3) + clean
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data

TASK_ID="${TASK_ID:-0}"
TAG_DATA="${TAG_DATA:-scripted_obj${TASK_ID}_osc_v8}"
REPO=$DATA/libero_sawyer_${TAG_DATA}

# Cap cans: ik_set live + retract-aware retrack (OSC live is too weak for rim insert).
GRASP_MODE="${GRASP_MODE:-ik_set}"
START_POSE="${START_POSE:-home_w30}"
APPROACH_NOISE_M="${APPROACH_NOISE_M:-0.015}"
APPROACH_NOISE_COPIES="${APPROACH_NOISE_COPIES:-3}"
MAX_INITS="${MAX_INITS:-250}"
PATH_STEP_M="${PATH_STEP_M:-0.008}"
MIN_LIFT_DZ="${MIN_LIFT_DZ:-0.08}"
GRASP_REPEAT=0
PRECLOSE_REPEAT=0
PLACE_REPEAT=0
RELEASE_REPEAT=0
RELEASE_PAD_MIN="${RELEASE_PAD_MIN:-25}"

GPU_TRAIN="${GPU_TRAIN:-1}"
GPU_NORM="${GPU_NORM:-1}"
GPU_EVAL="${GPU_EVAL:-4}"
PORT="${PORT:-8100}"
STEP="${STEP:-10000}"
BS="${BS:-4}"
NPROC="${NPROC:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"
TRAIN_TAG="bs${BS}_${STEP}_${TAG_DATA}"
LOGDIR=$EVAL/logs_sawyer_oracle
LOG=$LOGDIR/${TAG_DATA}_pipeline.nohup.log

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export OPENPI_XEMB_ROOT=$XEMB
export LIBERO_DATA_ROOT=$DATA
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}

mkdir -p "$LOGDIR"
log() { echo "[$(date -Iseconds)] [osc-v8-obj${TASK_ID}] $*" | tee -a "$LOG"; }

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

log "START tag=$TAG_DATA task=$TASK_ID grasp=$GRASP_MODE start_pose=$START_POSE eef=${APPROACH_NOISE_M}x${APPROACH_NOISE_COPIES} no_appear step=$STEP gpu=$GPU_TRAIN"
cd "$XEMB"

# ----- 1) EXPORT -----
log "EXPORT start"
rm -rf "$REPO"
uv run python "$EVAL/export_scripted_demos.py" \
  --suite object --task-ids "$TASK_ID" --max-inits "$MAX_INITS" \
  --motion-profile compact --grasp-mode "$GRASP_MODE" \
  --start-pose-strategy "$START_POSE" --action-space delta_osc --retrack-osc \
  --require-open-loop --path-step-m "$PATH_STEP_M" \
  --min-lift-dz "$MIN_LIFT_DZ" \
  --grasp-repeat "$GRASP_REPEAT" \
  --preclose-repeat "$PRECLOSE_REPEAT" \
  --place-repeat "$PLACE_REPEAT" --release-repeat "$RELEASE_REPEAT" \
  --release-pad-min "$RELEASE_PAD_MIN" \
  --approach-noise-m "$APPROACH_NOISE_M" --approach-noise-copies "$APPROACH_NOISE_COPIES" \
  --overwrite --output-suffix "_${TAG_DATA}" \
  2>&1 | tee -a "$LOGDIR/export_${TAG_DATA}.log"

n_ep=$(wc -l < "$REPO/meta/episodes.jsonl")
log "EXPORT done episodes=$n_ep"
[[ "$n_ep" -ge 60 ]] || { log "FATAL too few episodes"; exit 1; }
python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
p=Path("$REPO")/"meta"/"scripted_export_summary_${TASK_ID}.json"
d=json.loads(p.read_text())
print(f"[check] fps={d.get('dataset_fps')} control={d.get('control_freq')} "
      f"motion={d.get('motion_profile')} noise={d.get('approach_noise_m')}x{d.get('approach_noise_copies')} "
      f"grasp_repeat={d.get('grasp_repeat')} exported={d.get('exported_episodes')}")
assert d.get("dataset_fps")==20 and d.get("control_freq")==20, d
assert d.get("motion_profile")=="compact", d
assert int(d.get("grasp_repeat") or 0)==0, d
assert float(d.get("approach_noise_m") or 0) >= 0.014, d
assert int(d.get("approach_noise_copies") or 0) >= 3, d
PY

patch_config "$REPO"

# ----- 2) NORM -----
if [[ ! -f "$REPO/norm_stats.json" ]]; then
  log "compute_norm_stats"
  CUDA_VISIBLE_DEVICES=$GPU_NORM uv run scripts/compute_norm_stats.py \
    --config-name pi05_lora_b_object_sawyer \
    2>&1 | tee -a "$LOGDIR/compute_norm_stats_${TAG_DATA}.log"
fi

# ----- 3) TRAIN -----
log "TRAIN start tag=$TRAIN_TAG"
rm -rf "$CKPT_BASE/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}"
export CUDA_VISIBLE_DEVICES=$GPU_TRAIN NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/train_pytorch_lora.py pi05_lora_b_object_sawyer \
  --exp-name "sawyer_obj_${TRAIN_TAG}" --batch-size "$BS" --num-train-steps "$STEP" \
  --save-interval 1000 --no-wandb-enabled \
  2>&1 | tee -a "$LOGDIR/train_sawyer_obj_${TRAIN_TAG}.log"
log "DONE TRAIN"

# ----- 4) MERGE -----
LORA="checkpoints/pi05_lora_b_object_sawyer/sawyer_obj_${TRAIN_TAG}/${STEP}"
MERGED="checkpoints/pi05_lora_b_object_sawyer_merged_${TRAIN_TAG}"
log "MERGE $LORA -> $MERGED"
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/merge_lora_checkpoint.py \
  --config-name pi05_lora_b_object_sawyer \
  --lora-checkpoint-dir "$LORA" --output-dir "$MERGED" \
  2>&1 | tee -a "$LOGDIR/merge_${TAG_DATA}.log"

# ----- 5) SMOKE -----
log "SMOKE TF/open-loop"
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
sleep 2
SLOG=$EVAL/serve_${TAG_DATA}_smoke.log
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
  policy:checkpoint --policy.config pi05_lora_b_object_sawyer \
  --policy.dir "$XEMB/$MERGED" > "$SLOG" 2>&1 &
SPID=$!
for _ in $(seq 1 90); do
  grep -q "server listening" "$SLOG" 2>/dev/null && break
  sleep 2
done
grep -E "Loaded norm stats|listening|Error" "$SLOG" | tee -a "$LOG"
grep "Loaded norm stats" "$SLOG" | grep -q "obj${TASK_ID}_osc_v8" || { log "FATAL wrong norm"; kill $SPID; exit 1; }

SMOKE_OUT=$EVAL/smoke_open_closed/sawyer_osc_v8_obj${TASK_ID}_ep0_pair
mkdir -p "$SMOKE_OUT"
REPO="$REPO" PORT="$PORT" ACTION_SPACE=delta_osc ROBOT=sawyer EP=0 TASK_ID="$TASK_ID" \
OUT_DIR="$SMOKE_OUT" N_STEPS=150 \
  bash "$EVAL/run_smoke_open_closed_traj.sh" \
  2>&1 | tee -a "$LOGDIR/smoke_${TAG_DATA}.log"
python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
p=Path("$SMOKE_OUT")/"ep0_init0_summary.json"
m=json.loads(p.read_text())["metrics"]
keys=["teacher_force_action_l2_mean","open_loop_policy_vs_replay_joint_l2_mean","verdict"]
print("[SMOKE]", {k:m.get(k) for k in keys})
PY

# ----- 6) STAGE EVAL -----
log "EVAL trials=$NUM_TRIALS task=$TASK_ID"
cd "$ROOT/openpi/examples/libero"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL
VDIR=$EVAL/videos_${TAG_DATA}
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
grep -E "SUCCESS_RATE|STAGE_RATES" "$EVAL/eval_${TAG_DATA}.log" | tee -a "$LOG"
log "ALL DONE episodes=$n_ep"
echo OSC_V8_OBJ${TASK_ID}_DONE
