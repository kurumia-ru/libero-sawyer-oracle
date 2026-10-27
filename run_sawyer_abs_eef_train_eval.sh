#!/usr/bin/env bash
# After abs_eef export finishes: train LoRA then eval with abs_eef IK hybrid.
# Usage: TASK_ID=1 bash run_sawyer_abs_eef_train_eval.sh
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

TASK_ID="${TASK_ID:?set TASK_ID}"
ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data
LOG=$EVAL/logs_sawyer_oracle
REPO="${REPO:-$DATA/libero_sawyer_scripted_obj${TASK_ID}_abs_v2}"
STEP="${STEP:-30000}"
BS="${BS:-4}"
TAG="${TAG:-bs${BS}_${STEP}_scripted_obj${TASK_ID}_abs_v2}"
GPU_NORM="${GPU_NORM:-0}"
GPU_TRAIN="${GPU_TRAIN:-0}"
NPROC="${NPROC:-1}"
GPU_EVAL="${GPU_EVAL:-0}"
PORT="${PORT:-8022}"
NUM_TRIALS="${NUM_TRIALS:-10}"
SEED="${SEED:-7}"
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export LIBERO_DATA_ROOT=$DATA

mkdir -p "$LOG"
RUN_LOG="$LOG/sawyer_abs_eef_train_eval_obj${TASK_ID}.log"
log() { echo "[$(date -Iseconds)] [abs-obj${TASK_ID}] $*" | tee -a "$RUN_LOG"; }

wait_export() {
  local summary="$REPO/meta/scripted_export_summary_${TASK_ID}.json"
  log "WAIT export repo=$REPO"
  for _ in $(seq 1 7200); do
    if [[ -f "$summary" ]]; then
      local n_ep
      n_ep=$(wc -l < "$REPO/meta/episodes.jsonl" 2>/dev/null || echo 0)
      log "export done episodes=$n_ep"
      [[ "$n_ep" -ge 200 ]] || { log "FATAL too few episodes"; exit 1; }
      return 0
    fi
    # Still exporting?
    if ! pgrep -f "export_scripted_demos.py.*obj${TASK_ID}_abs_v2|export_scripted_demos.py.*task-ids ${TASK_ID}.*abs_eef" >/dev/null 2>&1; then
      if [[ -f "$REPO/meta/episodes.jsonl" ]]; then
        local n_ep
        n_ep=$(wc -l < "$REPO/meta/episodes.jsonl")
        if [[ "$n_ep" -ge 200 ]]; then
          log "export process gone but episodes=$n_ep — continue"
          return 0
        fi
      fi
      log "FATAL export not running and incomplete"
      exit 1
    fi
    sleep 30
  done
  log "FATAL timeout waiting for export"
  exit 1
}

patch_config() {
  local cfg="$XEMB/src/openpi/training/config.py"
  local target="$REPO"
  if grep -q "$target" "$cfg"; then
    log "config already points to $target"
    return 0
  fi
  log "PATCH b_sawyer repo -> $target"
  sed -i "s|$HOME/openpi_data/libero_sawyer_scripted[^\"]*\"|${target}\"|" "$cfg"
}

compute_norm_stats() {
  local config=pi05_lora_b_object_sawyer
  local stats="$REPO/norm_stats.json"
  if [[ -f "$stats" ]]; then
    log "SKIP norm_stats (exists)"
    return 0
  fi
  log "COMPUTE norm_stats gpu=$GPU_NORM"
  cd "$XEMB"
  export PYTHONPATH="$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="$GPU_NORM"
  uv run scripts/compute_norm_stats.py --config-name "$config" \
    2>&1 | tee -a "$LOG/compute_norm_stats_scripted_obj${TASK_ID}_abs.log"
  [[ -f "$stats" ]] || { log "FATAL norm_stats missing"; exit 1; }
}

run_train() {
  local config=pi05_lora_b_object_sawyer
  local exp="sawyer_obj_${TAG}"
  export CUDA_VISIBLE_DEVICES="$GPU_TRAIN"
  cd "$XEMB"
  export PYTHONPATH="$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_DISABLE=1
  log "TRAIN DDP $config exp=$exp gpus=$GPU_TRAIN"
  rm -rf "$XEMB/checkpoints/pi05_lora_b_object_sawyer/${exp}"
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/train_pytorch_lora.py "$config" \
    --exp-name "$exp" \
    --batch-size "$BS" \
    --num-train-steps "$STEP" \
    --save-interval 1000 \
    --no-wandb-enabled \
    2>&1 | tee -a "$LOG/train_sawyer_obj_${TAG}.log"
  log "DONE TRAIN"
}

merge_and_eval() {
  local config=pi05_lora_b_object_sawyer
  local exp="sawyer_obj_${TAG}"
  local lora_rel="checkpoints/pi05_lora_b_object_sawyer/${exp}/${STEP}"
  local merged_rel="checkpoints/pi05_lora_b_object_sawyer_merged_${TAG}"
  local ckpt="$XEMB/$lora_rel/model.safetensors"
  local client=$ROOT/openpi/examples/libero
  local elog="$EVAL/eval_sawyer_scripted_obj${TASK_ID}_abs.log"
  local vdir="$EVAL/videos_sawyer_scripted_obj${TASK_ID}_abs"

  [[ -f "$ckpt" ]] || { log "FATAL missing $ckpt"; exit 1; }
  if [[ ! -f "$XEMB/$merged_rel/model.safetensors" ]]; then
    log "MERGE $lora_rel -> $merged_rel"
    cd "$XEMB"
    export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
    uv run scripts/merge_lora_checkpoint.py \
      --config-name "$config" \
      --lora-checkpoint-dir "$lora_rel" \
      --output-dir "$merged_rel" \
      2>&1 | tee -a "$LOG/merge_sawyer_scripted_obj${TASK_ID}_abs.log"
  fi

  mkdir -p "$vdir"
  pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
  sleep 2
  cd "$XEMB"
  export CUDA_VISIBLE_DEVICES="$GPU_EVAL"
  uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
    policy:checkpoint --policy.config "$config" \
    --policy.dir "$XEMB/$merged_rel" > "$EVAL/serve_sawyer_scripted_obj${TASK_ID}_abs.log" 2>&1 &
  local spid=$!
  for _ in $(seq 1 90); do
    grep -q "server listening" "$EVAL/serve_sawyer_scripted_obj${TASK_ID}_abs.log" 2>/dev/null && break
    sleep 2
  done
  grep -q "server listening" "$EVAL/serve_sawyer_scripted_obj${TASK_ID}_abs.log" || {
    log "FATAL serve failed"; tail -30 "$EVAL/serve_sawyer_scripted_obj${TASK_ID}_abs.log"; exit 1
  }

  cd "$client" && source .venv/bin/activate
  export PYTHONPATH="$ROOT/openpi/third_party/libero:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export LIBERO_CONFIG_PATH="$EVAL/libero_config"
  log "EVAL abs_eef trials=$NUM_TRIALS"
  python "$EVAL/eval_libero_sawyer.py" \
    --args.host localhost --args.port "$PORT" \
    --args.task-suite-name libero_object \
    --args.task-ids "$TASK_ID" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.seed "$SEED" \
    --args.robot-variant sawyer \
    --args.sawyer-home-w30 \
    --args.sawyer-action-space abs_eef \
    --args.video-out-path "$vdir" > "$elog" 2>&1
  kill "$spid" 2>/dev/null || true
  grep "SUCCESS_RATE" "$elog" | tee -a "$RUN_LOG" || true
  log "ALL DONE"
}

main() {
  log "START abs_eef pipeline task=$TASK_ID"
  wait_export
  patch_config
  compute_norm_stats
  run_train
  merge_and_eval
}

main "$@"
