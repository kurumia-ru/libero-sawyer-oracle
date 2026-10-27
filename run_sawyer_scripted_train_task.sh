#!/usr/bin/env bash
# Train pi05_lora_b_object_sawyer on one-task scripted repo.
# Usage: TASK_ID=1 bash run_sawyer_scripted_train_task.sh
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

TASK_ID="${TASK_ID:?set TASK_ID}"
ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data
LOG=$EVAL/logs_sawyer_oracle
REPO="$DATA/libero_sawyer_scripted_obj${TASK_ID}_v1"
STEP="${STEP:-30000}"
BS="${BS:-4}"
TAG="bs${BS}_${STEP}_scripted_obj${TASK_ID}"
GPU_NORM="${GPU_NORM:-5}"
GPU_TRAIN="${GPU_TRAIN:-5,6}"
NPROC="${NPROC:-2}"
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export LIBERO_DATA_ROOT=$DATA

mkdir -p "$LOG"
TRAIN_LOG="$LOG/sawyer_scripted_train_obj${TASK_ID}.log"

log() { echo "[$(date -Iseconds)] [train-obj${TASK_ID}] $*" | tee -a "$TRAIN_LOG"; }

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

require_repo() {
  [[ -f "$REPO/meta/episodes.jsonl" ]] || { log "FATAL missing $REPO"; exit 1; }
  local n_ep
  n_ep=$(wc -l < "$REPO/meta/episodes.jsonl")
  log "dataset episodes=$n_ep task_id=$TASK_ID"
  [[ "$n_ep" -ge 50 ]] || { log "FATAL too few episodes ($n_ep)"; exit 1; }
}

compute_norm_stats() {
  local config=pi05_lora_b_object_sawyer
  local stats="$REPO/norm_stats.json"
  if [[ -f "$stats" ]]; then
    log "SKIP norm_stats (exists): $stats"
    return 0
  fi
  log "COMPUTE norm_stats gpu=$GPU_NORM"
  cd "$XEMB"
  export PYTHONPATH="$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="$GPU_NORM"
  uv run scripts/compute_norm_stats.py --config-name "$config" \
    2>&1 | tee -a "$LOG/compute_norm_stats_scripted_obj${TASK_ID}.log"
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
  log "TRAIN DDP $config exp=$exp gpus=$GPU_TRAIN nproc=$NPROC batch=$BS"
  rm -rf "$XEMB/checkpoints/pi05_lora_b_object_sawyer/${exp}"
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/train_pytorch_lora.py "$config" \
    --exp-name "$exp" \
    --batch-size "$BS" \
    --num-train-steps "$STEP" \
    --save-interval 1000 \
    --no-wandb-enabled \
    2>&1 | tee -a "$LOG/train_sawyer_obj_${TAG}.log"
  log "DONE TRAIN -> checkpoints/pi05_lora_b_object_sawyer/${exp}/${STEP}"
}

main() {
  log "START tag=$TAG repo=$REPO"
  require_repo
  patch_config
  compute_norm_stats
  run_train
  log "ALL DONE train obj${TASK_ID}"
}

main "$@"
