#!/usr/bin/env bash
# Train pi05_lora_b_object_sawyer on libero_sawyer_scripted_v1 (500 object ep).
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
DATA=$HOME/openpi_data
LOG=$EVAL/logs_sawyer_oracle
REPO="$DATA/libero_sawyer_scripted_v1"
STEP="${STEP:-30000}"
BS="${BS:-4}"
TAG="bs${BS}_${STEP}_scripted_v1"
# norm_stats: single GPU; train: DDP on two GPUs (total batch = BS).
GPU_NORM="${GPU_NORM:-5}"
GPU_TRAIN="${GPU_TRAIN:-5,6}"
NPROC="${NPROC:-2}"
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export LIBERO_DATA_ROOT=$DATA

mkdir -p "$LOG"

log() { echo "[$(date -Iseconds)] [scripted-train] $*" | tee -a "$LOG/sawyer_scripted_train.log"; }

patch_config_scripted() {
  local cfg="$XEMB/src/openpi/training/config.py"
  if grep -q 'libero_sawyer_scripted_v1' "$cfg"; then
    log "config already points to libero_sawyer_scripted_v1"
    return 0
  fi
  log "PATCH b_sawyer repo -> libero_sawyer_scripted_v1"
  sed -i 's|libero_b_sawyer_v2"|libero_sawyer_scripted_v1"|' "$cfg"
}

require_repo() {
  [[ -f "$REPO/meta/episodes.jsonl" ]] || { log "FATAL missing $REPO"; exit 1; }
  local n_ep
  n_ep=$(wc -l < "$REPO/meta/episodes.jsonl")
  log "dataset episodes=$n_ep"
  [[ "$n_ep" -eq 500 ]] || log "WARN expected 500 episodes, got $n_ep"
}

compute_norm_stats() {
  local config=pi05_lora_b_object_sawyer
  local stats="$REPO/norm_stats.json"
  if [[ -f "$stats" ]]; then
    log "SKIP norm_stats (exists): $stats"
    return 0
  fi
  log "COMPUTE norm_stats for $config gpu=$GPU_NORM"
  cd "$XEMB"
  export PYTHONPATH="$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="$GPU_NORM"
  uv run scripts/compute_norm_stats.py --config-name "$config" \
    2>&1 | tee -a "$LOG/compute_norm_stats_scripted_v1.log"
  if [[ ! -f "$stats" ]] && [[ -f "$REPO/norm_stats.json" ]]; then
    log "norm_stats at $stats"
  elif [[ ! -f "$stats" ]]; then
    local alt="$XEMB/assets/${DATA#/}/libero_sawyer_scripted_v1/norm_stats.json"
    if [[ -f "$alt" ]]; then
      cp "$alt" "$stats"
      log "copied norm_stats from assets"
    fi
  fi
  [[ -f "$stats" ]] || { log "FATAL norm_stats missing after compute"; exit 1; }
}

run_train() {
  local config=pi05_lora_b_object_sawyer
  local exp="sawyer_obj_${TAG}"
  export CUDA_VISIBLE_DEVICES="$GPU_TRAIN"
  cd "$XEMB"
  export PYTHONPATH="$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}"
  if [[ "$NPROC" -gt 1 ]]; then
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    log "TRAIN DDP $config exp=$exp gpus=$GPU_TRAIN nproc=$NPROC total_batch=$BS"
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
      scripts/train_pytorch_lora.py "$config" \
      --exp-name "$exp" \
      --batch-size "$BS" \
      --num-train-steps "$STEP" \
      --save-interval 1000 \
      --no-wandb-enabled \
      2>&1 | tee -a "$LOG/train_sawyer_obj_${TAG}.log"
  else
    log "TRAIN single-GPU $config exp=$exp gpu=$GPU_TRAIN batch=$BS"
    uv run scripts/train_pytorch_lora.py "$config" \
      --exp-name "$exp" \
      --batch-size "$BS" \
      --num-train-steps "$STEP" \
      --save-interval 1000 \
      --no-wandb-enabled \
      2>&1 | tee -a "$LOG/train_sawyer_obj_${TAG}.log"
  fi
  log "DONE TRAIN $config -> checkpoints/pi05_lora_b_object_sawyer/${exp}/${STEP}"
}

main() {
  log "START scripted v1 norm_gpu=$GPU_NORM train_gpus=$GPU_TRAIN nproc=$NPROC tag=$TAG"
  require_repo
  patch_config_scripted
  compute_norm_stats
  rm -rf "$XEMB/checkpoints/pi05_lora_b_object_sawyer/sawyer_obj_${TAG}"
  run_train
  log "ALL DONE scripted v1 BO train"
}

main "$@"
