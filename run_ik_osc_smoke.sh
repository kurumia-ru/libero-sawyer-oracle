#!/usr/bin/env bash
# Smoke: ik_set success -> abs_eef (next-state) -> hybrid IK open-loop filter.
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT="${OPENPI_ROOT:-$HOME/openpi_ws}"
EVAL="${OPENPI_EVAL_ROOT:-$ROOT/openpi_eval}"
XEMB="${OPENPI_XEMB_ROOT:-$ROOT/openpi-xemb}"
TASK_ID="${TASK_ID:-1}"
MAX_INITS="${MAX_INITS:-3}"
SUFFIX="${SUFFIX:-_scripted_obj${TASK_ID}_abs_smoke}"

export MUJOCO_GL=egl
export WANDB_DISABLED=true
export OPENPI_LIBERO_ROOT="${OPENPI_LIBERO_ROOT:-$ROOT/openpi/third_party/libero}"
export OPENPI_EVAL_ROOT="$EVAL"
export OPENPI_XEMB_ROOT="$XEMB"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$HOME/openpi_data}"
export LIBERO_CONFIG_PATH="$EVAL/libero_config"
export PYTHONPATH="$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL:${PYTHONPATH:-}"

mkdir -p "$EVAL/logs_sawyer_oracle"
cd "$XEMB"
LOG="$EVAL/logs_sawyer_oracle/abs_eef_smoke_obj${TASK_ID}.log"
echo "[abs-eef-smoke] task=${TASK_ID} inits=${MAX_INITS} (ik_set + abs_eef IK hybrid)" | tee "$LOG"
uv run python "$EVAL/export_scripted_demos.py" \
  --suite object \
  --task-ids "$TASK_ID" \
  --max-inits "$MAX_INITS" \
  --motion-profile compact \
  --grasp-mode ik_set \
  --action-space abs_eef \
  --no-retrack-osc \
  --require-open-loop \
  --overwrite \
  --output-suffix "$SUFFIX" \
  2>&1 | tee -a "$LOG"
echo "DONE smoke" | tee -a "$LOG"
