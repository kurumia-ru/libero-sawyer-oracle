#!/usr/bin/env bash
# Export successful scripted demos for object tasks 0/1 across init states.
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT="${OPENPI_ROOT:-$HOME/openpi_ws}"
EVAL="${OPENPI_EVAL_ROOT:-$ROOT/openpi_eval}"

export MUJOCO_GL=egl
export WANDB_DISABLED=true
export OPENPI_LIBERO_ROOT="${OPENPI_LIBERO_ROOT:-$ROOT/openpi/third_party/libero}"
export OPENPI_EVAL_ROOT="$EVAL"
export OPENPI_XEMB_ROOT="${OPENPI_XEMB_ROOT:-$ROOT/openpi-xemb}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$HOME/openpi_data}"
export LIBERO_CONFIG_PATH="$EVAL/libero_config"

mkdir -p "$EVAL/logs_sawyer_oracle"
cd "$OPENPI_XEMB_ROOT"
export PYTHONPATH="$OPENPI_LIBERO_ROOT:$OPENPI_XEMB_ROOT/src:$EVAL:${PYTHONPATH:-}"

echo "[scripted-export] object tasks 0,1 -> libero_sawyer_scripted_v1"
uv run python "$EVAL/export_scripted_demos.py" \
  --suite object \
  --task-ids 0 1 \
  --max-inits 50 \
  --motion-profile compact \
  --overwrite \
  --output-suffix _scripted_v1

echo "DONE"
