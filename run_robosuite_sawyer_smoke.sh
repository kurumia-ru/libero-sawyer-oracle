#!/usr/bin/env bash
# Sawyer smoke: robosuite Lift (standard Sawyer) + LIBERO (SawyerCustom).
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT="${OPENPI_ROOT:-$HOME/openpi_ws}"
EVAL="${OPENPI_EVAL_ROOT:-$ROOT/openpi_eval}"
OUT="$EVAL/robosuite_sawyer_smoke"

export MUJOCO_GL=egl
export WANDB_DISABLED=true
export OPENPI_LIBERO_ROOT="${OPENPI_LIBERO_ROOT:-$ROOT/openpi/third_party/libero}"
export OPENPI_EVAL_ROOT="$EVAL"
export OPENPI_XEMB_ROOT="${OPENPI_XEMB_ROOT:-$ROOT/openpi-xemb}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$HOME/openpi_data}"
export LIBERO_CONFIG_PATH="$EVAL/libero_config"

mkdir -p "$OUT" "$EVAL/logs_sawyer_oracle"

cd "${OPENPI_XEMB_ROOT:-$ROOT/openpi-xemb}"
export PYTHONPATH="$OPENPI_LIBERO_ROOT:$OPENPI_XEMB_ROOT/src:$EVAL:${PYTHONPATH:-}"

echo "[smoke] robosuite + libero Sawyer -> $OUT"
uv run python "$EVAL/robosuite_sawyer_smoke.py" \
  --mode both \
  --env-name Lift \
  --robot Sawyer \
  --libero-suite object \
  --libero-task-id 0 \
  --libero-robot-variant sawyer \
  --out-dir "$OUT" \
  2>&1 | tee "$EVAL/logs_sawyer_oracle/robosuite_sawyer_smoke.log"

echo "DONE — summary: $OUT/summary.json"
