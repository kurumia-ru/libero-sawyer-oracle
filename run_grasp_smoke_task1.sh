#!/usr/bin/env bash
# object task 1: cream cheese (flat box) — side pinch + basket place
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT="${OPENPI_ROOT:-$HOME/openpi_ws}"
EVAL="${OPENPI_EVAL_ROOT:-$ROOT/openpi_eval}"
OUT="$EVAL/grasp_smoke/task1"

export MUJOCO_GL=egl
export WANDB_DISABLED=true
export OPENPI_LIBERO_ROOT="${OPENPI_LIBERO_ROOT:-$ROOT/openpi/third_party/libero}"
export OPENPI_EVAL_ROOT="$EVAL"
export OPENPI_XEMB_ROOT="${OPENPI_XEMB_ROOT:-$ROOT/openpi-xemb}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$HOME/openpi_data}"
export LIBERO_CONFIG_PATH="$EVAL/libero_config"

mkdir -p "$OUT"
cd "$OPENPI_XEMB_ROOT"
export PYTHONPATH="$OPENPI_LIBERO_ROOT:$OPENPI_XEMB_ROOT/src:$EVAL:${PYTHONPATH:-}"

echo "[grasp-smoke] task1 cream_cheese side+place -> $OUT"
uv run python "$EVAL/libero_grasp_smoke.py" \
  --suite object \
  --task-id 1 \
  --robots sawyer \
  --grasp-mode ik_set \
  --grasp-style side \
  --side-x 0.025 \
  --side-z 0.015 \
  --start-pose-strategy home_w30 \
  --force-gripper-open \
  --motion-profile compact \
  --place-to-basket \
  --out-dir "$OUT"

echo "DONE $OUT/summary.json"
