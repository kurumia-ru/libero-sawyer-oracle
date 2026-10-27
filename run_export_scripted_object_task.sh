#!/usr/bin/env bash
# Export scripted Sawyer demos for one object-suite task (many inits).
# Default: ik_set live success -> abs_eef next-state -> hybrid IK open-loop verified.
# Usage: TASK_ID=1 MAX_INITS=250 bash run_export_scripted_object_task.sh
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

TASK_ID="${TASK_ID:?set TASK_ID}"
MAX_INITS="${MAX_INITS:-250}"
GRASP_MODE="${GRASP_MODE:-ik_set}"
MOTION_PROFILE="${MOTION_PROFILE:-compact}"
ACTION_SPACE="${ACTION_SPACE:-abs_eef}"
SUFFIX="${SUFFIX:-_scripted_obj${TASK_ID}_abs_v1}"
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

LOG="$EVAL/logs_sawyer_oracle/scripted_export_obj${TASK_ID}_abs_eef.log"
echo "[scripted-export] object task ${TASK_ID} max_inits=${MAX_INITS} mode=${GRASP_MODE} action_space=${ACTION_SPACE} -> libero_sawyer${SUFFIX}" | tee "$LOG"
uv run python "$EVAL/export_scripted_demos.py" \
  --suite object \
  --task-ids "$TASK_ID" \
  --max-inits "$MAX_INITS" \
  --motion-profile "$MOTION_PROFILE" \
  --grasp-mode "$GRASP_MODE" \
  --action-space "$ACTION_SPACE" \
  --no-retrack-osc \
  --require-open-loop \
  --overwrite \
  --output-suffix "$SUFFIX" \
  2>&1 | tee -a "$LOG"
echo "DONE" | tee -a "$LOG"
