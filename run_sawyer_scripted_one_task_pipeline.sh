#!/usr/bin/env bash
# One task end-to-end: export -> train -> eval.
# Usage: TASK_ID=1 bash run_sawyer_scripted_one_task_pipeline.sh
#        TASK_ID=5 MAX_INITS=250 bash run_sawyer_scripted_one_task_pipeline.sh
set -euo pipefail

export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

TASK_ID="${TASK_ID:?set TASK_ID (e.g. 1 or 5)}"
MAX_INITS="${MAX_INITS:-250}"
ROOT=$HOME/openpi_ws
EVAL=$ROOT/openpi_eval
LOG=$EVAL/logs_sawyer_oracle
DRIVER="$LOG/sawyer_one_task_obj${TASK_ID}_pipeline.log"

mkdir -p "$LOG"
log() { echo "[$(date -Iseconds)] [pipeline-obj${TASK_ID}] $*" | tee -a "$DRIVER"; }

log "START export -> train -> eval (max_inits=$MAX_INITS)"

export TASK_ID MAX_INITS
bash "$EVAL/run_export_scripted_object_task.sh" 2>&1 | tee -a "$DRIVER"

export TASK_ID
bash "$EVAL/run_sawyer_scripted_train_task.sh" 2>&1 | tee -a "$DRIVER"
bash "$EVAL/run_sawyer_scripted_eval_task.sh" 2>&1 | tee -a "$DRIVER"

log "ALL DONE pipeline obj${TASK_ID}"
