#!/usr/bin/env bash
# Re-evaluate the existing libero-init Full-LoRA checkpoint WITH --args.sawyer-home-w30.
#
# The earlier libero-init evals omitted that flag, so task 0's
# {"force_gripper_open": True, "start_pose_strategy": "home_w30"} preset never ran.
# Training data (export_scripted_demos.py) is generated with it, and the gripper
# state dims have std ~6e-4, so the mismatch lands far outside the training range.
set -euo pipefail
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

ROOT=$HOME/openpi_ws
XEMB=$ROOT/openpi-xemb
EVAL=$ROOT/openpi_eval
CLIENT=$ROOT/openpi/examples/libero
DATA=$HOME/openpi_data
LOGDIR=$EVAL/logs_sawyer_oracle

CONFIG=pi05_lora_b_object_sawyer
MERGED=$XEMB/checkpoints/${CONFIG}_merged_libero_init_sawyer_obj0_fulllora_s10000_scripted_obj0_osc_v8
REPO=$DATA/libero_sawyer_scripted_obj0_osc_v8

GPU_EVAL="${GPU_EVAL:-5}"
PORT="${PORT:-8862}"
TASK_ID=0
SEED=7
NUM_TRIALS="${NUM_TRIALS:-50}"
CONTROL_FREQ=20

ETAG=sawyer_fulllora_liberoinit_t0_osc_v8_s7_homew30
ELOG=$EVAL/eval_${ETAG}.log
SLOG=$EVAL/serve_${ETAG}.log
VDIR=$EVAL/videos_${ETAG}

export WANDB_DISABLED=true
export MUJOCO_GL=egl
export OPENPI_LIBERO_ROOT=$ROOT/openpi/third_party/libero
export OPENPI_EVAL_ROOT=$EVAL
export OPENPI_XEMB_ROOT=$XEMB
export LIBERO_CONFIG_PATH=$EVAL/libero_config
export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL

mkdir -p "$LOGDIR" "$VDIR"
echo "[reeval] merged=$MERGED"
[[ -f "$MERGED/model.safetensors" ]] || { echo "FATAL missing $MERGED"; exit 1; }
[[ -d "$REPO" ]] || { echo "FATAL missing $REPO"; exit 1; }

# Norm stats are read from the repo the config points at.
sed -i "s|\"b_sawyer\": \"[^\"]*\"|\"b_sawyer\": \"${REPO}\"|" "$XEMB/src/openpi/training/config.py"
grep -n '"b_sawyer":' "$XEMB/src/openpi/training/config.py" | head -1

cd "$XEMB"
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
sleep 2
CUDA_VISIBLE_DEVICES=$GPU_EVAL uv run scripts/serve_policy.py --env LIBERO --port "$PORT" \
  policy:checkpoint --policy.config "$CONFIG" --policy.dir "$MERGED" >"$SLOG" 2>&1 &
SPID=$!
for _ in $(seq 1 180); do
  grep -q "server listening" "$SLOG" 2>/dev/null && break
  sleep 2
done
grep -E "Loaded norm stats|server listening|Error|Traceback" "$SLOG" || true
grep -q "server listening" "$SLOG" || { echo "FATAL serve failed"; tail -30 "$SLOG"; kill $SPID 2>/dev/null || true; exit 1; }

(
  cd "$CLIENT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PYTHONPATH=$OPENPI_LIBERO_ROOT:$XEMB/src:$EVAL
  export LIBERO_CONFIG_PATH=$EVAL/libero_config
  python "$EVAL/eval_libero_sawyer.py" \
    --args.host localhost --args.port "$PORT" \
    --args.task-suite-name libero_object --args.task-ids "$TASK_ID" \
    --args.num-trials-per-task "$NUM_TRIALS" --args.seed "$SEED" \
    --args.robot-variant sawyer --args.sawyer-home-w30 \
    --args.control-freq "$CONTROL_FREQ" \
    --args.video-out-path "$VDIR" \
    >"$ELOG" 2>&1
)
kill "$SPID" 2>/dev/null || true
pkill -f "scripts/serve_policy.py --env LIBERO --port ${PORT}" 2>/dev/null || true
grep -E "SUCCESS_RATE|STAGE_RATES" "$ELOG" | tail -3 || tail -20 "$ELOG"
echo "REEVAL_HOMEW30_DONE etag=$ETAG"
