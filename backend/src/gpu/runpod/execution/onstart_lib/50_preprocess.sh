# Optional generic preprocessing step.
#
# RL cases normally generate their data online and leave `preprocess_cmd`
# empty in the case registry, in which case this step is a no-op. When a case
# does register one (e.g. building an offline-RL dataset under data/processed/),
# it runs exactly once here, from backend/, before training. Failure is fatal:
# training on a half-built dataset wastes the whole pod.
if [ -n "<PREPROCESS_CMD>" ]; then
  mark "54_preprocess_started"
  echo "[onstart] step=preprocess case=<CASE> cmd=<PREPROCESS_CMD>"
  # Long preprocessing emits no markers by itself, which would trip the 900s
  # STALL_THRESHOLD watcher. A background heartbeat keeps it informed.
  (
    n=1
    while sleep 240; do
      mark "54_preprocess_heartbeat_${n}"
      n=$((n + 1))
    done
  ) &
  PREPROCESS_HB_PID=$!
  # iter11: call the venv python directly (no `uv run`, see 40_uv_and_dvc_pull.sh).
  if ! ( cd backend && "${PY_BIN}" -m <PREPROCESS_CMD> ); then
    kill "${PREPROCESS_HB_PID}" 2>/dev/null || true
    echo "[onstart] step=preprocess FAILED (exit code != 0)" >&2
    mark "55_preprocess_failed"
    exit 1
  fi
  kill "${PREPROCESS_HB_PID}" 2>/dev/null || true
  echo "[onstart] step=preprocess done"
  mark "55_preprocess_done"
else
  echo "[onstart] step=preprocess_skip (no preprocess_cmd registered for <CASE>)"
fi

# The interactive-mode hold and the preprocess-only exit follow at the top of
# 60_train.sh.
